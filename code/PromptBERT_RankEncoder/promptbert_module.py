import apex
import re
import sys
import io, os
import faiss
import csv
import math
import json
import torch
import numpy as np
import logging
import tqdm
import time
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from torch.nn.functional import normalize
from tqdm import tqdm
# Set up logger

class PromptBERTEncoder:
    def __init__(self, args, gpu_ids, device):
        self.args = args

        # Load transformers' model checkpoint
        if self.args.mask_embedding_sentence_org_mlp:
            #only for bert-base
            from transformers import BertForMaskedLM, BertConfig
            self.config = BertConfig.from_pretrained("bert-base-uncased")
            self.mlp = BertForMaskedLM.from_pretrained('bert-base-uncased', config=self.config).cls.predictions.transform
            if 'result' in self.args.model_name_or_path:
                state_dict = torch.load(self.args.model_name_or_path+'/pytorch_model.bin')
                new_state_dict = {}
                for key, param in state_dict.items():
                    # Replace "mlp" to "pooler"
                    if 'pooler' in key:
                        key = key.replace("pooler.", "")
                        new_state_dict[key] = param
                self.mlp.load_state_dict(new_state_dict)
        self.model = AutoModel.from_pretrained(self.args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=True)

        if self.args.mask_embedding_sentence_autoprompt:
            state_dict = torch.load(self.args.model_name_or_path+'/pytorch_model.bin')
            self.p_mbv = state_dict['p_mbv']
            template = self.args.mask_embedding_sentence_template
            template = template.replace('*mask*', self.tokenizer.mask_token)\
                    .replace('*sep+*', '')\
                    .replace('*cls*', '').replace('*sent_0*', ' ').replace('_', ' ')
            mask_embedding_template = self.tokenizer.encode(template)
            mask_index = mask_embedding_template.index(self.tokenizer.mask_token_id)
            index_mbv = mask_embedding_template[1:mask_index] + mask_embedding_template[mask_index+1:-1]
            #mask_embedding_template = [ 101, 2023, 6251, 1997, 1000, 1000, 2965, 103, 1012, 102]
            #index_mbv = mask_embedding_template[1:7] + mask_embedding_template[8:9]

            self.dict_mbv = index_mbv
            self.fl_mbv = [i <= 3 for i, k in enumerate(index_mbv)]
            
        self.device = torch.device(device)
        n_gpu = len(gpu_ids)
        self.encoder = self.model

        if n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)
            self.encoder = self.model.module
        self.model.eval()
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #device = torch.device("cpu")
        self.model = self.model.to(self.device)
        if self.args.mask_embedding_sentence_org_mlp:
            self.mlp = self.mlp.to(self.device)

        if self.args.mask_embedding_sentence_delta:
            self.delta, self.template_len = self.get_delta(self.model, self.args.mask_embedding_sentence_template, self.tokenizer, self.device, self.args)

        if self.args.remove_continue_word:
            self.pun_remove_set = {'?', '*', '#', '´', '’', '=', '…', '|', '~', '/', '‚', '¿', '–', '»', '-', '€', '‘', '"', '(', '•', '`', '$', ':', '[', '”', '%', '£', '<', '[UNK]', ';', '“', '@', '_', '{', '^', ',', '.', '!', '™', '&', ']', '>', '\\', "'", ')', '+', '—'}
            if self.args.model_name_or_path == 'roberta-base':
                self.remove_set = {'Ġ.', 'Ġa', 'Ġthe', 'Ġin', 'a', 'Ġ, ', 'Ġis', 'Ġto', 'Ġof', 'Ġand', 'Ġon', 'Ġ\'', 's', '.', 'the', 'Ġman', '-', 'Ġwith', 'Ġfor', 'Ġat', 'Ġwoman', 'Ġare', 'Ġ"', 'Ġthat', 'Ġit', 'Ġdog', 'Ġsaid', 'Ġplaying', 'Ġwas', 'Ġas', 'Ġfrom', 'Ġ:', 'Ġyou', 'Ġan', 'i', 'Ġby'}
            else:
                self.remove_set = {".", "a", "the", "in", ",", "is", "to", "of", "and", "'", "on", "man", "-", "s", "with", "for", "\"", "at", "##s", "woman", "are", "it", "two", "that", "you", "dog", "said", "playing", "i", "an", "as", "was", "from", ":", "by", "white"}

            self.vocab = self.tokenizer.get_vocab()
    
    def get_delta(self, model, template, tokenizer, device, args):
        model.eval()

        template = template.replace('*mask*', tokenizer.mask_token)\
                        .replace('*sep+*', '')\
                        .replace('*cls*', '').replace('*sent_0*', ' ')
        # strip for roberta tokenizer
        bs_length = len(tokenizer.encode(template.split(' ')[0].replace('_', ' ').strip())) - 2 + 1
        # replace for roberta tokenizer
        batch = tokenizer([template.replace('_', ' ').strip().replace('   ', ' ')], return_tensors='pt')
        batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).to(self.device).unsqueeze(0)
        for k in batch:
            batch[k] = batch[k].repeat(512, 1).to(device)
        m_mask = batch['input_ids'] == tokenizer.mask_token_id

        with torch.no_grad():
            outputs = self.model(**batch,  output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]
            delta = last_hidden[m_mask]
        delta.requires_grad = False
        #import pdb;pdb.set_trace()
        template_len = batch['input_ids'].shape[1]
        return delta, template_len

    def batcher(self, batch, max_length=None):
        sentences = batch
        if self.args.mask_embedding_sentence and self.args.mask_embedding_sentence_template is not None:
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            template = self.args.mask_embedding_sentence_template
            template = template.replace('*mask*', self.tokenizer.mask_token )\
                               .replace('_', ' ').replace('*sep+*', '')\
                               .replace('*cls*', '')
            template_tokens = self.tokenizer.encode(template.replace('*sent 0*', ""), add_special_tokens=False)
            len_template_tokens = len(template_tokens)
            
            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                s_tokens = self.tokenizer.encode(s, add_special_tokens=False)
                limit = 512 - 2 - len_template_tokens
                if limit < len(s_tokens):
                    s_tokens = s_tokens[:limit]
                s = self.tokenizer.decode(s_tokens)
                sentences[i] = template.replace('*sent 0*', s).strip()
        elif self.args.remove_continue_word:
            for i, s in enumerate(sentences):
                sentences[i] = ' ' if self.args.model_name_or_path == 'roberta-base' else ''
                es = self.tokenizer.encode(' ' + s, add_special_tokens=False)
                for iw, w in enumerate(self.tokenizer.convert_ids_to_tokens(es)):
                    if self.args.model_name_or_path == 'roberta-base':
                        # roberta base
                        if 'Ġ' not in w or w in self.remove_set:
                            pass
                        else:
                            if re.search('[a-zA-Z0-9]', w) is not None:
                                sentences[i] += w.replace('Ġ', '').lower() + ' '
                    elif w not in self.remove_set and w not in self.pun_remove_set and '##' not in w:
                        # bert base
                        sentences[i] += w.lower() + ' '
                if len(sentences[i]) == 0: sentences[i] = '[PAD]'

        if max_length is not None:
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True
            )

        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(self.device) if batch[k] is not None else None
        
        # Get raw embeddings
        with torch.no_grad():
            if self.args.embedding_only:
                hidden_states = None
                pooler_output = None
                last_hidden = self.encoder.embeddings.word_embeddings(batch['input_ids'])
                position_ids = self.encoder.embeddings.position_ids[:, 0 : last_hidden.shape[1]]
                token_type_ids = torch.zeros(batch['input_ids'].shape, dtype=torch.long,
                                                device=self.encoder.embeddings.position_ids.device)

                position_embeddings = self.encoder.embeddings.position_embeddings(position_ids)
                token_type_embeddings = self.encoder.embeddings.token_type_embeddings(token_type_ids)

                if self.args.remove_continue_word:
                    batch['attention_mask'][batch['input_ids'] == self.tokenizer.cls_token_id] = 0
                    batch['attention_mask'][batch['input_ids'] == self.tokenizer.sep_token_id] = 0
            elif self.args.mask_embedding_sentence_autoprompt:
                input_ids = batch['input_ids']
                inputs_embeds = self.encoder.embeddings.word_embeddings(input_ids)
                p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
                b = torch.arange(input_ids.shape[0]).to(input_ids.device)
                for i, k in enumerate(self.dict_mbv):
                    if self.fl_mbv[i]:
                        index = ((input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((input_ids == k) * -p).min(-1)[1]
                    inputs_embeds[b, index] = self.p_mbv[i]
                batch['input_ids'], batch['inputs_embeds'] = None, inputs_embeds
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True)
                batch['input_ids'] = input_ids

                last_hidden = outputs.last_hidden_state
                pooler_output = last_hidden[input_ids == self.tokenizer.mask_token_id]

                if self.args.mask_embedding_sentence_org_mlp:
                    pooler_output = self.mlp(pooler_output)
                if self.args.mask_embedding_sentence_delta:
                    blen = batch['attention_mask'].sum(-1) - self.template_len
                    if self.args.mask_embedding_sentence_org_mlp:
                        pooler_output -= self.mlp(self.delta[blen])
                    else:
                        pooler_output -= self.delta[blen]
                if self.args.mask_embedding_sentence_use_pooler:
                    pooler_output = self.encoder.pooler.dense(pooler_output)
                    pooler_output = self.encoder.pooler.activation(pooler_output)

            else:
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True)

                try:
                    pooler_output = outputs.pooler_output
                except AttributeError:
                    pooler_output = outputs['last_hidden_state'][:, 0, :]
                if self.args.mask_embedding_sentence:
                    last_hidden = outputs.last_hidden_state
                    pooler_output = last_hidden[batch['input_ids'] == self.tokenizer.mask_token_id]
                    if self.args.mask_embedding_sentence_org_mlp:
                        pooler_output = self.mlp(pooler_output)
                    if self.args.mask_embedding_sentence_delta:
                        blen = batch['attention_mask'].sum(-1) - self.template_len
                        if self.args.mask_embedding_sentence_org_mlp:
                            pooler_output -= self.mlp(self.delta[blen])
                        else:
                            pooler_output -= self.delta[blen]
                    if self.args.mask_embedding_sentence_use_org_pooler:
                        pooler_output = self.mlp(pooler_output)
                    if self.args.mask_embedding_sentence_use_pooler:
                        pooler_output = self.encoder.pooler.dense(pooler_output)
                        pooler_output = self.encoder.pooler.activation(pooler_output)
                else:
                    last_hidden = outputs.last_hidden_state
                    hidden_states = outputs.hidden_states

        sentence_embedding = None
        # Apply different pooler
        if self.args.mask_embedding_sentence:
            sentence_embedding = pooler_output.view(batch['input_ids'].shape[0], -1)
        elif self.args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            sentence_embedding = pooler_output
        elif self.args.pooler == 'cls_before_pooler':
            batch['input_ids'][(batch['input_ids'] == 0) | (batch['input_ids'] == 101) | (batch['input_ids'] == 102)] = batch['input_ids'].max()
            index = batch['input_ids'].topk(3, dim=-1, largest=False)[1]
            index2 = torch.arange(batch['input_ids'].shape[0]).to(index.device)
            r = last_hidden[index2, index[:, 0], :]
            for i in range(1, 3):
                r += last_hidden[index2, index[:, i], :]
            sentence_embedding = (r/3)
        elif self.args.pooler == "avg":
            sentence_embedding = ( \
                (last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1) \
            )
        elif self.args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            sentence_embedding = pooled_result
        elif self.args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            sentence_embedding = pooled_result
        else:
            raise NotImplementedError
        
        # Retriever encoding
        #retriever_embedding = normalize(sentence_embedding)
        
        return sentence_embedding

    def embed(self, sentences):
        sentence_vectors = []
        for i in range(0, len(sentences), self.args.batch_size):
            batch = sentences[i:i+self.args.batch_size]
            vectors = self.batcher(batch)
            assert vectors.size()[1] == self.encoder.config.hidden_size
            sentence_vectors.append(vectors)
        sentence_vectors = torch.cat(sentence_vectors, dim=0)
        return sentence_vectors
