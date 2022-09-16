import re
import sys
import io, os
import math
import torch
import numpy as np
import logging
import tqdm
import argparse
from prettytable import PrettyTable
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def get_delta(model, template, tokenizer, device, args):
    model.eval()

    template = template.replace('*mask*', tokenizer.mask_token)\
                    .replace('*sep+*', '')\
                    .replace('*cls*', '').replace('*sent_0*', ' ')
    # strip for roberta tokenizer
    bs_length = len(tokenizer.encode(template.split(' ')[0].replace('_', ' ').strip())) - 2 + 1
    # replace for roberta tokenizer
    batch = tokenizer([template.replace('_', ' ').strip().replace('   ', ' ')], return_tensors='pt')
    batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).to(device).unsqueeze(0)
    for k in batch:
        batch[k] = batch[k].repeat(512, 1).to(device)
    m_mask = batch['input_ids'] == tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(**batch,  output_hidden_states=True, return_dict=True)
        last_hidden = outputs.hidden_states[-1]
        delta = last_hidden[m_mask]
    delta.requires_grad = False
    #import pdb;pdb.set_trace()
    template_len = batch['input_ids'].shape[1]
    return delta, template_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_file", type=str, required=True)
    parser.add_argument("--vector_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_only", action='store_true')
    parser.add_argument('--mlm_head_predict', action='store_true')
    parser.add_argument('--remove_continue_word', action='store_true')
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_org_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument('--mask_embedding_sentence_delta', action='store_true')
    parser.add_argument('--mask_embedding_sentence_use_pooler', action='store_true')
    parser.add_argument('--mask_embedding_sentence_autoprompt', action='store_true')
    parser.add_argument('--mask_embedding_sentence_org_mlp', action='store_true')
    parser.add_argument("--tokenizer_name", type=str, default='')
    parser.add_argument("--model_name_or_path", type=str,
            help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str,
            choices=['cls', 'cls_before_pooler', 'avg',  'avg_first_last'],
            default='cls', 
            help="Which pooler to use")

    args = parser.parse_args()

    # Load transformers' model checkpoint
    if args.mask_embedding_sentence_org_mlp:
        #only for bert-base
        from transformers import BertForMaskedLM, BertConfig
        config = BertConfig.from_pretrained("bert-base-uncased")
        mlp = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config).cls.predictions.transform
        if 'result' in args.model_name_or_path:
            state_dict = torch.load(args.model_name_or_path+'/pytorch_model.bin')
            new_state_dict = {}
            for key, param in state_dict.items():
                # Replace "mlp" to "pooler"
                if 'pooler' in key:
                    key = key.replace("pooler.", "")
                    new_state_dict[key] = param
            mlp.load_state_dict(new_state_dict)
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    if args.mask_embedding_sentence_autoprompt:
        state_dict = torch.load(args.model_name_or_path+'/pytorch_model.bin')
        p_mbv = state_dict['p_mbv']
        template = args.mask_embedding_sentence_template
        template = template.replace('*mask*', tokenizer.mask_token)\
                .replace('*sep+*', '')\
                .replace('*cls*', '').replace('*sent_0*', ' ').replace('_', ' ')
        mask_embedding_template = tokenizer.encode(template)
        mask_index = mask_embedding_template.index(tokenizer.mask_token_id)
        index_mbv = mask_embedding_template[1:mask_index] + mask_embedding_template[mask_index+1:-1]
        #mask_embedding_template = [ 101, 2023, 6251, 1997, 1000, 1000, 2965, 103, 1012, 102]
        #index_mbv = mask_embedding_template[1:7] + mask_embedding_template[8:9]

        dict_mbv = index_mbv
        fl_mbv = [i <= 3 for i, k in enumerate(index_mbv)]

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()
    encoder = model

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = model.module
    model.eval()

    #device = torch.device("cpu")
    model = model.to(device)
    if args.mask_embedding_sentence_org_mlp:
        mlp = mlp.to(device)

    if args.mask_embedding_sentence_delta:
        delta, template_len = get_delta(model, args.mask_embedding_sentence_template, tokenizer, device, args)

    if args.remove_continue_word:
        pun_remove_set = {'?', '*', '#', '´', '’', '=', '…', '|', '~', '/', '‚', '¿', '–', '»', '-', '€', '‘', '"', '(', '•', '`', '$', ':', '[', '”', '%', '£', '<', '[UNK]', ';', '“', '@', '_', '{', '^', ',', '.', '!', '™', '&', ']', '>', '\\', "'", ')', '+', '—'}
        if args.model_name_or_path == 'roberta-base':
            remove_set = {'Ġ.', 'Ġa', 'Ġthe', 'Ġin', 'a', 'Ġ, ', 'Ġis', 'Ġto', 'Ġof', 'Ġand', 'Ġon', 'Ġ\'', 's', '.', 'the', 'Ġman', '-', 'Ġwith', 'Ġfor', 'Ġat', 'Ġwoman', 'Ġare', 'Ġ"', 'Ġthat', 'Ġit', 'Ġdog', 'Ġsaid', 'Ġplaying', 'Ġwas', 'Ġas', 'Ġfrom', 'Ġ:', 'Ġyou', 'Ġan', 'i', 'Ġby'}
        else:
            remove_set = {".", "a", "the", "in", ",", "is", "to", "of", "and", "'", "on", "man", "-", "s", "with", "for", "\"", "at", "##s", "woman", "are", "it", "two", "that", "you", "dog", "said", "playing", "i", "an", "as", "was", "from", ":", "by", "white"}

        vocab = tokenizer.get_vocab()


    def batcher(batch, max_length=None):
        # Handle rare token encoding issues in the dataset
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        if args.mask_embedding_sentence and args.mask_embedding_sentence_template is not None:
            # *cls*_This_sentence_of_"*sent_0*"_means*mask*.*sep+*
            template = args.mask_embedding_sentence_template
            template = template.replace('*mask*', tokenizer.mask_token )\
                               .replace('_', ' ').replace('*sep+*', '')\
                               .replace('*cls*', '')
            template_tokens = tokenizer.encode(template.replace('*sent 0*', ""), add_special_tokens=False)
            len_template_tokens = len(template_tokens)

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                s_tokens = tokenizer.encode(s, add_special_tokens=False)
                limit = 512 - 2 - len_template_tokens
                if limit < len(s_tokens):
                    s_tokens = s_tokens[:limit]
                s = tokenizer.decode(s_tokens)
                sentences[i] = template.replace('*sent 0*', s).strip()
        elif args.remove_continue_word:
            for i, s in enumerate(sentences):
                sentences[i] = ' ' if args.model_name_or_path == 'roberta-base' else ''
                es = tokenizer.encode(' ' + s, add_special_tokens=False)
                for iw, w in enumerate(tokenizer.convert_ids_to_tokens(es)):
                    if args.model_name_or_path == 'roberta-base':
                        # roberta base
                        if 'Ġ' not in w or w in remove_set:
                            pass
                        else:
                            if re.search('[a-zA-Z0-9]', w) is not None:
                                sentences[i] += w.replace('Ġ', '').lower() + ' '
                    elif w not in remove_set and w not in pun_remove_set and '##' not in w:
                        # bert base
                        sentences[i] += w.lower() + ' '
                if len(sentences[i]) == 0: sentences[i] = '[PAD]'
        
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                max_length=max_length,
                truncation=True
            )
        else:
            batch = tokenizer.batch_encode_plus(
                sentences,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
        
        # Move to the correct device
        for k in batch:
            batch[k] = batch[k].to(device) if batch[k] is not None else None
        
        # Get raw embeddings
        with torch.no_grad():
            if args.embedding_only:
                hidden_states = None
                pooler_output = None
                last_hidden = encoder.embeddings.word_embeddings(batch['input_ids'])
                position_ids = encoder.embeddings.position_ids[:, 0 : last_hidden.shape[1]]
                token_type_ids = torch.zeros(batch['input_ids'].shape, dtype=torch.long,
                                                device=encoder.embeddings.position_ids.device)

                position_embeddings = encoder.embeddings.position_embeddings(position_ids)
                token_type_embeddings = encoder.embeddings.token_type_embeddings(token_type_ids)

                if args.remove_continue_word:
                    batch['attention_mask'][batch['input_ids'] == tokenizer.cls_token_id] = 0
                    batch['attention_mask'][batch['input_ids'] == tokenizer.sep_token_id] = 0
            elif args.mask_embedding_sentence_autoprompt:
                input_ids = batch['input_ids']
                inputs_embeds = encoder.embeddings.word_embeddings(input_ids)
                p = torch.arange(input_ids.shape[1]).to(input_ids.device).view(1, -1)
                b = torch.arange(input_ids.shape[0]).to(input_ids.device)
                for i, k in enumerate(dict_mbv):
                    if fl_mbv[i]:
                        index = ((input_ids == k) * p).max(-1)[1]
                    else:
                        index = ((input_ids == k) * -p).min(-1)[1]
                    inputs_embeds[b, index] = p_mbv[i]
                batch['input_ids'], batch['inputs_embeds'] = None, inputs_embeds
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
                batch['input_ids'] = input_ids

                last_hidden = outputs.last_hidden_state
                pooler_output = last_hidden[input_ids == tokenizer.mask_token_id]

                if args.mask_embedding_sentence_org_mlp:
                    pooler_output = mlp(pooler_output)
                if args.mask_embedding_sentence_delta:
                    blen = batch['attention_mask'].sum(-1) - template_len
                    if args.mask_embedding_sentence_org_mlp:
                        pooler_output -= mlp(delta[blen])
                    else:
                        pooler_output -= delta[blen]
                if args.mask_embedding_sentence_use_pooler:
                    pooler_output = encoder.pooler.dense(pooler_output)
                    pooler_output = encoder.pooler.activation(pooler_output)

            else:
                outputs = model(**batch, output_hidden_states=True, return_dict=True)

                try:
                    pooler_output = outputs.pooler_output
                except AttributeError:
                    pooler_output = outputs['last_hidden_state'][:, 0, :]
                if args.mask_embedding_sentence:
                    last_hidden = outputs.last_hidden_state
                    pooler_output = last_hidden[batch['input_ids'] == tokenizer.mask_token_id]
                    if args.mask_embedding_sentence_org_mlp:
                        pooler_output = mlp(pooler_output)
                    if args.mask_embedding_sentence_delta:
                        blen = batch['attention_mask'].sum(-1) - template_len
                        if args.mask_embedding_sentence_org_mlp:
                            pooler_output -= mlp(delta[blen])
                        else:
                            pooler_output -= delta[blen]
                    if args.mask_embedding_sentence_use_org_pooler:
                        pooler_output = mlp(pooler_output)
                    if args.mask_embedding_sentence_use_pooler:
                        pooler_output = encoder.pooler.dense(pooler_output)
                        pooler_output = encoder.pooler.activation(pooler_output)
                else:
                    last_hidden = outputs.last_hidden_state
                    hidden_states = outputs.hidden_states

        # Apply different pooler
        if args.mask_embedding_sentence:
            return pooler_output.view(batch['input_ids'].shape[0], -1).cpu()
        elif args.pooler == 'cls':
            # There is a linear+activation layer after CLS representation
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            batch['input_ids'][(batch['input_ids'] == 0) | (batch['input_ids'] == 101) | (batch['input_ids'] == 102)] = batch['input_ids'].max()
            index = batch['input_ids'].topk(3, dim=-1, largest=False)[1]
            index2 = torch.arange(batch['input_ids'].shape[0]).to(index.device)
            r = last_hidden[index2, index[:, 0], :]
            for i in range(1, 3):
                r += last_hidden[index2, index[:, i], :]
            return (r/3).cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError
    
    sentences = None
    with open(args.sentence_file, "r") as f:
        sentences = f.readlines()
        sentences = [s.strip().split() for s in tqdm(sentences, desc="Preprocessing")]
    
    sentences = sorted([(i, s) for i, s in enumerate(sentences)], key=lambda x: len(x[1]), reverse=True)
    inds, sentences = map(list, zip(*sentences))
    sort_inds = sorted([(i, j) for i, j in enumerate(inds)], key=lambda x: x[1])
    sort_inds, _ = map(list, zip(*sort_inds))

    sentence_vectors = [] 
    n_batch = math.ceil(len(sentences) / args.batch_size)
    for i in tqdm(range(0, len(sentences), args.batch_size), desc="Embedding sentences..."):
        batch = sentences[i:i + args.batch_size]
        vectors = batcher(batch).numpy()
        assert vectors.shape[1] == encoder.config.hidden_size
        sentence_vectors.append(vectors)
    sentence_vectors = np.concatenate(sentence_vectors)
    sentence_vectors = sentence_vectors[sort_inds]
    
    print("Saving...")
    os.makedirs(os.path.dirname(args.vector_file), exist_ok=True)
    with open(args.vector_file, "wb") as f:
        np.save(f, sentence_vectors)
    
if __name__ == "__main__":
    main()
