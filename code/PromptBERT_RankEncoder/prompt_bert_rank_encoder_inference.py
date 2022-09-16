import apex
import re
import sys
import io, os
import faiss
import math
import json
import torch
import numpy as np
import logging
import tqdm
import time
import argparse
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr
from scipy.special import softmax
from scipy.stats import rankdata
import string
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)

import senteval
from senteval.sts import STS12Eval, STS13Eval, STS14Eval, STS15Eval, STS16Eval, STSBenchmarkEval
from senteval.sts import SICKRelatednessEval

def normalize(vecs):
    eps = 1e-8
    return vecs / (np.sqrt(np.sum(np.square(vecs), axis=1)) + eps)[:,None]

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def read_benchmark_data(senteval_path, task):
    task2class = { \
        'STS12': STS12Eval,
        'STS13': STS13Eval,
        'STS14': STS14Eval,
        'STS15': STS15Eval,
        'STS16': STS16Eval,
        'STSBenchmark': STSBenchmarkEval,
        'SICKRelatedness': SICKRelatednessEval
    }
    dataset_path = None
    print("SentEval path: {}".format(senteval_path))
    if task in ["STS12", "STS13", "STS14", "STS15", "STS16"]:
        dataset_path = os.path.join(senteval_path, "downstream/STS/", "{}{}".format(task, "-en-test"))
    elif task == "STSBenchmark":
        dataset_path = os.path.join(senteval_path, "downstream/STS/", "{}".format(task))
    elif task == "SICKRelatedness":
        dataset_path = os.path.join(senteval_path, "downstream/SICK")
    print(dataset_path)
    data = {}
    task_data = task2class[task](dataset_path)
    for dset in task_data.datasets:
        input1, input2, gs_scores = task_data.data[dset]
        data[dset] = (input1, input2, gs_scores)
    return data

def compute_similarity(q0, q0_sim, q1, q1_sim, lmb=0.0):
    normalized_q0 = normalize(np.reshape(q0, (1, -1)))
    normalized_q1 = normalize(np.reshape(q1, (1, -1)))
    add_score, _ = spearmanr(q0_sim, q1_sim)
    score = np.sum(np.matmul(normalized_q0, normalized_q1.T))
    score = lmb * score + (1.0 - lmb) * add_score
    return score

def evaluate_retrieval_augmented_promptbert(args, data, batcher, sentence_vecs):
    results = {}
    all_sys_scores = []
    all_gs_scores = []
    for dset in data:
        sys_scores = []
        input1, input2, gs_scores = data[dset]
        for ii in range(0, len(gs_scores), args.batch_size):
            batch1 = input1[ii:ii + args.batch_size]
            batch2 = input2[ii:ii + args.batch_size]

            # we assume get_batch already throws out the faulty ones
            if len(batch1) == len(batch2) and len(batch1) > 0:
                enc1 = batcher(batch1)
                enc2 = batcher(batch2)
                sim1 = np.matmul( \
                    enc1, sentence_vecs.T \
                )
                sim2 = np.matmul( \
                    enc2, sentence_vecs.T \
                )
                
                for kk in range(enc1.shape[0]):
                    sys_score = compute_similarity( \
                        enc1[kk], sim1[kk], \
                        enc2[kk], sim2[kk], \
                        args.lmb \
                    )
                    sys_scores.append(sys_score)
        all_sys_scores.extend(sys_scores)
        all_gs_scores.extend(gs_scores)
        results[dset] = {
            'pearson': pearsonr(sys_scores, gs_scores),
            'spearman': spearmanr(sys_scores, gs_scores),
            'nsamples': len(sys_scores)
        }
        logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                      (dset, results[dset]['pearson'][0],
                       results[dset]['spearman'][0]))
        
    weights = [results[dset]['nsamples'] for dset in results.keys()]
    list_prs = np.array([results[dset]['pearson'][0] for
                        dset in results.keys()])
    list_spr = np.array([results[dset]['spearman'][0] for
                        dset in results.keys()])

    avg_pearson = np.average(list_prs)
    avg_spearman = np.average(list_spr)
    wavg_pearson = np.average(list_prs, weights=weights)
    wavg_spearman = np.average(list_spr, weights=weights)
    all_pearson = pearsonr(all_sys_scores, all_gs_scores)
    all_spearman = spearmanr(all_sys_scores, all_gs_scores)
    results['all'] = {'pearson': {'all': all_pearson[0],
                                      'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'all': all_spearman[0],
                                       'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
    logging.debug('ALL : Pearson = %.4f, \
        Spearman = %.4f' % (all_pearson[0], all_spearman[0]))
    logging.debug('ALL (weighted average) : Pearson = %.4f, \
        Spearman = %.4f' % (wavg_pearson, wavg_spearman))
    logging.debug('ALL (average) : Pearson = %.4f, \
        Spearman = %.4f\n' % (avg_pearson, avg_spearman))
    results["pred_scores"] = all_sys_scores
    results["gs_scores"] = all_gs_scores
    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_vecs", type=str, required=True)
    parser.add_argument("--senteval_path", type=str, default="SentEval/data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lmb", type=float, default=1.0)

    # PromptBERT args
    parser.add_argument('--mask_embedding_sentence', action='store_true')
    parser.add_argument('--mask_embedding_sentence_template', type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str,
            help="Transformers' model name or path")
    args = parser.parse_args()
    return args

def main(args):
    # Load transformers' model checkpoint
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    model = model.to(device)
    encoder = model
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = model.module
    model.eval()

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

            for i, s in enumerate(sentences):
                if len(s) > 0 and s[-1] not in '.?"\'': s += '.'
                sentences[i] = template.replace('*sent 0*', s).strip()

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
            outputs = model(**batch, output_hidden_states=True, return_dict=True)

            try:
                pooler_output = outputs.pooler_output
            except AttributeError:
                pooler_output = outputs['last_hidden_state'][:, 0, :]
            if args.mask_embedding_sentence:
                last_hidden = outputs.last_hidden_state
                pooler_output = last_hidden[batch['input_ids'] == tokenizer.mask_token_id]
            else:
                last_hidden = outputs.last_hidden_state
                hidden_states = outputs.hidden_states

        sentence_embedding = None
        if args.mask_embedding_sentence:
            sentence_embedding = pooler_output.view(batch['input_ids'].shape[0], -1).cpu()
        else:
            raise NotImplementedError
        
        sentence_embedding = normalize(sentence_embedding.numpy())
        
        return sentence_embedding
     
    print("Loading {}".format(args.sentence_vecs))
    sentence_vecs = np.load(args.sentence_vecs)

    # Load benchmark datasets
    target_tasks = [ \
        'STS12', 'STS13', 'STS14', 'STS15', 'STS16', \
        'STSBenchmark', \
        'SICKRelatedness' \
    ]
    # Reference: https://github.com/facebookresearch/SentEval/blob/main/senteval/sts.py
    results = {}
    for task in target_tasks:
        data = read_benchmark_data(args.senteval_path, task)
        result = evaluate_retrieval_augmented_promptbert(args, data, batcher, sentence_vecs)
        results[task] = result
    
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append("%.2f" % (results[task]['all']['spearman']['all'] * 100))
            else:
                scores.append("%.2f" % (results[task]['test']['spearman'].correlation * 100))
        else:
            scores.append("0.00")
    task_names.append("Avg.")
    scores.append("%.2f" % (sum([float(score) for score in scores]) / len(scores)))
    print_table(task_names, scores)
    
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
