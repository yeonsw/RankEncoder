import argparse
import json
import os
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from scipy.stats import spearmanr
import torch.nn as nn
import string
from tqdm import tqdm

PUNCTUATION = list(string.punctuation)

def calculate_vectors(tokenizer, model, texts):

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    for _ in inputs:
        inputs[_] = inputs[_].cuda()

    temp = inputs["input_ids"]
    
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).last_hidden_state.cpu()

    embeddings = embeddings[temp == tokenizer.mask_token_id]

    embeddings = embeddings.numpy()

    return embeddings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--corpus_file", type=str, required=True)
    parser.add_argument("--sentence_vectors_np_file", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    tokenizer = BertTokenizer(vocab_file=os.path.join(args.checkpoint, "vocab.txt"))

    temp = {"mask_token": tokenizer.mask_token}
    tokenizer.add_special_tokens(temp)

    model = BertModel.from_pretrained(args.checkpoint).cuda()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    #device = torch.device("cpu")
    model = model.to(device)
    
    batch_size = 128

    with open(args.corpus_file, "r") as f:
        sentences = f.readlines()
    
    outputs = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Computing..."):
        batch_sentences = sentences[i:i+batch_size]
        batch = []
        for line in batch_sentences:
            text = line.strip()
            text = text + " ." if text.strip()[-1] not in PUNCTUATION else text
            text = '''This sentence : " ''' + text + ''' " means [MASK] .'''
            batch.append(text)
        vectors = calculate_vectors(tokenizer=tokenizer, model=model, texts=batch)
        outputs.append(vectors)
    outputs = np.concatenate(outputs, axis=0)

    os.makedirs(os.path.dirname(args.sentence_vectors_np_file), exist_ok=True)
    with open(args.sentence_vectors_np_file, "wb") as f:
        np.save(f, outputs)
