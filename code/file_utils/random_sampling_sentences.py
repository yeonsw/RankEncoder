import os

import argparse
import random
from tqdm import tqdm

def write_corpus_file(sentences, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w") as f:
        for s in tqdm(sentences, desc="Writing"):
            f.write("{}\n".format(s))
    return 0

def read_wiki1m(fname):
    print("Reading {}".format(fname))
    with open(fname, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines

def sampling_sentences(sentences, ratio, n_sentences):
    n = None
    if args.ratio != None:
        n = int(len(sentences) * ratio)
    elif args.n_sentences != None:
        n = args.n_sentences
    #assert n_sentences <= len(sentences)
    n = min(n, len(sentences))
    samples = random.sample(sentences, n)
    return samples

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--ratio", type=float, default=None)
    parser.add_argument("--n_sentences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args

def main(args):
    assert args.ratio != None or args.n_sentences != None
    
    random.seed(args.seed)
    texts = read_wiki1m(args.sentence_file)
    samples = sampling_sentences(texts, args.ratio, args.n_sentences)
    _ = write_corpus_file(samples, args.output_file)
    return 0

if __name__ == "__main__":
    args = parse_args()
    _ = main(args)
