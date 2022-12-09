# Getting Started

### Requirements
You need at least two GPUs to proceed the following instructions.

### Setting
1. Set the project directory
```bash
export PROJECT_DIR=/path/to/this/project/folder
```
Note that there is no "/" at the end, e.g., /home/RankEncoder.

2. Download the SentEval folder at https://github.com/princeton-nlp/SimCSE and locate the file at code/SimCSE\_RankEncoder/
3. Go to SentEval/data/downstream and execute the following command
```bash 
bash download_dataset.sh
```
4. Download the Wiki1m dataset with the following command and locate this dataset at data/corpus/corpus.txt
```bash
bash wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
mv wiki1m_for_simcse.txt ../../data/corpus/corpus.txt
```
5. Go to code/file\_utils/ and execute the following command.
```bash
bash random_sampling_sentences.sh
```

### Training the base encoder ([SimCSE](https://aclanthology.org/2021.emnlp-main.552/))
Go to code/SimCSE\_RankEncoder and execute the following command
```bash
bash run_simcse.sh
```

### Training RankEncoder
1. Get sentence vectors with the base encoder
```bash
bash get_simcse_embedding.sh
```
2. Train RankEncoder
```bash
bash run_simcse_rank_encoder.sh
```

We provide the checkpoint of the trained RankeEncoder-SimCSE [here](https://drive.google.com/file/d/15BvamHk4zuCSU1slOWb37bnncJ2GGIwX/view?usp=sharing)

### Evaluation
1. The following command compute the performance of RankEncoder (without Eq.7 in [our paper](https://arxiv.org/pdf/2209.04333.pdf))
```bash
bash evaluation.sh
```

2. The following command compute the performance of RankEncoder (Eq.7 in [our paper](https://arxiv.org/pdf/2209.04333.pdf))
```bash
bash get_rank_encoder_embedding.sh
bash simcse_rank_encoder_inference.sh
```
Note that we only sample 10,000 sentences for computational efficiency. Please use 100,000 sentences to replicate our experimental results.
