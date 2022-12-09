# Getting Started

### Requirements
You need at least two GPUs to proceed the following instructions.

### Setting
1. Set the project directory
```bash
export PROJECT_DIR=/path/to/this/project/folder
```
Note that there is no "/" at the end, e.g., /home/RankEncoder.

2. Download the SentEval folder at https://github.com/princeton-nlp/SimCSE and locate the file at code/PromptBERT\_RankEncoder/
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

### Training the base encoder ([PromptBERT](https://arxiv.org/abs/2201.04337))
Go to code/PromptBERT\_RankEncoder and execute the following command
```bash
bash run_prompt_bert.sh
```

### Training RankEncoder
1. Get sentence vectors with the base encoder
```bash
bash get_prompt_bert_embedding.sh
```
2. Train RankEncoder
```bash
bash run_prompt_bert_rank_encoder.sh
```

The above command will give you the performance of RankEncoder (without Eq.7 in [our paper](https://arxiv.org/pdf/2209.04333.pdf))

We provide the checkpoint of the trained RankeEncoder-PromptBERT [here](https://drive.google.com/file/d/1ixIt_TNx2c1fSfpzMPeuVMQVuqDAYMZo/view?usp=sharing)

### Evaluation

The following command compute the performance of RankEncoder (with Eq.7 in [our paper](https://arxiv.org/pdf/2209.04333.pdf))

```bash
bash get_prompt_bert_rank_encoder_embedding.sh
bash prompt_bert_rank_encoder_inference.sh
```
Note that we only sample 10,000 sentences for computational efficiency. Please use 100,000 sentences to replicate our experimental results.
