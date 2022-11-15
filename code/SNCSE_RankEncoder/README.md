# Getting Started

### Requirements
You need at least two GPUs to proceed the following instructions.

### Setting
1. Set the project directory
```bash
export PROJECT_DIR=/path/to/this/project/folder
```
Note that there is no "/" at the end, e.g., /home/RankEncoder.

2. Download the SentEval folder at https://github.com/princeton-nlp/SimCSE and locate the file at code/SNCSE\_RankEncoder/
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

### Training the base encoder ([SNCSE](https://arxiv.org/abs/2201.05979))
We recommend to use the SNCSE checkpoint provided by the authors. Please follow the instruction below.
1. Download SNCSE-bert-base-uncased.zip file at [this link](https://drive.google.com/drive/folders/1w2srzbtTMLlaxUx-7ETV9vQWdw_6lVuN?usp=sharing)
2. Unzip this file and locate the SNCSE-bert-base-uncased folder in $PROJECT\_DIR/outputs/sncse/checkpoints/

Please see details at [this link](https://github.com/Sense-GVT/SNCSE) if you want to train SNCSE from scratch. 

### Training RankEncoder
1. Download the following model
```bash
python -m spacy download en_core_web_sm
```
2. Generate soft negative samples (please see details in the [SNCSE paper](https://arxiv.org/abs/2201.05979))
```bash
bash generate_soft_negative_samples.sh
```
3. Get sentence vectors with the base encoder
```bash
bash get_sncse_embedding.sh
```
4. Train RankEncoder
```bash
cd ./sncse_rank_encoder/
bash sncse_rank_encoder.sh
```
We provide the checkpoint of the trained RankeEncoder-SNCSE [here](https://drive.google.com/file/d/1YSxcTl6bVXqkC2oARq9MyH_9RdBT9G-d/view?usp=share_link)

### Evaluation
The following command will give you the performance of RankEncoder (without Eq.7 in [our paper](https://arxiv.org/pdf/2209.04333.pdf))
```bash
bash evaluation_sncse_rank_encoder.sh
```

The following command compute the performance of RankEncoder (with Eq.7 in [our paper](https://arxiv.org/pdf/2209.04333.pdf))

```bash
bash get_sncse_rank_encoder_embedding.sh
bash sncse_rank_encoder_inference.sh
```
Note that we only sample 10,000 sentences for computational efficiency. Please use 100,000 sentences to replicate our experimental results.
