## RANKING-ENHANCED UNSUPERVISED SENTENCE REPRESENTATION LEARNING
**[Yeon Seonwoo](https://yeonsw.github.io/), Guoyin Wang, Changmin Seo, Sajal Choudhary, Jiwei Li, Xiang Li, Puyang Xu, Sunghyun Park, Alice Oh** | ACL 2023 | [Paper](https://arxiv.org/abs/2209.04333)

KAIST, Amazon, Zhejiang Univ.

## Abstract
Unsupervised sentence representation learning has progressed through contrastive learning and data augmentation methods such as dropout masking. Despite this progress, sentence encoders are still limited to using only an input sentence when predicting its semantic vector. In this work, we show that the semantic meaning of a sentence is also determined by nearest-neighbor sentences that are similar to the input sentence. Based on this finding, we propose a novel unsupervised sentence encoder, RankEncoder. RankEncoder predicts the semantic vector of an input sentence by leveraging its relationship with other sentences in an external corpus, as well as the input sentence itself. We evaluate RankEncoder on semantic textual benchmark datasets. From the experimental results, we verify that 1) RankEncoder achieves 80.07% Spearman's correlation, a 1.1% absolute improvement compared to the previous state-of-the-art performance, 2) RankEncoder is universally applicable to existing unsupervised sentence embedding methods, and 3) RankEncoder is specifically effective for predicting the similarity scores of similar sentence pairs.

## Getting Started
### SimCSE-RankEncoder
Please see [README.md](https://github.com/yeonsw/RankEncoder/tree/main/code/SimCSE_RankEncoder) at code/SimCSE\_RankEncoder

### PromptBERT-RankEncoder
Please see [README.md](https://github.com/yeonsw/RankEncoder/tree/main/code/PromptBERT_RankEncoder) at code/PromptBERT\_RankEncoder

### SNCSE-RankEncoder
Please see [README.md](https://github.com/yeonsw/RankEncoder/tree/main/code/SNCSE_RankEncoder) at code/SNCSE\_RankEncoder

## Code Reference
https://github.com/princeton-nlp/SimCSE

https://github.com/kongds/Prompt-BERT

https://github.com/Sense-GVT/SNCSE
