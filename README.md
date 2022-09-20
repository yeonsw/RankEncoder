## RANKING-ENHANCED UNSUPERVISED SENTENCE REPRESENTATION LEARNING
**Yeon Seonwoo, Guoyin Wang, Sajal Choudhary, Changmin Seo, Jiwei Li, Xiang Li, Puyang Xu, Sunghyun Park, Alice Oh** | [Paper](https://arxiv.org/abs/2209.04333) | KAIST, Amazon, Zhejiang Univ.

## Abstract
Previous unsupervised sentence embedding studies have focused on data augmentation methods such as dropout masking and rule-based sentence transformation methods. However, these approaches have a limitation of controlling the fine-grained semantics of augmented views of a sentence. This results in inadequate supervision signals for capturing a semantic similarity of similar sentences. In this work, we found that using neighbor sentences enables capturing a more accurate semantic similarity between similar sentences. Based on this finding, we propose RankEncoder, which uses relations between an input sentence and sentences in a corpus for training unsupervised sentence encoders. We evaluate RankEncoder from three perspectives: 1) the semantic textual similarity performance, 2) the efficacy on similar sentence pairs, and 3) the universality of RankEncoder. Experimental results show that RankEncoder achieves 80.07% Spearman's correlation, a 1.1% absolute improvement compared to the previous state-of-the-art performance. The improvement is even more significant, a 1.73% improvement, on similar sentence pairs. Also, we demonstrate that RankEncoder is universally applicable to existing unsupervised sentence encoders.

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
