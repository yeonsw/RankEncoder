#!/bin/bash
DATA_DIR=$PROJECT_DIR/data
OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
GPU=0,1
ES=125 # --eval_steps
BMETRIC=avg_sts # --metric_for_best_model
TRAIN_FILE=$DATA_DIR/corpus/corpus.txt

args=() # flags for training
eargs=() # flags for evaluation

BC=(python train_prompt_bert_rank_encoder.py)
BATCH=256
EPOCH=1
LR=1e-5
TEMPLATE="*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
TEMPLATE2="*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"
MODEL=bert-base-uncased
args=(--mlp_only_train --mask_embedding_sentence\
      --mask_embedding_sentence_template $TEMPLATE\
      --mask_embedding_sentence_different_template $TEMPLATE2\
      --mask_embedding_sentence_delta\
      --seed $SEED \
      --baseE_model_name_or_checkpoint $OUTPUT_DIR/promptbert/checkpoints/prompt_bert_seed_$SEED \
      --corpus_vecs $OUTPUT_DIR/promptbert/index_vecs/corpus_0.01_prompt_bert_seed_$SEED.npy \
      --baseE_lmb 0.05 \
      --baseE_sim_thresh_low 0.5 \
      --baseE_sim_thresh_upp 0.8 \
      --loss_type hinge \
      --mask_embedding_sentence_delta_no_delta_eval ) 
eargs=(--mask_embedding_sentence \
       --mask_embedding_sentence_template $TEMPLATE )

CHECKPOINT=$OUTPUT_DIR/promptbert/checkpoints/prompt_bert_rank_encoder_seed_$SEED
CUDA_VISIBLE_DEVICES=$GPU ${BC[@]}\
          --model_name_or_path $MODEL\
          --train_file $TRAIN_FILE\
          --output_dir $CHECKPOINT\
          --num_train_epochs $EPOCH\
          --per_device_train_batch_size $BATCH \
          --learning_rate $LR \
          --max_seq_length 32\
          --evaluation_strategy steps\
          --metric_for_best_model $BMETRIC\
          --load_best_model_at_end\
          --eval_steps $ES \
          --overwrite_output_dir \
          --temp 0.05\
          --do_train\
          --fp16\
          --preprocessing_num_workers 10\
          ${args[@]}

CUDA_VISIBLE_DEVICES=$GPU python evaluation.py \
    --model_name_or_path   $CHECKPOINT \
    --pooler avg\
    --mode test\
    ${eargs[@]}
