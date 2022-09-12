#!/bin/bash

OUTPUT_DIR=$PROJECT_DIR/outputs
DATA_DIR=$PROJECT_DIR/data

SEED=61507
VECTOR_FILE=$OUTPUT_DIR/simcse/index_vecs/corpus_0.01_seed_$SEED.npy
CUDA_VISIBLE_DEVICES=0,1 python train_simcse_rank_encoder.py \
    --baseE_sim_thresh_upp 0.9999 \
    --baseE_sim_thresh_low 0.5 \
    --baseE_lmb 0.05 \
    --simf Spearmanr \
    --loss_type weighted_sum \
    --baseE_model_name_or_path $OUTPUT_DIR/simcse/checkpoints/simcse_unsup_seed_$SEED \
    --corpus_vecs $VECTOR_FILE \
    --model_name_or_path bert-base-uncased \
    --train_file $DATA_DIR/corpus/corpus.txt \
    --output_dir $OUTPUT_DIR/simcse/checkpoints/simcse_unsup_rank_encoder_seed_$SEED \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed $SEED \
    "$@"
