#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
DATA_DIR=$PROJECT_DIR/data
OUTPUT_DIR=$PROJECT_DIR/outputs
SEED=61507
CUDA_VISIBLE_DEVICES=0 python train_simcse.py \
    --model_name_or_path bert-base-uncased \
    --train_file $DATA_DIR/corpus/corpus.txt \
    --output_dir $OUTPUT_DIR/simcse/checkpoints/simcse_unsup_seed_$SEED \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed $SEED \
    --load_best_model_at_end \
    "$@"
