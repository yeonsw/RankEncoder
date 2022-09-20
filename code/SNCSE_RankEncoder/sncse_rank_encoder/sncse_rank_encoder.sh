#!/bin/sh

OUTPUT_DIR=$PROJECT_DIR/outputs
DATA_DIR=$PROJECT_DIR/data

SEED=61507
CUDA_VISIBLE_DEVICES=0,1 python train_SNCSE.py \
    --baseE_sim_thresh_upp 0.9999 \
    --baseE_sim_thresh_low 0.5 \
    --baseE_lmb 0.05 \
    --baseE_model_name_or_path $OUTPUT_DIR/sncse/checkpoints/SNCSE-bert-base-uncased \
    --corpus_vecs $OUTPUT_DIR/sncse/index_vecs/corpus_0.01_sncse.npy \
    --model_name_or_path bert-base-uncased \
    --train_file $DATA_DIR/corpus/corpus.txt \
    --output_dir $OUTPUT_DIR/sncse/checkpoints/sncse_rank_encoder_seed_$SEED \
    --num_train_epoch 1 \
    --per_device_train_batch_size 128 \
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_step 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --preprocessing_num_workers 10 \
    --seed $SEED \
    --soft_negative_file $OUTPUT_DIR/sncse/soft_negative_samples.txt
