OUTPUT_DIR=$PROJECT_DIR/outputs

LMB=0.9
SEED=61507
CUDA_VISIBLE_DEVICES=0,1 python simcse_rank_encoder_inference.py \
    --sentence_vecs $OUTPUT_DIR/simcse/index_vecs/corpus_0.01_rank_encoder_seed_$SEED.npy \
    --batch_size 256 \
    --model_name_or_path $OUTPUT_DIR/simcse/checkpoints/simcse_unsup_rank_encoder_seed_$SEED \
    --lmb $LMB
