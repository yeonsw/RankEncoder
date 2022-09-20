OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
CUDA_VISIBLE_DEVICES=0,1 python sncse_rank_encoder_inference.py \
    --sentence_vecs $OUTPUT_DIR/sncse/index_vecs/corpus_0.01_sncse_rank_encoder_seed_$SEED.npy \
    --senteval_path SentEval/data \
    --batch_size 256 \
    --model_name_or_path $OUTPUT_DIR/sncse/checkpoints/sncse_rank_encoder_seed_$SEED \
    --lmb 0.9
