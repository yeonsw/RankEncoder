OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
CUDA_VISIBLE_DEVICES=0,1 python get_embedding.py \
    --checkpoint $OUTPUT_DIR/sncse/checkpoints/sncse_rank_encoder_seed_$SEED \
    --corpus_file $OUTPUT_DIR/corpus/corpus_0.01.txt \
    --sentence_vectors_np_file $OUTPUT_DIR/sncse/index_vecs/corpus_0.01_sncse_rank_encoder_seed_$SEED.npy
