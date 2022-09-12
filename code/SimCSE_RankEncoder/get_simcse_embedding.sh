OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
CUDA_VISIBLE_DEVICES=1,2 python get_embedding.py \
    --corpus_file $OUTPUT_DIR/corpus/corpus_0.01.txt \
    --vector_file $OUTPUT_DIR/simcse/index_vecs/corpus_0.01_seed_$SEED.npy \
    --batch_size 128 \
    --model_name_or_path $OUTPUT_DIR/simcse/checkpoints/simcse_unsup_seed_$SEED \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
