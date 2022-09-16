OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
CHECKPOINT=$OUTPUT_DIR/promptbert/checkpoints/prompt_bert_seed_$SEED
CUDA_VISIBLE_DEVICES=0,1 python get_embedding.py \
    --model_name_or_path $CHECKPOINT \
    --pooler avg \
    --mask_embedding_sentence \
    --mask_embedding_sentence_template "*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*" \
    --sentence_file $OUTPUT_DIR/corpus/corpus_0.01.txt \
    --vector_file $OUTPUT_DIR/promptbert/index_vecs/corpus_0.01_prompt_bert_seed_$SEED.npy \
    --batch_size 64
