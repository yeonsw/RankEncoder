OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
CUDA_VISIBLE_DEVICES=0,1 python prompt_bert_rank_encoder_inference.py \
    --sentence_vecs $OUTPUT_DIR/promptbert/index_vecs/corpus_0.01_prompt_bert_rank_encoder_seed_$SEED.npy \
    --senteval_path SentEval/data \
    --batch_size 256 \
    --model_name_or_path $OUTPUT_DIR/promptbert/checkpoints/prompt_bert_rank_encoder_seed_$SEED \
    --mask_embedding_sentence \
    --mask_embedding_sentence_template "*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*" \
    --lmb 0.9
