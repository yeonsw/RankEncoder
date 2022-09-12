OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
python evaluation.py \
    --model_name_or_path $OUTPUT_DIR/simcse/checkpoints/simcse_unsup_rank_encoder_seed_$SEED \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test
