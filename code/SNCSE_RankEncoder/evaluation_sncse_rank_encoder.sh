OUTPUT_DIR=$PROJECT_DIR/outputs

SEED=61507
python evaluation.py \
    --model_name_or_path $OUTPUT_DIR/sncse/checkpoints/sncse_rank_encoder_seed_$SEED \
    --task_set sts \
    --mode test
