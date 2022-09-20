DATA_DIR=$PROJECT_DIR/data
OUTPUT_DIR=$PROJECT_DIR/outputs

python generate_soft_negative_samples.py \
    --in_file $DATA_DIR/corpus/corpus.txt \
    --out_file $OUTPUT_DIR/sncse/soft_negative_samples.txt
