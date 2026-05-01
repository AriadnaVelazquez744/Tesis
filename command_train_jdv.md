4. Train
MODEL_DIR_NAME="Qwen2.5-3B-Instruct" \
TRAIN_DATA_PATH="src/Vagueness_Judge/data/augmented/interaction_data_train.jsonl" \
MAX_SEQ_LENGTH=1024 \
EPOCHS=3 \
LR=2e-5 \
BS_PER_DEVICE=2 \
GRAD_ACCUM=4 \
RESUME=1 \
bash src/Vagueness_Judge/training/sft.sh
5. Resume after interruption (same command)
# Just run step 4 again -- auto-resumes from last checkpoint
