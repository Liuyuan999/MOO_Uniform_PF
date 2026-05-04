#!/bin/bash
#SBATCH -p mollab
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH -t 24:00:00
#SBATCH -J qwen_reinforce
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err

# Load environment
source activate llm_ft311

# Execute training
accelerate launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  -m qwen.train_reinforce \
  --config configs/model/qwen_0p5b_dora.yaml \
         configs/task/reddit_summarization.yaml \
         configs/rl/reinforce.yaml \
  --weight 1 \
  --resume outputs/run/Qwen2.5-0.5B-Instruct-rf-summary-w1-0407-145211/checkpoint_step_0000187
