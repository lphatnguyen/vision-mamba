#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=output_ddp_2_procs.log

module load anaconda3/latest
module load cuda12.2/toolkit/12.2.2

source activate visionMamba

cd /home/lng5982/codes/vision-mamba
torchrun --nproc_per_node 2 vim/main_emotion.py --batch-size 128 --output_dir output/test_ddp_2_procs_per_node --opt adam
