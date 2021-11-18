#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=100:05:00
#SBATCH --mem=75GB

module load tensorboardx/1.2.0-py36-cuda90
module load pytorch/1.2.0-py36-cuda90
module load python/3.6.1
module load apex/0.1

python main.py \
    --config ./config/nturgbd120-cross-subject/transformer/train_joint_only_sgcn_temporal_transformer.yaml
