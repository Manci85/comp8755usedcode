#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --time=10:05:00
#SBATCH --mem=200GB

module load tensorboardx/1.2.0-py36-cuda90
module load pytorch/1.2.0-py36-cuda90
module load python/3.6.1
module load apex/0.1

python ntu120_gendata_velocity.py