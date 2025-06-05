#!/bin/bash
#SBATCH --job-name=clmbr_model_eval
#SBATCH --output=model.out
#SBATCH --error=model.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

source /hpc/group/engelhardlab/xg97/venvs/nncompare/bin/activate
cd /hpc/group/engelhardlab/xg97/EHRSHOT_ASSETS/
python model_compare_dcc.py
