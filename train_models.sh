#!/usr/bin/bash
#$ -t 1-25
#$ -N train
#$-cwd
#$ -l gpu=1
#$ -l m_mem_free=20G
python train_models.py


