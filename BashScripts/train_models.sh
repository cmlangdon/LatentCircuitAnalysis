#!/usr/bin/bash
#$ -t 1-25
#$ -N train
#$-cwd
#$ -l m_mem_free=20G
#$ -pe threads 1
python train_models.py


