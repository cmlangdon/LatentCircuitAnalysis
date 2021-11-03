#!/usr/bin/bash
#$ -t 1-1
#$ -N train
#$-cwd
#$ -l m_mem_free=20G
python train_models.py


