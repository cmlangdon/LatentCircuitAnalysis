#!/usr/bin/bash
#$ -t 1-5
#$ -N train
#$-cwd
#$ -l m_mem_free=20G
python train_models.py


