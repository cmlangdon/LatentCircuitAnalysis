#!/usr/bin/bash
#$ -t 1-25
#$ -N langdon_model_train
#$-cwd
#$ -l gpu=1
#$ -l m_mem_free=20G
python train_models_3.py


