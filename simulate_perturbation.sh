#!/usr/bin/bash
#$ -t 1-200
#$ -N langdon
#$-cwd
#$ -l gpu=1
#$ -l m_mem_free=20G
python simulate_perturbation.py