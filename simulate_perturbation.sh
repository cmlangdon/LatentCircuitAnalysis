#!/usr/bin/bash
#$ -t 1-1
#$ -N langdon
#$-cwd
#$ -l gpu=1
#$ -l m_mem_free=20G
python simulate_perturbation.py