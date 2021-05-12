#!/usr/bin/bash
#$ -t 1-100
#$ -N langdon_lca
#$-cwd
#$ -l gpu=1
#$ -l m_mem_free=20G
python -u lca_script.py