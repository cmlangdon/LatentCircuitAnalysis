#!/usr/bin/bash
#$ -t 1-5
#$ -N lca
#$-cwd
#$ -l gpu=1
#$ -l m_mem_free=20G
python -u lca_script.py