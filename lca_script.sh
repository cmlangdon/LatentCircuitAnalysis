#!/usr/bin/bash
#$ -t 1-25
#$ -N lca
#$-cwd
#$ -l m_mem_free=20G
#$ -pe threads 1
python -u lca_script.py