#!/usr/bin/bash
#$ -t 1-25
#$ -N lca
#$-cwd

#$ -l m_mem_free=20G
python -u lca_script.py