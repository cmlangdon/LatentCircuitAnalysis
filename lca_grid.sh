#!/usr/bin/bash
#$ -t 1-400
#$ -N lca_grid
#$-cwd
#$ -l gpu=1
#$ -l m_mem_free=20G
python lca_grid.py

