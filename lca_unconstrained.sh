#!/usr/bin/bash
#$ -t 1-125
#$ -N lca
#$-cwd
#$ -l m_mem_free=20G
python -u lca_unconstrained.py