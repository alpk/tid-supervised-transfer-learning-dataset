#!/bin/bash
#

# akya-cuda barbun-cuda single(1 core 15 gün)
# short(4 saat) mid1(4 gün) mid2(8 gün) long(15gün)
#SBATCH -p mid2
#SBATCH -A akindiroglu
#SBATCH -J=run_preprocess
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=ALL
#SBATCH --time=0-04:00:00
#SBATCH -o /truba_scratch/akindiroglu/Slurm/output/out-%j.out  # send stdout to outfile
#SBATCH -e /truba_scratch/akindiroglu/Slurm/error/err-%j.err  # send stderr to errfile


module load centos7.3/lib/cuda/10.1
module load centos7.3/comp/gcc/6.4


/truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python generate_csv_files.py
