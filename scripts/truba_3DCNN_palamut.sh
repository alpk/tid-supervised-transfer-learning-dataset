#!/bin/bash
experiment_notes_var=bsign_default
# akya-cuda barbun-cuda single(1 core 15 gün) palamut-cuda
# short(4 saat) mid1(4 gün) mid2(8 gün) long(15gün)
#SBATCH -p palamut-cuda,barbun-cuda,akya-cuda
#SBATCH -A akindiroglu
#SBATCH -J=$experiment_notes_var
#SBATCH -N 1    #number of nodes
#SBATCH -n 16   #number of cpus
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --time=02-00:00:00
#SBATCH -o /truba_scratch/akindiroglu/Slurm/output/out-%j.out  # send stdout to outfile
#SBATCH -e /truba_scratch/akindiroglu/Slurm/error/err-%j.err  # send stderr to errfile

module purge
module load centos7.3/lib/cuda/10.1
module load centos7.3/comp/gcc/6.4


/truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py \
 --model_type 'mc3_18' \
 --pretrained_model ''  \
 --num_workers 0 \
 --num_epochs 1000 \
 --input_size 128 \
 --batch_size = 64 \
 --dropout = 0.5 \
 --max_num_classes -1 \
 --iterations_per_epoch -1 \
 --display_batch_progress \
 --transfer_method 'single_target' \
 --transfer_train_source 'AUTSL_train_shared' \
 --transfer_train_target 'bsign22k_train_shared'\
 --transfer_validation 'bsign22k_val_shared'\
 --experiment_notes $experiment_notes_var
