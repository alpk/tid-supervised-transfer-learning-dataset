#!/bin/bash
# akya-cuda barbun-cuda single(1 core 15 g  n) palamut-cuda
# short(4 saat) mid1(4 g  n) mid2(8 g  n) long(15g  n)
#SBATCH -p barbun-cuda,akya-cuda
#SBATCH -A akindiroglu
#SBATCH -J=GCN_c_bsign
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
 --model_type 'GCN' \
 --pretrained_model ''  \
 --num_workers 0 \
 --num_epochs 750 \
 --batch_size 16 \
 --dropout 0 \
 --max_num_classes -1 \
 --iterations_per_epoch 300 \
 --display_batch_progress \
 --transfer_method 'combined' \
 --transfer_train_source 'AUTSL_train_whole' \
 --transfer_train_target 'bsign22k_train_shared'\
 --transfer_validation 'bsign22k_val_shared'\
 --experiment_notes ''


