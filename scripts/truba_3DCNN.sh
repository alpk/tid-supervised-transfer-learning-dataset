#!/bin/bash
#

# akya-cuda barbun-cuda single(1 core 15 gün)
# short(4 saat) mid1(4 gün) mid2(8 gün) long(15gün)
#SBATCH -p akya-cuda
#SBATCH -A akindiroglu
#SBATCH -J=run_taf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --time=4-00:00:00
#SBATCH -o /truba_scratch/akindiroglu/Slurm/output/out-%j.out  # send stdout to outfile
#SBATCH -e /truba_scratch/akindiroglu/Slurm/error/err-%j.err  # send stderr to errfile


module load centos7.3/lib/cuda/10.1
module load centos7.3/comp/gcc/6.4


/truba/home/akindiroglu/Workspace/Libs/pytorch_nightly/bin/python main.py \
 --model_type 'r2plus1d_18' \
 --pretrained_model ''  \
 --num_workers 0 \
 --num_epochs 1000 \
 --input_size 64 \
 --num_workers 10 \
 --max_num_classes -1 \
 --iterations_per_epoch -1 \
 --display_batch_progress \
 --transfer_train_source 'AUTSL_train_shared' \
 --transfer_train_target 'bsign22k_train_shared'\
 --transfer_validation 'bsign22k_val_shared'\
 --experiment_notes bsign_default
