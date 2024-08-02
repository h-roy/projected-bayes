#!/bin/bash
#BSUB -J resnet_ood_metrics
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 20:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.11.4 cuda/12.3.2 cudnn/v8.9.1.23-prod-cuda-12.X
source proj/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false


python experiments/ood_metrics.py --dataset CIFAR-10 --model ResNet_small --posterior_type MAP --experiment CIFAR-10-OOD --parameter_path ./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params --posterior_path ./checkpoints/posterior_samples/CIFAR-10/ResNet_small/cifar_samples_1_seed_0_params --ood_batch_size 128 --num_samples_per_class 500