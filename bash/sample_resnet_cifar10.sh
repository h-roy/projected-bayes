#!/bin/bash
#BSUB -J sample_resnets
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 60:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.11.4 cuda/12.3.2 cudnn/v8.9.1.23-prod-cuda-12.X
source proj/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python experiments/sample_classification.py --dataset CIFAR-10 --model ResNet_small --checkpoint_path ./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params --run_name cifar_samples_0 --num_samples 600 --num_iterations 150 --sample_batch_size 16 --vmap_dim 200

python experiments/sample_classification_ood.py --dataset CIFAR-10 --model ResNet_small --checkpoint_path ./checkpoints/CIFAR10/ResNet_small_CIFAR-10_0_params --run_name cifar_ood_0 --num_samples 100 --num_iterations 100 --sample_batch_size 16 --vmap_dim 100

python experiments/sample_classification.py --dataset CIFAR-10 --model ResNet_small --checkpoint_path ./checkpoints/CIFAR10/ResNet_small_CIFAR-10_1_params --run_name cifar_samples_1 --num_samples 600 --num_iterations 150 --sample_batch_size 16 --vmap_dim 200
python experiments/sample_classification.py --dataset CIFAR-10 --model ResNet_small --checkpoint_path ./checkpoints/CIFAR10/ResNet_small_CIFAR-10_2_params --run_name cifar_samples_2 --num_samples 600 --num_iterations 150 --sample_batch_size 16 --vmap_dim 200
