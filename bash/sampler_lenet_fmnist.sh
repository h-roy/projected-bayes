#!/bin/bash
#BSUB -J sample_resnets
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


python experiments/sample_classification.py --dataset FMNIST --model LeNet --checkpoint_path ./checkpoints/FMNIST/LeNet_FMNIST_0_params  --num_samples 30 --num_iterations 1000 --sample_batch_size 16 --macro_batch_size -1 --sample_seed 0 --run_name fmnist_samples
python experiments/sample_classification.py --dataset FMNIST --model LeNet --checkpoint_path ./checkpoints/FMNIST/LeNet_FMNIST_1_params  --num_samples 30 --num_iterations 1000 --sample_batch_size 16 --macro_batch_size -1 --sample_seed 1 --run_name fmnist_samples
python experiments/sample_classification.py --dataset FMNIST --model LeNet --checkpoint_path ./checkpoints/FMNIST/LeNet_FMNIST_2_params  --num_samples 30 --num_iterations 1000 --sample_batch_size 16 --macro_batch_size -1 --sample_seed 2 --run_name fmnist_samples

# python experiments/sample_lenet_fmnist.py --checkpoint_path ./checkpoints/FMNIST/LeNet_FMNIST_0_params --run_name fmnist_samples_1 --num_samples 30 --num_iterations 1000 --sample_batch_size 16
# python experiments/sample_lenet_fmnist.py --checkpoint_path ./checkpoints/FMNIST/LeNet_FMNIST_1_params --run_name fmnist_samples_2 --num_samples 30 --num_iterations 1000 --sample_batch_size 16
# python experiments/sample_classification.py --dataset MNIST --model LeNet --checkpoint_path ./checkpoints/MNIST/LeNet_MNIST_0_params  --num_samples 30 --num_iterations 1000 --sample_batch_size 16