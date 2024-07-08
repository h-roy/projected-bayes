#!/bin/bash
#BSUB -J sample_mnist
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -W 2:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.11.4 cuda/12.3.2 cudnn/v8.9.1.23-prod-cuda-12.X
source proj/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python experiments/sample_classification.py --dataset MNIST --model LeNet --checkpoint_path ./checkpoints/MNIST/LeNet_MNIST_0_params  --num_samples 30 --num_iterations 1000 --sample_batch_size 16 --macro_batch_size -1 --sample_seed 0 --run_name mnist_samples
python experiments/sample_classification.py --dataset MNIST --model LeNet --checkpoint_path ./checkpoints/MNIST/LeNet_MNIST_1_params  --num_samples 30 --num_iterations 1000 --sample_batch_size 16 --macro_batch_size -1 --sample_seed 1 --run_name mnist_samples
python experiments/sample_classification.py --dataset MNIST --model LeNet --checkpoint_path ./checkpoints/MNIST/LeNet_MNIST_2_params  --num_samples 30 --num_iterations 1000 --sample_batch_size 16 --macro_batch_size -1 --sample_seed 2 --run_name mnist_samples

# python experiments/sample_lenet_mnist.py --checkpoint_path ./checkpoints/MNIST/LeNet_MNIST_0_params --run_name mnist_samples_1 --num_samples 30 --num_iterations 1000 --sample_batch_size 16