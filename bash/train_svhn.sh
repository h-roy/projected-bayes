#!/bin/bash
#BSUB -J train_small_data
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

python experiments/train_classification.py --model ResNet_small --dataset SVHN --trainer Classification --run_name SVHN --seed 0
python experiments/train_classification.py --model ResNet_small --dataset SVHN --trainer Classification --run_name SVHN --seed 1
python experiments/train_classification.py --model ResNet_small --dataset SVHN --trainer Classification --run_name SVHN --seed 2

python experiments/train_classification.py --model VisionTransformer --dataset SVHN --trainer VIT --run_name SVHN --seed 0
python experiments/train_classification.py --model VisionTransformer --dataset SVHN --trainer VIT --run_name SVHN --seed 1
python experiments/train_classification.py --model VisionTransformer --dataset SVHN --trainer VIT --run_name SVHN --seed 2

python experiments/train_classification.py --model DenseNet --dataset SVHN --trainer Classification --run_name SVHN --seed 0
python experiments/train_classification.py --model DenseNet --dataset SVHN --trainer Classification --run_name SVHN --seed 1
python experiments/train_classification.py --model DenseNet --dataset SVHN --trainer Classification --run_name SVHN --seed 2

python experiments/train_classification.py --model GoogleNet --dataset SVHN --trainer Classification --run_name SVHN --seed 0
python experiments/train_classification.py --model GoogleNet --dataset SVHN --trainer Classification --run_name SVHN --seed 1
python experiments/train_classification.py --model GoogleNet --dataset SVHN --trainer Classification --run_name SVHN --seed 2
