### Installation
To install the requirements just run:
```bash
sh setup.sh
```

Every time you start the SSH session, you just need to load the modules and activate the virtual environment by running:
```bash
module load python3/3.11.4 cuda/12.3.2 cudnn/v8.9.1.23-prod-cuda-12.X
source proj/bin/activate

# Whichever / however many GPUs you want to use
export CUDA_VISIBLE_DEVICES=1
# Don't preallocate memory on interactive clusters
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```


### Sampler

To sample from the projected posterior, simply run:

```bash
python experiments/sample_classification.py --dataset DATASET --model MODEL --checkpoint_path CHECKPOINT_PATH  --num_samples NUM_SAMPLES --num_iterations NUM_ITERS
```


