from flax import linen as nn
from src.models import ResNetBlock_small
# Model Hyperparameters

def get_model_hyperparams(n_classes, model_name):
    if model_name == "ResNet18":
        hparams = {
            "num_classes": n_classes,
        }
    elif model_name in ["DenseNet", "GoogleNet"]:
        hparams = {
            "num_classes": n_classes,
            "act_fn": nn.relu
        }
    elif model_name == "VisionTransformer":
        hparams = {
            "embed_dim": 256,
            "hidden_dim": 512,
            "num_heads": 8,
            "num_layers": 6,
            "patch_size": 4,
            "num_channels": 3,
            "num_patches": 64,
            "num_classes": n_classes,
            "dropout_prob": 0.1,
        }
    elif model_name == "ResNet_small":
        hparams = {
            "num_classes": n_classes,
            "c_hidden": (16, 32, 64),
            "num_blocks": (3, 3, 3),
            "act_fn": nn.relu,
            "block_class": ResNetBlock_small
        }
    elif model_name == "MLP":
        hparams = {
            "n_classes": n_classes,
            "hidden_dim": 64,
            "num_layers": 3
        }
    elif model_name == "LeNet":
        hparams = {
            "output_dim": n_classes,
            "activation": "tanh"
        }
    else:
        raise ValueError(f"Configs for Model {model_name} not implemented yet.")
    return hparams
# MLP
    # {"n_classes": n_classes, "hidden_dim": 64, "num_layers": 3}
            # if args["model"] == "MLP"
            # else {
            #     "n_classes": n_classes,
            # }

# Optimizer Hyperparameters

def get_optimizer_hyperparams(model_name):
    if model_name in ["GoogleNet", "DenseNet"]:
        optimizer_name = "adamw"
        optimizer_hparams={"lr": 1e-3,
                            "weight_decay": 1e-4}
    elif model_name == "VisionTransformer":
        optimizer_name = "adamw"
        optimizer_hparams={"lr": 3e-4, 
                            "weight_decay": 0.01}
    elif model_name == "ResNet_small":
        optimizer_name = "SGD"
        optimizer_hparams={"lr": 0.1,
                           "momentum": 0.9,
                           "weight_decay": 1e-4}
    elif model_name == "ResNet18":
        optimizer_name = "SGD"
        optimizer_hparams={"lr": 1e-3,
                           "momentum": 0.9,
                           "weight_decay": 1e-4}
    elif model_name in ["LeNet", "MLP"]:
        optimizer_name = "adamw"
        optimizer_hparams={"lr": 0.01,
                           "weight_decay": 0.001}
    else:
        raise ValueError(f"Configs for Model {model_name} not implemented yet.")

    return optimizer_name, optimizer_hparams

def get_model_apply_fn(model_name, model_apply, batch_stats=None, rng=None):
    if model_name in ["ResNet_small", "ResNet18", "DenseNet", "GoogleNet"]:
        assert batch_stats is not None, "Batch statistics must be provided for ResNet and DenseNet models."
        model_fn = lambda params, imgs: model_apply({'params': params, 'batch_stats': batch_stats}, 
                                    imgs,
                                    train=False,
                                    mutable=False)
    elif model_name in ["LeNet", "MLP"]:
        model_fn = model_apply
    elif model_name == "VisionTransformer":
        assert rng is not None, "RNG key must be provided for Vision Transformer model."
        model_fn = lambda params, imgs: model_apply({'params': params},
                                                                imgs,
                                                                train=False,
                                                                rngs={'dropout': rng})
    else:
        raise ValueError(f"Configs for Model {model_name} not implemented yet.")
    
    return model_fn