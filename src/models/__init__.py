from src.models.mlp import MLP
from src.models.convnet import ConvNet
from src.models.lenet import LeNet
from src.models.resnet_small import ResNet_small, ResNetBlock_small, PreActResNetBlock_small
from src.models.densenet import DenseNet, DenseBlock, DenseLayer, TransitionLayer
from src.models.inception import InceptionBlock, GoogleNet
from src.models.vit import VisionTransformer, AttentionBlock
from src.models.unet import Unet, get_ddpm_params
# from src.models.diffusion_utils import create_ema_decay_schedule, TrainState, p_loss, copy_params_to_ema, apply_ema_decay, sample_loop, ddpm_sample_step
from src.models.utils import load_model

MODELS_DICT = {
    "MLP": "src.models.MLP",
    "LeNet": "src.models.LeNet",
    "ResNet_small": "src.models.ResNet_small",
    "ResNet18": "src.models.resnet_18.ResNet18",
    "ResNet34": "src.models.resnet_18.ResNet34",
    "ResNet50": "src.models.resnet_18.ResNet50",
    "ResNet101": "src.models.resnet_18.ResNet101",
    "ResNet152": "src.models.resnet_18.ResNet152",
    "ResNet200": "src.models.resnet_18.ResNet200",
    "DenseNet": "src.models.DenseNet",
    "GoogleNet": "src.models.GoogleNet",
    "VisionTransformer": "src.models.VisionTransformer",
}