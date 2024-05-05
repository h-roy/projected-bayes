from src.models.mlp import MLP
from src.models.convnet import ConvNet
from src.models.lenet import LeNet
from src.models.resnet_small import ResNet_small, ResNetBlock_small, PreActResNetBlock_small
from src.models.densenet import DenseNet, DenseBlock, DenseLayer, TransitionLayer
from src.models.inception import InceptionBlock, GoogleNet
from src.models.vit import VisionTransformer, AttentionBlock
from src.models.utils import load_model

MODELS_DICT = {
    "MLP": "src.models.MLP",
    "LeNet": "src.models.LeNet",
    "ResNet_small": "src.models.ResNet_small",
    "ResNet18": "src.models.resnet_18.ResNet18",
    "DenseNet": "src.models.DenseNet",
    "GoogleNet": "src.models.GoogleNet",
    "VisionTransformer": "src.models.VisionTransformer",
}