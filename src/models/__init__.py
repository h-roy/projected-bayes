from src.models.mlp import MLP
from src.models.lenet import LeNet
from src.models.resnet import ResNet, ResNetBlock, PreActResNetBlock
from src.models.densenet import DenseNet, DenseBlock, DenseLayer, TransitionLayer
from src.models.inception import InceptionBlock, GoogleNet
from src.models.vit import VisionTransformer, AttentionBlock
from src.models.utils import load_model

MODELS_DICT = {
    "MLP": "src.models.MLP",
    "LeNet": "src.models.LeNet",
    "ResNet": "src.models.ResNet",
    "DenseNet": "src.models.DenseNet",
    "GoogleNet": "src.models.GoogleNet",
    "VisionTransformer": "src.models.VisionTransformer",
}