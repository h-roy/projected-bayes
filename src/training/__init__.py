from .train_utils import train
from .classification_trainer import Classification_Trainer
from .vit_trainer import VIT_Trainer
from .lenet_trainer import LeNet_Trainer
from .configs import get_model_hyperparams, get_optimizer_hyperparams
TRAINERS = {
    "Classification": Classification_Trainer,
    "VIT": VIT_Trainer,
    "lenet": LeNet_Trainer
}