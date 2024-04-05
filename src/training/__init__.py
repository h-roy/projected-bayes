from .train_utils import train
from .classification_trainer import Classification_Trainer
from .vit_trainer import VIT_Trainer
TRAINERS = {
    "Classification": Classification_Trainer,
    "VIT": VIT_Trainer
}