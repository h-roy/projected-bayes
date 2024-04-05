import jax.random as random


def train(*args, trainermodule, train_loader, val_loader, test_loader, num_epochs=200, **kwargs):
    # Create a trainer module with specified hyperparameters
    trainer = trainermodule(*args, **kwargs)
    if not trainer.checkpoint_exists():  # Skip training if pretrained model exists
        trainer.train_model(train_loader, val_loader, num_epochs=num_epochs)
        trainer.load_model()
    else:
        trainer.load_model(pretrained=True)
    # Test trained model
    val_acc = trainer.eval_model(val_loader, eval_type="val_set")
    test_acc = trainer.eval_model(test_loader, eval_type="test_set")
    return trainer, {"val": val_acc, "test": test_acc}