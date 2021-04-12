import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from laughing_net.models.LightBERT.model import LightBERT
from laughing_net.utils.get_paths import get_data_paths
from laughing_net.logger import init_logger
from laughing_net.config import params

def train(task="task-1"):
    init_logger(tags=["debug"])

    data_paths = get_data_paths(task)

    model = LightBERT(
        train_file_path=data_paths['train'],
        dev_file_path=data_paths['dev'],
        test_file_path=data_paths['test'],
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        verbose=True,
        mode="min"
    )

    trainer = pl.Trainer(
        checkpoint_callback=early_stop_callback,
        deterministic=True,
        logger=False,
        max_epochs=params.models.FunBERT.epochs,
        gpus=0
    )

    trainer.fit(model)

if __name__ == "__main__":
    train()