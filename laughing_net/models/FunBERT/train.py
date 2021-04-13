import pytorch_lightning as pl

from laughing_net.config import params
from laughing_net.logger import init_logger
from laughing_net.utils.lightning import DVCLiveCompatibleModelCheckpoint, DVCLiveNextStepCallback
from laughing_net.models.FunBERT.data import FunDataModule
from laughing_net.models.FunBERT.model import FunBERT

def train():
    init_logger(tags=["debug", "FunBERT"])

    model = FunBERT(
        lr=params.models.FunBERT.learning_rate,
        model_type=params.models.FunBERT.type,
    )
    dm = FunDataModule(model_type=params.models.FunBERT.type)

    checkpoint_callback = DVCLiveCompatibleModelCheckpoint(
        dirpath="artifacts",
        filename="FunBERT",
        save_top_k=-1,
        verbose=True,
    )

    dvclive_next_step_callback = DVCLiveNextStepCallback()

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        deterministic=True,
        logger=False,
        max_epochs=params.models.FunBERT.epochs,
        gpus=1,
        callbacks=[dvclive_next_step_callback]
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    train()