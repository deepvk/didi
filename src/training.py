from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    device,
    num_devices,
    num_steps,
    logging_step,
    val_interval,
    seed: int = 42,
    project_name: str = "didi",
):
    seed_everything(seed)

    wandb_logger = WandbLogger(project=project_name)
    checkpoint_callback = ModelCheckpoint(
        wandb_logger.experiment.dir,
        filename="step_{step}",
        every_n_train_steps=val_interval,
        save_top_k=-1,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )

    lr_logger = LearningRateMonitor("step")

    trainer = Trainer(
        accelerator=device,
        devices=num_devices,
        callbacks=[lr_logger, checkpoint_callback],
        log_every_n_steps=logging_step,
        logger=wandb_logger,
        max_steps=num_steps,
        val_check_interval=val_interval,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
