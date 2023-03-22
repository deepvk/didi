from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    train_parameters: dict,
    *,
    seed: int = 42,
    save_interval: int = -1,
    project_name: str = "didi",
):
    seed_everything(seed)

    wandb_logger = WandbLogger(project=project_name)
    checkpoint_callback = ModelCheckpoint(
        str(wandb_logger.experiment.dir),
        filename="step_{step}",
        every_n_train_steps=save_interval if save_interval > 0 else train_parameters["max_steps"],
        save_top_k=-1,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )

    lr_logger = LearningRateMonitor("step")

    trainer = Trainer(
        callbacks=[lr_logger, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        **train_parameters,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
