from os.path import join

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DeepSpeedStrategy


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    train_parameters: dict,
    *,
    seed: int = 42,
    save_interval: int = -1,
    project_name: str = "didi",
    ckpt_dir: str = None,
    resume: str = None,
):
    seed_everything(seed)

    wandb_logger = WandbLogger(project=project_name)
    wandb_logger.log_hyperparams(train_parameters)

    ckpt_dir = join(ckpt_dir, str(wandb_logger.experiment.id)) if ckpt_dir is not None else wandb_logger.experiment.dir
    checkpoint_callback = ModelCheckpoint(
        str(ckpt_dir),
        filename="step_{step}",
        every_n_train_steps=save_interval if save_interval > 0 else train_parameters["max_steps"],
        save_top_k=-1,
        auto_insert_metric_name=False,
    )

    lr_logger = LearningRateMonitor("step")

    strategy = DeepSpeedStrategy(stage=2, allgather_bucket_size=int(5e8), reduce_bucket_size=int(5e8))
    trainer = Trainer(
        accelerator="gpu",
        strategy=strategy,
        callbacks=[lr_logger, checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=10,
        **train_parameters,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=resume)
