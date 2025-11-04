import logging
import os
import shutil
from pathlib import Path
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from qbml.ml.transformer import Transformer
import qbml.ml.tomographydataset as tgd
import qbml.ml.lossfuncs as qmloss

from qbml.dynamics.tools import count_parameters


@hydra.main(version_base=None)
def main(cfg: DictConfig):
    """
    Train QubitML neural network.

    :param cfg: Configuration file containing hyperparmeters and paths to data
    :type cfg: DictConfig
    """
    
    training_history_path = Path(cfg.prj_dir) / 'models' / cfg.title
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    os.mkdir(training_history_path)
    os.mkdir(training_history_path / 'predictions')

    # Define the params of the training.
    BATCH_SIZE = cfg.wandb_cfg.config.batch_size
    EPOCHS = cfg.wandb_cfg.config.epochs
    DEVICE = cfg.cfg_info.device

    # Create dataloader from dataset.
    tomo_dataset = torch.load(f'{cfg.set_path}', weights_only=False)
    tomo, spd = tomo_dataset[0]
    src_len = len(tomo)
    train_loader, validation_loader = tgd.construct_qubitml_dataloader(
        tomography_set=tomo_dataset,
        mdl_input_seq_len=cfg.wandb_cfg.config.input_seq_len,
        mdl_target_seq_len=cfg.wandb_cfg.config.target_seq_len,
        shuffle=True,
        batch_size=BATCH_SIZE,
        split=cfg.dataset.split,
    )
    # Spawn the model.
    model = Transformer(
        n_encoders=cfg.wandb_cfg.config.n_encoders,
        n_embd=3,
        n_heads=3,
        src_len=src_len,
        n_tgt=2,
        pred_len=cfg.wandb_cfg.config.target_seq_len,
        device=DEVICE,
        rectifier=cfg.wandb_cfg.config.rectifier,
        block_size=cfg.wandb_cfg.config.block_size,
    )
    model = model.to(device=DEVICE)
    loss_fn = qmloss.MSELoss_Positive_Definite(
        negative_value_punishment=cfg.wandb_cfg.config.loss_nvp
    )
    count_parameters(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.wandb_cfg.config.learning_rate,
        weight_decay=cfg.wandb_cfg.config.weight_decay,
    )
    with wandb.init(
            project=cfg.wandb_cfg.project,
            entity=cfg.wandb_cfg.entity,
            job_type=cfg.wandb_cfg.job_type,
            name=cfg.wandb_cfg.name,
            config=OmegaConf.to_object(cfg.wandb_cfg.config)
    ) as run:
        for epoch in tqdm(range(EPOCHS)):
            train_loss = model.train_loop(train_loader, loss_fn, optimizer)
            val_loss = model.val_loop(validation_loader, loss_fn)
            run.log({"training loss" : train_loss})
            run.log({"validation loss" : val_loss})

    torch.save(model, training_history_path / 'model.pth')
    wandb.finish()
    shutil.move(output_dir / '.hydra', training_history_path / 'hydra')


if __name__ == "__main__":
    main()
