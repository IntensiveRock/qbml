import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class Committee:
    """
    Committee of several trained neural networks.

    A collection of nns that will be trained ahead of time.
    Functionality for training will be implemented later.

    This class is purely for predictions at the moment.

    :param models: A list of paths to the models in the networks.
    :type models: list[Path, ...]
    """

    def __init__(
        self,
        models: list,
        src_len: int,
        pred_len: int,
        device: str = "cpu",
    ):
        """Initialize the Committee."""
        super().__init__()
        self.models = models
        self.num_of_models = len(models)
        self.device = device
        self.src_len = src_len
        self.pred_len = pred_len

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> torch.Tensor:
        """
        Define the way the committee makes predictions.

        There are two outputs of this function:
            - The average over committee member prediction.
            - The actual predictions made by all members so you can collect STDEV.
        """
        models = [torch.load(self.models[i], weights_only=False, map_location=torch.device('cpu')) for i in range(self.num_of_models)]
        batch_size = dataloader.batch_size
        # batch_preds = torch.zeros(
        #     (len(dataloader),
        #      self.num_of_models,
        #      dataloader.batch_size,
        #      models[0].pred_len,
        #      2)
        # )
        batch_preds = torch.zeros(
            (self.num_of_models,
             len(dataloader)*batch_size,
             models[0].pred_len,
             2)
        )
        sum_preds = torch.zeros((len(dataloader)*batch_size, 400, 2))
        for batch, (src, tgt) in tqdm(enumerate(dataloader)):
            for m_idx, enc in enumerate(models):
                enc.device = self.device
                enc = enc.to(device=enc.device)
                src = src.to(device=enc.device)
                logits = enc(src)
                logits = logits.detach().cpu()
                start_idxs = batch_size*batch
                sum_preds[start_idxs:start_idxs+batch_size] += logits
                # batch_preds[batch, m_idx] = logits
                batch_preds[m_idx, start_idxs:start_idxs+batch_size] += logits
        return sum_preds / self.num_of_models, batch_preds
