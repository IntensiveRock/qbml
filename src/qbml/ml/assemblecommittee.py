import os
import pickle
from pathlib import Path
import torch
import click
from qbml.ml.committee import Committee


def assemble_the_committee(name, device, paths):
    # ONLY USE THE PATH FROM THE PARENT DIRECTORY FOR SAFETY.
    committee_save_path = Path(os.getcwd()) / "committees" / name
    os.mkdir(committee_save_path)
    os.mkdir(committee_save_path / "predictions")
    model = torch.load(paths[0], weights_only=False)
    with open(committee_save_path / 'committee.pkl', 'wb') as outp:
        assembled_committee = Committee(
            models=[Path(m_pth) for m_pth in paths],
            device=device,
            src_len=model.src_len,
            pred_len=model.pred_len,
        )
        pickle.dump(assembled_committee, outp, pickle.HIGHEST_PROTOCOL)
