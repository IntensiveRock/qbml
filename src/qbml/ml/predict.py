import pickle
import torch
from pathlib import Path
from qbml.ml.tomographydataset import construct_qubitml_dataloader
from qbml.ml.predictionset import TomoPredictionSet


def prediction(mdlpth, datapth, bs, device, is_model=True):
    mdlpth = Path(mdlpth)
    datapth = Path(datapth)
    file_name = datapth.stem + '.prd'
    prediction_save_pth = mdlpth.parent / 'predictions'
    if is_model:
        model = torch.load(mdlpth, weights_only=False, map_location=torch.device(device))
        model.device = device
        model.to(device=model.device)
    else:
        model = pickle.load(open(mdlpth, 'rb'))
    tomo_dataset = torch.load(datapth, weights_only=False)
    prediction_loader = construct_qubitml_dataloader(
        tomography_set=tomo_dataset,
        mdl_input_seq_len=model.src_len,
        mdl_target_seq_len=model.pred_len,
        shuffle=False,
        batch_size=bs,
    )
    if is_model:
        predictions = model.predict(prediction_loader)
        torch.save(predictions, prediction_save_pth / file_name)
    else:
        committee_averages, model_predictions = model.predict(prediction_loader)
        prediction_set = TomoPredictionSet(committee_averages,
                                           model_predictions,
                                           tomo_dataset.freq_axis)
        torch.save(prediction_set, prediction_save_pth / file_name)


# if __name__ == "__main__":
#     main()
