from torch import Tensor
from torch.utils.data import Dataset


class TomoPredictionSet(Dataset):
    def __init__(
            self,
            averages : Tensor,
            indp_predictions : Tensor,
            set_freq_axis : Tensor,
            transform = None,
    ):
        self.averages = averages
        self.indp_predictions = indp_predictions
        self.freq_axis = set_freq_axis
        self.transform = transform

    def __len__(self):
        """Redefine len to return the number of simulations in the set."""
        return len(self.averages)

    def __getitem__(self, idx: int = None) -> Tensor:
        """
        Define getitem to agree with the Pytorch API.

        :param idx: index of the tomography/spec_den pair to retrieve.
        :type idx: int
        :returns: The tomography and spec_dens at the provided index.
        :rtype: torch.Tensor
        
        """
        if self.transform:
            average_item = self.transform(self.averages[idx])
            average_item = average_item.float()
            indp_item = self.transform(self.indp_predictions[:, idx])
            indp_item = indp_item.float()
        else:
            average_item = self.averages[idx]
            average_item = average_item.float()
            indp_item = self.indp_predictions[:, idx]
            indp_item = indp_item.float()
        return average_item, indp_item


# def construct_predictedtomo_loader(
#         tomography_set : TomographyDataSet,
#         mdl_input_seq_len: int,
#         mdl_target_seq_len : int,
#         shuffle : bool = False,
#         batch_size: int = None,
#         split: list = None,
# ) -> DataLoader:
