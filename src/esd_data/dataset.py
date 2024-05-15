import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as torchvision_transforms

sys.path.append(".")
from src.preprocessing.subtile import Subtile
from src.utilities import SatelliteType


class ESDDataset(Dataset):
    """
    A PyTorch Dataset class for creating batches in a DataLoader.

    Parameters:
        processed_dir: Path to the processed directory
        transform: Compose of the transformations for training
        satellite_type_list: List of satellite types
        slice_size: Slice size for the subtiles

    Methods:
        __init__(processed_dir, transform, satellite_type_list, slice_size) -> None:
            Initializes the dataset and populates self.subtile_dirs.

        __len__() -> int:
            Returns the number of subtiles in the dataset.

        __getitem__(idx) -> Tuple[np.ndarray, np.ndarray]:
            Retrieves a subtile at idx, aggregates time bands, applies transformations (self.transform),
            and returns processed input (X) and ground truth (y).

    """

    def __init__(
        self,
        processed_dir: Path,
        transform: torchvision_transforms.transforms.Compose,
        satellite_type_list: List[SatelliteType],
        slice_size: Tuple[int, int] = (4, 4),
    ) -> None:
        self.transform = transform
        self.satellite_type_list = satellite_type_list
        self.slice_size = slice_size

        # --- start here ---
        # initialize a list for the subtile_dirs

        # iterate over the tiles in the processed_dir / subtiles

            # iterate over the subtiles within the tile

                # append the subtile to the subtile_dirs list
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns number of subtiles in the dataset

        Output: int
            length: number of subtiles in the dataset
        """
        # --- start here ---
        raise NotImplementedError

    def __aggregate_time(self, array: np.ndarray):
        """
        Aggregates time dimension in order to
        feed it to the machine learning model.

        This function needs to be changed in the
        final project to better suit your needs.

        For homework 2, you will simply stack the time bands
        such that the output is shaped (time*bands, width, height),
        i.e., all the time bands are treated as a new band.

        Input:
            img: np.ndarray
                (time, bands, width, height) array
        Output:
            new_img: np.ndarray
                (time*bands, width, height) array
        """
        # merge the time and bands dimension (hint: use np.stack or np.reshape)
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Loads subtile by its directory at index idx, then
            - aggregates times
            - performs self.transform

        Input:
            idx: int
                index of subtile with respect to self.subtile_dirs

        Output:
            X: np.ndarray | torch.Tensor
                input data to ML model, of shape (time*bands, width, height)
            y: np.ndarray | torch.Tensor
                ground truth, of shape (1, width, height)
        """
        # create a subtile and load by directory. The directory to load from will be the subtile_dirs at idx.

        # get the data_array_list and ground truth from the subtile satellite list and ground truth

        # convert items to their .values form, we are stripping away the xarray so we can feed the data into the model

        # initalize a list to store X

        # iterate over each array in the stripped array from above

            # aggregate time and append the array to X


        # concatenate X

        # set y to be the ground truth data array .values squeezed on the 0 and 1 axis


        # if the transform is not none

            # apply the transform to X and y and store the result

            # set X to be the result for X

            # set y to be the result for y


        # return X and y-1, labels go from 1-4, so y-1 makes them zero indexed
        raise NotImplementedError