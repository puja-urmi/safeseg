# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import numpy as np
import torch
from monai.data import DataLoader, Dataset, load_decathlon_datalist
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from collections.abc import Hashable, Mapping
from monai.transforms import Transform
from monai.config import KeysCollection
from monai.networks.nets.segresnet import SegResNet
from monai.transforms.transform import MapTransform
from monai.utils.enums import TransformBackends
from monai.data.utils import NdarrayOrTensor
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)




class ConvertToMultiChannelBasedOnKits23Classes(Transform):
    """
    Convert labels to multi channels based on KiTS23 segmentation classes:
    - Label 1: Kidney
    - Label 2: Tumor
    - Label 3: Cyst

    Output channels:
    - C1: Kidney + Tumor + Cyst -> (label == 1) OR (label == 2) OR (label == 3)
    - C2: Tumor + Cyst -> (label == 2) OR (label == 3)
    - C3: Tumor -> (label == 2)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Convert input segmentation labels into multi-channel format based on kits23 classes.

        Args:
            img (NdarrayOrTensor): Input segmentation map. Expected shape:
                - 3D: (H, W, D)
                - 4D with channel: (C, H, W, D), where C=1 (will be squeezed).

        Returns:
            NdarrayOrTensor: Multi-channel segmentation map with shape (3, H, W, D).
        """
        # If the input has a channel dimension (C=1), remove it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Define the channels based on label values (BraTS24 logic)
        result = [
            (img == 1) | (img == 2) | (img == 3),  # C1: Kidney + Tumor + Cyst
            (img == 2) | (img == 3),               # C2: Tumor + Cyst
            (img == 2)                             # C3: Tumor
        ]

        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class ConvertToMultiChannelBasedOnKitsClassesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnKits23Classes`.
    """

    backend = ConvertToMultiChannelBasedOnKits23Classes.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnKits23Classes()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def main():
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_base_dir", default="/home/psaha03/scratch/dataset_kits23/dataset", type=str)
    parser.add_argument("--datalist_json_path", default="/home/psaha03/scratch/dataset_kits23/datalist/site-test.json", type=str)
    args = parser.parse_args()

    # Set basic settings and paths
    dataset_base_dir = args.dataset_base_dir
    datalist_json_path = args.datalist_json_path
    model_path = args.model_path
    infer_roi_size = (208, 128, 168)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set datalists
    test_list = load_decathlon_datalist(
        data_list_file_path=datalist_json_path,
        is_segmentation=True,
        data_list_key="validation",
        base_dir=dataset_base_dir,
    )
    print(f"Testing Size: {len(test_list)}")

    # Network, optimizer, and loss
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=1,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)
    model_weights = torch.load(model_path)
    model_weights = model_weights["model"]
    model.load_state_dict(model_weights)

    # Inferer, evaluation metric
    inferer = SlidingWindowInferer(roi_size=infer_roi_size, sw_batch_size=1, overlap=0.5)
    valid_metric = DiceMetric(include_background=True, reduction="mean")

    transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            ConvertToMultiChannelBasedOnKitsClassesd(keys="label"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.5, 0.5, 0.5),
                mode=("bilinear", "nearest"),
            ),
            DivisiblePadd(keys=["image", "label"], k=32),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    transform_post = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Set dataset
    test_dataset = Dataset(data=test_list, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    # Train
    model.eval()
    with torch.no_grad():
        metric = 0
        metric_tc = 0
        metric_wt = 0
        metric_et = 0
        smooth = 1e-6
        ct = 0
        ct_tc = 0
        ct_wt = 0
        ct_et = 0
        for i, batch_data in enumerate(test_loader):
            images = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)
            # Inference
            outputs = inferer(images, model)
            outputs = transform_post(outputs)
            # Compute metric
            metric_score = valid_metric(y_pred=outputs, y=labels)
            if not np.isnan(metric_score[0][0].item()):
                metric += metric_score[0][0].item()
                ct += 1
                metric_tc += metric_score[0][0].item()
                ct_tc += 1
            if not np.isnan(metric_score[0][1].item()):
                metric += metric_score[0][1].item()
                ct += 1
                metric_wt += metric_score[0][1].item()
                ct_wt += 1
            if not np.isnan(metric_score[0][2].item()):
                metric += metric_score[0][2].item()
                ct += 1
                metric_et += metric_score[0][2].item()
                ct_et += 1
        # compute mean dice over whole validation set
        metric_tc /= ct_tc + smooth
        metric_wt /= ct_wt + smooth
        metric_et /= ct_et + smooth
        metric /= ct
        print(f"Test Dice: {metric:.4f}, Valid count: {ct}")
        print(f"Test Dice KTC: {metric_tc:.4f}, Valid count: {ct_tc}")
        print(f"Test Dice TC: {metric_wt:.4f}, Valid count: {ct_wt}")
        print(f"Test Dice T: {metric_et:.4f}, Valid count: {ct_et}")


if __name__ == "__main__":
    main()
