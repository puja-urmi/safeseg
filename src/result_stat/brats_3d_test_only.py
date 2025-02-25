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
from collections.abc import Hashable, Mapping
from monai.metrics import DiceMetric
from monai.config import KeysCollection
from monai.networks.nets.segresnet import SegResNet
from monai.transforms import Transform
from monai.transforms.transform import MapTransform
from monai.utils.enums import TransformBackends
from monai.data.utils import NdarrayOrTensor
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)



class ConvertToMultiChannelBasedOnBrats24Classes(Transform):
    """
    Convert labels to multi channels based on BraTS24 segmentation classes:
    - Label 1: Necrotic and non-enhancing tumor core (NETC)
    - Label 2: Surrounding non-enhancing FLAIR hyperintensity (SNFH)
    - Label 3: Enhancing tissue (ET)
    - Label 4: Resection cavity (RC)

    Output channels:
    - C1: ET (Enhancing Tissue) -> (label == 3)
    - C2: NETC (Necrotic and Non-Enhancing Tumor Core) -> (label == 1)
    - C3: SNFH (Surrounding Non-Enhancing FLAIR Hyperintensity) -> (label == 2)
    - C4: RC (Resection Cavity) -> (label == 4)
    - C5: Combined ET + NETC -> (label == 3) OR (label == 1)
    - C6: Combined ET + SNFH + NETC -> (label == 3) OR (label == 2) OR (label == 1)
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Convert input segmentation labels into multi-channel format based on BraTS24 classes.

        Args:
            img (NdarrayOrTensor): Input segmentation map. Expected shape:
                - 3D: (H, W, D)
                - 4D with channel: (C, H, W, D), where C=1 (will be squeezed).

        Returns:
            NdarrayOrTensor: Multi-channel segmentation map with shape (6, H, W, D).
        """
        # If the input has a channel dimension (C=1), remove it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # Define the channels based on label values (BraTS24 logic)
        result = [
            (img == 3),                            # C1: ET
            (img == 1),                            # C2: NETC
            (img == 2),                            # C3: SNFH
            (img == 4),                            # C4: RC
            (img == 3) | (img == 1),               # C5: ET + NETC
            (img == 3) | (img == 2) | (img == 1),  # C6: ET + SNFH + NETC
        ]

        return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBrats24Classes`.
    """

    backend = ConvertToMultiChannelBasedOnBrats24Classes.backend

    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvertToMultiChannelBasedOnBrats24Classes()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
    

def main():
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--dataset_base_dir", default="/home/psaha03/scratch/dataset_brats24/dataset", type=str)
    parser.add_argument("--datalist_json_path", default="/home/psaha03/scratch/dataset_brats24/datalist/site-test.json", type=str)
    args = parser.parse_args()

    # Set basic settings and paths
    dataset_base_dir = args.dataset_base_dir
    datalist_json_path = args.datalist_json_path
    model_path = args.model_path
    infer_roi_size = (168, 208, 168)

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
        in_channels=4,
        out_channels=6,
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
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
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
        metric_et = 0
        metric_netc = 0
        metric_snfh = 0
        metric_rc = 0
        metric_tc = 0
        metric_wt = 0
        ct = 0
        ct_et = 0
        ct_netc = 0
        ct_snfh = 0
        ct_rc = 0
        ct_tc = 0
        ct_wt = 0
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
                metric_et += metric_score[0][0].item()
                ct_et += 1
            if not np.isnan(metric_score[0][1].item()):
                metric += metric_score[0][1].item()
                ct += 1
                metric_netc += metric_score[0][1].item()
                ct_netc += 1
            if not np.isnan(metric_score[0][2].item()):
                metric += metric_score[0][2].item()
                ct += 1
                metric_snfh += metric_score[0][2].item()
                ct_snfh += 1
            if not np.isnan(metric_score[0][3].item()):
                metric += metric_score[0][3].item()
                ct += 1
                metric_rc += metric_score[0][3].item()
                ct_rc += 1
            if not np.isnan(metric_score[0][4].item()):
                metric += metric_score[0][4].item()
                ct += 1
                metric_tc += metric_score[0][4].item()
                ct_tc += 1
            if not np.isnan(metric_score[0][5].item()):
                metric += metric_score[0][5].item()
                ct += 1
                metric_wt += metric_score[0][5].item()
                ct_wt += 1
        # compute mean dice over whole validation set
        metric_et /= ct_et
        metric_netc /= ct_netc
        metric_snfh /= ct_snfh
        metric_rc /= ct_rc
        metric_tc /= ct_tc
        metric_wt /= ct_wt
        metric /= ct
        print(f"Test Dice: {metric:.4f}, Valid count: {ct}")
        print(f"Test Dice TC: {metric_et:.4f}, Valid count: {ct_et}")
        print(f"Test Dice WT: {metric_netc:.4f}, Valid count: {ct_netc}")
        print(f"Test Dice ET: {metric_snfh:.4f}, Valid count: {ct_snfh}")
        print(f"Test Dice TC: {metric_rc:.4f}, Valid count: {ct_rc}")
        print(f"Test Dice WT: {metric_tc:.4f}, Valid count: {ct_tc}")
        print(f"Test Dice ET: {metric_wt:.4f}, Valid count: {ct_wt}")


if __name__ == "__main__":
    main()
