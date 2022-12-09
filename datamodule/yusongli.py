import os

import numpy as np
import pytorch_lightning as pl
from monai.data import CacheDataset, DataLoader, Dataset, PersistentDataset, load_decathlon_datalist, partition_dataset
from monai.transforms import (
    AddChanneld,
    Compose,
    DeleteItemsd,
    FgBgToIndicesd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)


class YuSongliDataModule(pl.LightningDataModule):

    # ! <<< open debug yusongli
    # class_weight = np.asarray([0.01361341, 0.47459406, 0.51179253])
    # ! ===
    class_weight = np.asarray([0.0363463, 0.9636537])
    # ! >>> clos debug

    def __init__(
        self,
        root_dir=".",
        fold=0,
        train_patch_size=(32, 32, 32),
        num_samples=32,
        batch_size=1,
        cache_rate=None,
        cache_dir=None,
        num_workers=4,
        balance_sampling=True,
        train_transforms=None,
        val_transforms=None,
        **kwargs
    ):
        super().__init__()
        self.base_dir = root_dir
        self.fold = fold
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers

        if balance_sampling:
            pos = neg = 0.5
        else:
            pos = np.sum(self.class_weight[1:])
            neg = self.class_weight[0]

        if train_transforms is None:
            self.train_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                    AddChanneld(keys=["image", "label"]),
                    # ! <<< open debug yusongli
                    # Orientationd(keys=["image", "label"], axcodes="LPI"),
                    # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                    # ! ===
                    SpatialPadd(keys=["image", "label"], spatial_size=train_patch_size, mode="edge"),
                    FgBgToIndicesd(keys=["label"], image_key="image"),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        label_key="label",
                        spatial_size=train_patch_size,
                        pos=pos,
                        neg=neg,
                        num_samples=num_samples,
                        fg_indices_key="label_fg_indices",
                        bg_indices_key="label_bg_indices",
                    ),
                    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
                    DeleteItemsd(keys=["label_fg_indices", "label_bg_indices"]),
                    # ! >>> clos debug
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.train_transforms = train_transforms

        if val_transforms is None:
            self.val_transforms = Compose(
                [
                    LoadImaged(keys=["image", "label"], reader="NibabelReader"),
                    AddChanneld(keys=["image", "label"]),
                    # ! <<< open debug yusongli
                    # Orientationd(keys=["image", "label"], axcodes="LPI"),
                    # ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                    # ! ===
                    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1500, b_min=0.0, b_max=1.0, clip=True),
                    # ! >>> clos debug
                    ToTensord(keys=["image", "label"]),
                ]
            )
        else:
            self.val_transforms = val_transforms

    def _load_data_dicts(self, train=True):
        if train:
            # ! <<< open debug yusongli
            # data_dicts = load_decathlon_datalist(
            #     os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
            # )
            # data_dicts_list = partition_dataset(data_dicts, num_partitions=4, shuffle=True, seed=0)
            # train_dicts, val_dicts = [], []
            # for i, data_dict in enumerate(data_dicts_list):
            #     if i == self.fold:
            #         val_dicts.extend(data_dict)
            #     else:
            #         train_dicts.extend(data_dict)
            # ! ===
            import pickle
            splits_final = '/home/yusongli/_dataset/shidaoai/img/_out/nn/DATASET/nnUNet_preprocessed/Task607_CZ2/splits_final.pkl'
            with open(splits_final, 'rb') as f:
                splits_final = pickle.load(f)
            train_dicts = [
                {
                    'image': f'/home/yusongli/_dataset/shidaoai/img/_out/nn/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task607_CZ2/imagesTr/{idx}_0000.nii.gz',
                    'label': f'/home/yusongli/_dataset/shidaoai/img/_out/nn/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task607_CZ2/labelsTr/{idx}.nii.gz',
                }
                for idx in splits_final[0]['train']
                # for i, idx in enumerate(splits_final[0]['train']) if i <= 100
            ]
            val_dicts = [
                {
                    'image': f'/home/yusongli/_dataset/shidaoai/img/_out/nn/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task607_CZ2/imagesTr/{idx}_0000.nii.gz',
                    'label': f'/home/yusongli/_dataset/shidaoai/img/_out/nn/DATASET/nnUNet_raw_data_base/nnUNet_raw_data/Task607_CZ2/labelsTr/{idx}.nii.gz',
                }
                for idx in splits_final[0]['val']
                # for i, idx in enumerate(splits_final[0]['val']) if i <= 20
            ]
            # ! >>> clos debug
            return train_dicts, val_dicts
        else:
            pass

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_data_dicts, val_data_dicts = self._load_data_dicts()

            if self.cache_rate is not None:
                self.trainset = CacheDataset(
                    data=train_data_dicts,
                    transform=self.train_transforms,
                    cache_rate=self.cache_rate,
                    num_workers=self.num_workers,
                )
                self.valset = CacheDataset(
                    data=val_data_dicts, transform=self.val_transforms, cache_rate=1.0, num_workers=4
                )
            elif self.cache_dir is not None:
                self.trainset = PersistentDataset(
                    data=train_data_dicts, transform=self.train_transforms, cache_dir=self.cache_dir
                )
                self.valset = PersistentDataset(
                    data=val_data_dicts, transform=self.val_transforms, cache_dir=self.cache_dir
                )
            else:
                self.trainset = Dataset(data=train_data_dicts, transform=self.train_transforms)
                self.valset = Dataset(data=val_data_dicts, transform=self.val_transforms)
        elif stage == "validate":
            _, val_data_dicts = self._load_data_dicts()
            self.valset = CacheDataset(
                data=val_data_dicts, transform=self.val_transforms, cache_rate=1.0, num_workers=4
            )

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=1, num_workers=4)

    def test_dataloader(self):
        pass

    def calculate_class_weight(self):
        # ! <<< open debug yusongli
        # data_dicts = load_decathlon_datalist(
        #     os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
        # )
        # ! ===
        data_dicts, _ = self._load_data_dicts()
        # ! >>> clos debug

        class_weight = []
        for data_dict in data_dicts:
            label = LoadImage(reader="NibabelReader", image_only=True)(data_dict["label"])

            _, counts = np.unique(label, return_counts=True)
            counts = np.sum(counts) / counts
            # Normalize
            counts = counts / np.sum(counts)
            class_weight.append(counts)

        class_weight = np.asarray(class_weight)
        class_weight = np.mean(class_weight, axis=0)
        print("Class weight: ", class_weight)

    def calculate_class_percentage(self):
        # ! <<< open debug yusongli
        # data_dicts = load_decathlon_datalist(
        #     os.path.join(self.base_dir, "dataset.json"), data_list_key="training", base_dir=self.base_dir
        # )
        # ! ===
        data_dicts, _ = self._load_data_dicts()
        # ! >>> clos debug

        class_percentage = []
        for data_dict in data_dicts:
            label = LoadImage(reader="NibabelReader", image_only=True)(data_dict["label"])

            _, counts = np.unique(label, return_counts=True)
            # Normalize
            counts = counts / np.sum(counts)
            class_percentage.append(counts)

        class_percentage = np.asarray(class_percentage)
        class_percentage = np.mean(class_percentage, axis=0)
        print("Class Percentage: ", class_percentage)


if __name__ == "__main__":
    data_module = YuSongliDataModule(root_dir="/home/ubuntu/Task04_Hippocampus")
    # data_module.calculate_class_weight()
    data_module.calculate_class_percentage()
