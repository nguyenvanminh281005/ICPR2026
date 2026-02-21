
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

import torch

class SRDataset(Dataset):
    def __init__(self, root_dir, hr_size=(32, 64), scale=2):
        """
        hr_size: kích thước HR chuẩn (H, W)
        scale: hệ số SR (2, 4,...)
        """
        self.hr_paths = []
        self.hr_size = hr_size
        self.scale = scale

        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.startswith("hr-"):
                    self.hr_paths.append(os.path.join(root, f))

        self.to_tensor = T.ToTensor()

        print(f"Found {len(self.hr_paths)} HR images.")

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]

        hr = Image.open(hr_path).convert("RGB")

        hr = resize(
            hr,
            self.hr_size,
            interpolation=InterpolationMode.BICUBIC
        )

        lr_size = (self.hr_size[0] // self.scale,
                   self.hr_size[1] // self.scale)

        lr = resize(
            hr,
            lr_size,
            interpolation=InterpolationMode.BICUBIC
        )

        hr = self.to_tensor(hr)
        lr = self.to_tensor(lr)

        return lr, hr
