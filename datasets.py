import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


class AutoDriveDataset(Dataset):
    def __init__(self, data_folder, transform=None, mode='train'):
        """
        :param data_folder: 
        :param transform: 
        :param mode: 'train'or'val'
        """
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform
        assert self.mode in {'train', 'val'}

        txt_path = os.path.join(data_folder, f'{mode}.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f" {mode} : {txt_path}")

        self.file_list = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        img_path = parts[0]
                        if not os.path.isabs(img_path):
                            img_path = os.path.join(data_folder, img_path)

                        angle = float(parts[1])
                        self.file_list.append((img_path, angle))

    def __getitem__(self, idx):
        img_path, angle = self.file_list[idx]

        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f": {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            if self.transform:
                img = self.transform(img)

            return img, torch.tensor([angle], dtype=torch.float32)

        except Exception as e:
            print(f" {img_path} : {str(e)}")
            dummy_img = Image.new('RGB', (160, 120))  
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, torch.tensor([0.0])

    def __len__(self):
        return len(self.file_list)
