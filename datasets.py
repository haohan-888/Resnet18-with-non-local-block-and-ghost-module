import os
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset


class AutoDriveDataset(Dataset):
    def __init__(self, data_folder, transform=None, mode='train'):
        """
        改进版本：自动处理路径问题
        :param data_folder: 数据根目录
        :param transform: 图像变换
        :param mode: 'train'或'val'
        """
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform
        assert self.mode in {'train', 'val'}

        # 构建正确的txt文件路径
        txt_path = os.path.join(data_folder, f'{mode}.txt')
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"无法找到 {mode} 数据文件: {txt_path}")

        self.file_list = []
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        # 处理相对/绝对路径
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
                raise ValueError(f"无法读取图像: {img_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            if self.transform:
                img = self.transform(img)

            return img, torch.tensor([angle], dtype=torch.float32)

        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {str(e)}")
            # 返回空白图像避免训练中断
            dummy_img = Image.new('RGB', (160, 120))  # 匹配您的resize尺寸
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, torch.tensor([0.0])

    def __len__(self):
        return len(self.file_list)