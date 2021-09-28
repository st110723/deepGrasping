import torch
from torch.utils.data import Dataset
from natsort import natsorted
import os
from PIL import Image
import pandas as pd

class CustomDataSet(Dataset):

    def __init__(self, main_dir, labels_file, transform, flag):

        self.main_dir = main_dir
        self.transform = transform
        self.img_labels = pd.read_csv(labels_file)
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)
        self.flag = flag

    def __len__(self):

        return len(self.total_imgs)

    def __getitem__(self, idx): #TODO: barcha labels

        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        
        if self.flag == "training":
            label = self.img_labels[self.img_labels["img"] == self.total_imgs[idx]].iloc[0, 1:] #dima ye5ou awel rect
            label = torch.tensor(label.values.astype('float32'))
            return tensor_image, label

        if self.flag == "validation":
            label = self.img_labels[self.img_labels["img"] == self.total_imgs[idx]].iloc[:2, 1:]
            label = torch.tensor(label.values.astype('float32'))
            return tensor_image, label