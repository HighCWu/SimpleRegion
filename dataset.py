import glob
import numpy as np

import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.imgs = glob.glob(root + '/**/*.jpg', recursive=True) + \
                    glob.glob(root + '/**/*.png', recursive=True) + \
                    glob.glob(root + '/**/*.bmp', recursive=True)
        self.imgs = sorted(self.imgs)
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path).convert('RGB')

        return self.transform(img)
