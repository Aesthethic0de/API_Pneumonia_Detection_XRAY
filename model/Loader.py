from torch.utils.data import Dataset
from PIL import Image
import torch




class XrayDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image