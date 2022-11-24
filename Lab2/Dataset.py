import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class People(Dataset):

    def __init__(self, file_path, root_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.root_path = root_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_path, self.data.loc[index, 'id'])
        image = Image.open(image_path)
        label = self.data.loc[index, 'target_people']

        if self.transform is not None:
            image = self.transform(image)

        return image, label
