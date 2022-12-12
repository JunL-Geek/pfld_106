import numpy as np
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PFLDDatasets(Dataset):
    def __init__(self, file_list, transforms=None):
        super().__init__()
        self.image = None
        self.landmark = None
        self.euler_angle = None
        with open(file_list, 'r') as f:
            self.lines = f.readlines()
        self.transforms = transforms

    def __getitem__(self, index):
        line = self.lines[index].strip().split()
        self.image = cv2.imread(line[0])
        if self.transforms:
            self.image = self.transforms(self.image)
        self.landmark = np.asarray(line[1:213], dtype=np.float32)
        self.euler_angle = np.asarray(line[213:], dtype=np.float32)
        return self.image, self.landmark, self.euler_angle

    def __len__(self):
        return len(self.lines)


if __name__=='__main__':
    file_list = '..\\data\\train\\landmarks.txt'
    pfld_datasets = PFLDDatasets(file_list)
    pfld_dataloader = DataLoader(pfld_datasets,
                                 batch_size=256,
                                 shuffle=True,
                                 num_workers=0,
                                 drop_last=False)

    for image, landmark, euler_angle in pfld_dataloader:
        print(image.shape)
        print(landmark.size())
        print(euler_angle.size())
