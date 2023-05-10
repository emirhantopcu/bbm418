import torch
from torch.utils.data import Dataset
import cv2


class RaccoonDataset(Dataset):
    def __init__(self, annotations_df, root_dir, image_resize, transform=None):
        self.annotations = annotations_df
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = self.annotations[:][0]
        self.labels = self.annotations[:][1]
        self.image_resize = image_resize

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        path = self.root_dir + "/" + self.image_names.iloc[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shape_y = image.shape[0]
        shape_x = image.shape[1]

        image = self.transform(image)
        targets = self.labels[index].split(",")[:-1]
        object_class = int(self.labels[index].split(",")[-1])
        targets = [int(target) for target in targets]
        targets[0] = int(targets[0] * (self.image_resize/shape_x))
        targets[1] = int(targets[1] * (self.image_resize/shape_y))
        targets[2] = int(targets[2] * (self.image_resize/shape_x))
        targets[3] = int(targets[3] * (self.image_resize/shape_y))
        targets = torch.tensor(targets)
        object_class = torch.tensor(object_class)
        sample = (image, targets, object_class)

        return sample

