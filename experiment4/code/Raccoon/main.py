import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from natsort import index_natsorted, order_by_index
import torch.nn.functional as F
import cv2

from RaccoonDataset import RaccoonDataset

EPOCH_NUM = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.01
IMAGE_RESIZE = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
                normalize])


class ClassificationHead(nn.Module):
    def __init__(self, resnet_model):
        super(ClassificationHead, self).__init__()
        self.resnet_model = torchvision.models.resnet18(pretrained=True)
        for param in self.resnet_model.parameters():
            param.requires_grad = False
        self.resnet_model.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.resnet_model(x))
        return x


class RegressionHead(nn.Module):
    def __init__(self, resnet_model):
        super(RegressionHead, self).__init__()
        self.resnet_model = resnet_model
        for param in self.resnet_model.parameters():
            param.requires_grad = False
        self.resnet_model.fc = nn.Linear(512, 256)
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.resnet_model(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Load data

annotations = pd.read_csv(r'raccoon/train/_annotations.txt', header=None, sep=" ", on_bad_lines='skip')

annotations = annotations.reindex(index=order_by_index(annotations.index,
                                                       index_natsorted(annotations[0], reverse=False)))
annotations.reset_index(drop=True, inplace=True)


train_dataset = RaccoonDataset(annotations, 'raccoon/train', IMAGE_RESIZE, transform=train_transform)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

annotations_test = pd.read_csv(r'raccoon/test/_annotations.txt', header=None, sep=" ", on_bad_lines='skip')

annotations_test = annotations_test.reindex(index=order_by_index(annotations_test.index,
                                                       index_natsorted(annotations_test[0], reverse=False)))
annotations_test.reset_index(drop=True, inplace=True)

test_dataset = RaccoonDataset(annotations_test, 'raccoon/test', IMAGE_RESIZE, transform=train_transform)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

annotations_test = pd.read_csv(r'raccoon/valid/_annotations.txt', header=None, sep=" ", on_bad_lines='skip')

annotations_test = annotations_test.reindex(index=order_by_index(annotations_test.index,
                                                       index_natsorted(annotations_test[0], reverse=False)))
annotations_test.reset_index(drop=True, inplace=True)

valid_dataset = RaccoonDataset(annotations_test, 'raccoon/valid', IMAGE_RESIZE, transform=train_transform)
valid_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

init_resnet_model = torchvision.models.resnet18(pretrained=True)
init_resnet_model2 = torchvision.models.resnet18(pretrained=True)
modelC = ClassificationHead(init_resnet_model).to(device)
modelR = RegressionHead(init_resnet_model).to(device)
modelR.to(device)
optimizerSM = torch.optim.SGD(modelC.parameters(), lr=LEARNING_RATE)
optimizerL2 = torch.optim.Adam(modelR.parameters(), lr=LEARNING_RATE)
criterionSM = nn.CrossEntropyLoss()
criterionL2 = nn.MSELoss()


def train():

    valid_loss_array = []
    train_loss_array = []
    for epoch in range(EPOCH_NUM):

        for i, (images, labels, object_class) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(torch.float32)
            labels = labels.to(device)

            object_class = object_class.to(device)

            outputsC = modelC(images)
            outputsR = modelR(images)

            lossSM = criterionSM(outputsC, object_class)
            lossL2 = criterionL2(outputsR, labels)

            optimizerSM.zero_grad()
            optimizerL2.zero_grad()

            lossSM.backward()
            lossL2.backward()

            optimizerSM.step()
            optimizerL2.step()

        print(f'Epoch [{epoch + 1}/{EPOCH_NUM}], Classification Train Loss: {lossSM.item():.4f}')
        print(f'Epoch [{epoch + 1}/{EPOCH_NUM}], Regression Train Loss: {lossL2.item():.4f}')

        train_loss_array.append(lossL2.item())
        valid_loss_array.append(validate(valid_dataloader))


    plt.plot(range(1, EPOCH_NUM + 1), train_loss_array)
    plt.xlim(0, EPOCH_NUM + 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"BATCH SIZE={BATCH_SIZE} LEARNING RATE={LEARNING_RATE} TRAIN LOSS")
    plt.savefig(f"batch={BATCH_SIZE}_lr={LEARNING_RATE}_train_loss.png")
    plt.cla()
    plt.clf()


    plt.plot(range(1, EPOCH_NUM + 1), valid_loss_array)
    plt.xlim(0, EPOCH_NUM + 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"BATCH SIZE={BATCH_SIZE} LEARNING RATE={LEARNING_RATE} VALID LOSS")
    plt.savefig(f"batch={BATCH_SIZE}_lr={LEARNING_RATE}_valid_loss.png")
    plt.cla()
    plt.clf()


def validate(data_loader):
    valid_loss_val = 0
    correct = 0

    modelR.eval()
    with torch.no_grad():
        for i, (images, labels, object_class) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(torch.float32)
            labels = labels.to(device)

            outputsR = modelR(images)
            outputsC = modelC(images)

            valid_loss_val += F.mse_loss(outputsR, labels)
    valid_loss_val /= len(test_dataloader)
    return valid_loss_val


def test(data_loader):
    test_loss_val = 0
    test_loss_class = 0
    correct = 0
    iou_vals = []
    modelR.eval()
    with torch.no_grad():
        for i, (images, labels, object_class) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(torch.float32)
            labels = labels.to(device)

            outputsR = modelR(images)
            outputsC = modelC(images)

            for (box1, box2) in zip(outputsR.tolist(), labels.tolist()):
                iou_vals.append(intersection_over_union(box1, box2))

            test_loss_val += F.mse_loss(outputsR, labels)
    test_loss_val /= len(test_dataloader)
    test_loss_class /= len(test_dataloader)
    print(f"Classification loss after testing = {test_loss_class}")
    print(f"Regression loss after testing = {test_loss_val}")
    print(f"Max IoU: {max(iou_vals)}")
    print(f"Min IoU: {min(iou_vals)}")
    print(f"Mean IoU: {np.mean(iou_vals)}")
    print(f"Median IoU: {np.median(iou_vals)}")


def intersection_over_union(area_one, area_two):
    xMin = max(area_one[0], area_two[0])
    xMax = min(area_one[2], area_two[2])
    yMin = max(area_one[1], area_two[1])
    yMax = min(area_one[3], area_two[3])

    area_val_one = (area_one[2] - area_one[0]) * (area_one[3] - area_one[1])
    area_val_two = (area_two[2] - area_two[0]) * (area_two[3] - area_two[1])

    intersection = (xMin - xMax) * (yMin - yMax)

    union = area_val_one + area_val_two - intersection

    return intersection / union


train()
test(test_dataloader)



