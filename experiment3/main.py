import numpy as np
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch import nn
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 8
LEARNING_RATES = [0.1, 0.01, 0.001]
EPOCH_NUM = 3


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.conv4 = nn.Conv2d(128, 256, 5, 1)
        self.conv5 = nn.Conv2d(256, 512, 5, 1)
        self.fc1 = nn.Linear(512 * 5 * 5, 128)
        self.output = nn.Linear(128, 15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        x = F.log_softmax(x, dim=1)
        return x


dataset_path = 'pa3_dataset'

transform_image = transforms.Compose([
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform_image)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform_image)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

loss_dict = {}
acc_dict = {}

classes = ('Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal',
            'Broccoli', 'Cabbage', 'Capsicum', 'Carrot',
            'Cauliflower', 'Cucumber', 'Papaya', 'Potato',
            'Pumpkin', 'Radish', 'Tomato')

for lr in LEARNING_RATES:
    print("Learning rate = ", lr)
    model = CNN().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_total_steps = len(train_loader)
    for epoch in range(EPOCH_NUM):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % int(n_total_steps/4) == 0:
                print(f'Epoch [{epoch + 1}/{EPOCH_NUM}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    loss_dict[str(lr)] = loss.item()
    print('Finished Training')
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(15)]
        n_class_samples = [0 for i in range(15)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(BATCH_SIZE):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        acc_dict[str(lr)] = acc

loss_list = loss_dict.items()
x, y = zip(*loss_list)
plt.plot(x, y)
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title(f'Batch Size: {BATCH_SIZE}')
plt.show()


acc_list = acc_dict.items()
x, y = zip(*acc_list)
plt.plot(x, y)
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 105, 5))
plt.title(f'Batch Size: {BATCH_SIZE}')
plt.show()
