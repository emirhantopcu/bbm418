import os
import pandas as pd

import cv2
import torch as torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np


from torch.utils.data import Dataset


class VotDataset(Dataset):
    def __init__(self, images, ground_truth):
        super(VotDataset, self).__init__()
        self.images = images
        self.annotations = ground_truth

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        return self.images[item], self.annotations[item]


def box_finder(outputs, image):
    height, width, cr = image.shape
    bounding_box_list = []
    confidentiality_list = []
    coordinates = []

    for output in outputs:
        for elem in output:
            scores = elem[5:]
            predicted_class = np.argmax(scores)
            if predicted_class == 0:
                confidence = scores[predicted_class]
                if confidence > 0.4:
                    w, h = int(elem[2] * width), int(elem[3] * height)
                    x0, y0 = int(elem[0] * width), int(elem[1] * height)
                    x1, y1 = x0 - (w / 2), y0 - (h / 2)
                    x2, y2 = x0 + (w / 2), y0 - (h / 2)
                    x3, y3 = x0 + (w / 2), y0 + (h / 2)
                    x4, y4 = x0 - (w / 2), y0 + (h / 2)
                    confidentiality_list.append(confidence)
                    bounding_box_list.append([x1, y1, x3, y3])
                    coordinates.append([x1, y1, x2, y2, x3, y3, x4, y4])

    return coordinates, bounding_box_list


if __name__ == '__main__':
    name = "/soccer1"

    path = "vot2017"
    frames = []
    ground_truths = []
    ground_truth = pd.read_csv(path + name + "/groundtruth.txt", header=None, sep=",", on_bad_lines="skip")
    images_file = os.listdir(path + name + "/color")

    for file in range(len(images_file)):
        img = cv2.imread(path + name + "/color/" + images_file[file])
        frames.append(img)
        list_ground_truth = list(ground_truth.iloc[file])
        list_ground_truth = [float(gt) for gt in list_ground_truth]
        list_ground_truth = torch.tensor(list_ground_truth)
        ground_truths.append(list_ground_truth)

    # Load data

    votddataset = VotDataset(frames, ground_truths)
    votd_loader = DataLoader(votddataset, batch_size=1, pin_memory=True, num_workers=1)

    yolov3 = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")

    loss_function = nn.MSELoss()

    for i, (frame, gt) in enumerate(votd_loader):
        blob_frame = torch.Tensor.numpy(frame)
        blob_frame = blob_frame[0]
        gt = gt[0]

        blob = cv2.dnn.blobFromImage(blob_frame, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)

        yolov3.setInput(blob)
        layers = yolov3.getLayerNames()
        outputs_names = [layers[j - 1] for j in yolov3.getUnconnectedOutLayers()]

        outputs = yolov3.forward(outputs_names)
        coords, bbox = box_finder(outputs, blob_frame)
        if len(coords) != 0:
            loss_list = []
            for coord in coords:
                loss = loss_function(torch.Tensor(coord), gt)
                loss_list.append(loss.item())
            bounding_box = bbox[np.argmin(loss_list)]

            new_frame = cv2.rectangle(blob_frame, (int(bounding_box[0]), int(bounding_box[1])),
                                      (int(bounding_box[2]), int(bounding_box[3])),
                                      color=(0, 0, 255), thickness=2)

            cv2.imwrite(f"gif/image{i}.jpg", new_frame)

            print("Humanoid found in this frame..")

        else:
            cv2.imwrite(f"gif/image{i}.jpg", blob_frame)

            print("Humanoid not found in this frame.")

