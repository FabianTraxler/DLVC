from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn

import time

from dlvc.models.pytorch import CnnClassifier
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
import dlvc.ops as ops

np.random.seed(0)
torch.manual_seed(0)

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])

DATA_PATH = "../cifar-10-batches-py/"
train_data = PetsDataset(DATA_PATH, Subset.TRAINING)
val_data = PetsDataset(DATA_PATH, Subset.VALIDATION)
test_data = PetsDataset(DATA_PATH, Subset.TEST)


op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1 / 127.5),
    ops.hwc2chw()
])

train_batches = BatchGenerator(train_data, 128, False, op)
val_batches = BatchGenerator(val_data, 128, False, op)
test_batches = BatchGenerator(test_data, 128, False, op)


class Net(nn.Module):
    def __init__(self, img_size, num_classes):
        super(Net, self).__init__()
        self.img_size = img_size

        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()

        # Instantiate two convolutional blocks
        ## Block 1
        self.conv1_1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, padding=1)
        ## Block 2
        self.conv2_1 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Instantiate a fully connected layer
        self.fc = nn.Linear(
            int(img_size[0] / 2 / 2 * img_size[1] / 2 / 2 * 64), num_classes)

    def forward(self, x):
        # Apply block 1 followed by max pool
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        # Apply block 2followed by max pool
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)
        # Prepare the image for the fully connected layer
        x = x.view(-1, int(64 * 32 / 2 / 2 * 32 / 2 / 2))

        # Apply the fully connected layer and return the result
        return self.fc(x)


img_shape = train_data.image_shape()
num_classes = train_data.num_classes()

net = Net(img_shape, num_classes)

clf = CnnClassifier(net, (0, *img_shape), num_classes, 0.01, 0)

for epoch in range(100):
    losses = []
    for batch in train_batches:
        loss = clf.train(batch.data, batch.label)
        losses.append(loss)
    losses = np.array(losses)
    mean = round(np.mean(losses), 3)
    std = round(np.std(losses), 3)
    print("epoch {}".format(epoch))
    print("  train loss: {} +- {}".format(mean, std))
    accuracy = Accuracy()
    for batch in val_batches:
        predictions = clf.predict(batch.data)
        accuracy.update(predictions, batch.label)
    acc = round(accuracy.accuracy(), 3)
    print("  val acc: accuracy: {}".format(acc))
