import numpy as np
import cv2

import torch
import torch.nn as nn


from dlvc.datasets.pets import PetsDataset
from dlvc.models.pytorch import CnnClassifier

from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
from dlvc.dataset import Subset
import dlvc.ops as ops


np.random.seed(0)

pets_train = PetsDataset("../assignment1/cifar-10-batches-py/", Subset.TRAINING)

print(pets_train.__getitem__(0).data.shape)

op = ops.chain([
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
    ops.hflip(),
    ops.rcrop(32, 1, "constant"),
    ops.hwc2chw()
])

train_batches = BatchGenerator(pets_train, 100, False, op)


class Net(nn.Module):
    """
    """
    def __init__(self, num_classes):
        super(Net, self).__init__()

        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()

        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)

        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Instantiate a fully connected layer
        self.fc = nn.Linear(int(32 / 2 / 2 * 32 / 2 / 2 * 10), num_classes)

    def forward(self, x):
        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Prepare the image for the fully connected layer
        x = x.view(-1, int(10 * 32 / 2 /2 * 32 / 2 / 2))

        # Apply the fully connected layer and return the result
        return self.fc(x)

net = Net(2)
classifier = CnnClassifier(net, (0, 32, 32, 3), 2, 0.01, 0.01)


for batch in train_batches:
    loss = classifier.train(batch.data, batch.label)
    print(loss, end="\r")

print()


# show the first image
# item = pets_train.__getitem__(1)
# cv2.imshow('Test Image', item.data)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image

# cv2.imshow('Test Image', image)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image


