import numpy as np
import cv2

from dlvc.datasets.pets import PetsDataset
from dlvc.models.linear import LinearClassifier

from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
from dlvc.dataset import Subset
import dlvc.ops as ops

np.random.seed(0)

pets_train = PetsDataset("../assignment1/cifar-10-batches-py/", Subset.TRAINING)

op = ops.chain([
    ops.hflip(),
    ops.rcrop(25, 1, "constant")
])

train_batches = BatchGenerator(pets_train, 100, False, op)

for batch in train_batches:
    image = batch.data[1]
    print('Shape of the data batch: {}'.format(batch.data.shape))
    break


# show the first image
item = pets_train.__getitem__(1)
cv2.imshow('Test Image', item.data)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image

cv2.imshow('Test Image', image)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
