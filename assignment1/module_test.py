import numpy as np
import cv2

from dlvc.datasets.pets import PetsDataset
from dlvc.models.linear import LinearClassifier

from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
from dlvc.dataset import Subset
import dlvc.ops as ops

from dlvc.linear_cats_dogs import train_model

model = train_model(lr=0.001, momentum=0)

print(model.accuracy)