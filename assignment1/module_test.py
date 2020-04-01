import numpy as np
import cv2

from dlvc.datasets.pets import PetsDataset
from dlvc.batches import BatchGenerator
import dlvc.ops as ops

pets_train = PetsDataset("cifar-10-batches-py/", 1)

print('Number of Classes = {}'.format(pets_train.num_classes()))
print('Number of Images = {}'.format(pets_train.__len__()))
print('First 10 Classes >>> {}'.format(pets_train.labels[:10]))



op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

print(op(pets_train.__getitem__(0).data).shape)

batches = BatchGenerator(pets_train, 500, False, op)

print('Number of Batches = {}'.format(batches.__len__()))

for batch in batches:
     print('Shape of the data batch: {}'.format(batch.data.shape))
     print('Shape of the label batch: {}'.format(batch.label.shape))
     print('First 5 Elements of the first element: {}'.format(batch.data[0][0:5]))
     print('Label of the first element: {}'.format(batch.label[0]))
     break



# show the first image
# item = pets_train.__getitem__(0)
# print(item.data.shape)
# cv2.imshow(str(item.label), item.data)
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image

