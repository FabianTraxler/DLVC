import numpy as np
import cv2

from dlvc.datasets.pets import PetsDataset
from dlvc.models.linear import LinearClassifier

from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
from dlvc.dataset import Subset
import dlvc.ops as ops

np.random.seed(0)

pets_train = PetsDataset("../cifar-10-batches-py/", Subset.TRAINING)
pets_val = PetsDataset("../cifar-10-batches-py/", Subset.VALIDATION)

random_accuracy = Accuracy()
validation_accuracy = Accuracy()
train_accuracy = Accuracy()


print('Number of Classes = {}'.format(pets_train.num_classes()))
print('Number of Images = {}'.format(pets_train.__len__()))
print('First 10 Classes >>> {}'.format(pets_train.labels[:10]))




op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

train_batches = BatchGenerator(pets_train, 100, False, op)
validation_batches = BatchGenerator(pets_val, 100, False, op)

print('Number of Batches = {}'.format(train_batches.__len__()))

model  = LinearClassifier(3072, pets_train.num_classes(), lr=0.001, momentum=0.1, nesterov=True)

# test batch generator
for batch in train_batches:
    print('Shape of the data batch: {}'.format(batch.data.shape))
    print('Shape of the label batch: {}'.format(batch.label.shape))
    print('First 5 Elements of the first element: {}'.format(batch.data[0][0:5]))
    print('Label of the first element: {}'.format(batch.label[0]))
    model.train(batch.data, batch.label)
    break


# train model
for batch in train_batches:
    loss = model.train(batch.data, batch.label)
    print('Loss >>> %.2f' % loss, end='\r')



# validate model and random 
for batch in validation_batches:
    model_prediction = model.predict(batch.data)
    validation_accuracy.update(model_prediction, batch.label)

    random_prediction = np.random.random(model_prediction.shape)
    random_accuracy.update(random_prediction, batch.label)



print('Random Accuracy >>> {}'.format(random_accuracy.accuracy()))
print('Model Accuracy >>> {}'.format(validation_accuracy.accuracy()))


# show the first image
#item = pets_train.__getitem__(1)
#cv2.imshow('Test Image', item.data)
#cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows() # destroys the window showing image
