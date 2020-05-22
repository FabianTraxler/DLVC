from collections import namedtuple
import numpy as np
import pandas as pd

import time

from dlvc.models.linear import LinearClassifier
from dlvc.batches import BatchGenerator
from dlvc.test import Accuracy
from dlvc.datasets.pets import PetsDataset
from dlvc.dataset import Subset
import dlvc.ops as ops

TrainedModel = namedtuple('TrainedModel', ['model', 'accuracy'])


train_data = PetsDataset("../cifar-10-batches-py/", Subset.TRAINING)
val_data = PetsDataset("../cifar-10-batches-py/", Subset.VALIDATION)
test_data = PetsDataset('../cifar-10-batches-py/', Subset.TEST)


op = ops.chain([
    ops.vectorize(),
    ops.type_cast(np.float32),
    ops.add(-127.5),
    ops.mul(1/127.5),
])

train_batches = BatchGenerator(train_data, 50, False, op)
val_batches = BatchGenerator(val_data, 50, False, op)
test_batches = BatchGenerator(test_data, 50, False, op)

def train_model(lr: float, momentum: float) -> TrainedModel:
    '''
    Trains a linear classifier with a given learning rate (lr) and momentum.
    Computes the accuracy on the validation set.
    Returns both the trained classifier and accuracy.
    '''

    clf = LinearClassifier(input_dim=3072,  num_classes=train_data.num_classes(), lr=lr, momentum=momentum, nesterov=False)

    n_epochs = 10
    for i in range(n_epochs):
        for batch in train_batches:            
            clf.train(batch.data,batch.label)

    accuracy = Accuracy()
    for batch in val_batches:
        prediction = clf.predict(batch.data)
        accuracy.update(prediction, batch.label)


    return TrainedModel(clf, accuracy)


def grid_search(lr_options, momentum_options, nesterov):
    """Make a Grid Search with the given options
    
    Return the best model(TrainedModel) and a Pandas DataFrae with all validation accuracies
    """

    models = []
    best_model = TrainedModel(None, Accuracy())

    # create a dataframe with all validation accuracies 
    result_df = pd.DataFrame(index=lr_options, columns=momentum_options)
    result_df.index.name = 'Learning Rate'
    result_df.columns.name = 'Momentum'

    # loop over all combinations
    for lr in lr_options:
        for momentum in momentum_options:
            nesterov = True
            print('Training with parameters: lr={}, momentum={}, nesterov={}  '.format(lr, momentum, nesterov), end='\r')
            model = train_model(lr, momentum)
            models.append(model)
            result_df.at[model.model.lr, model.model.momentum] = model.accuracy.accuracy()

            if model.accuracy.__gt__(best_model.accuracy):
                best_model = model

    return (best_model, result_df)


lr_options = [0.5, 0.2, 0.1, 0.01, 0.001]
momentum_options = [0.9, 0.5, 0.1, 0.01, 0.001]

start_time = time.time()

best_model, result_df = grid_search(lr_options=lr_options, momentum_options=momentum_options, nesterov=True)

end_time = time.time()

print('Wall time: {}s'.format(round(end_time - start_time, 2)))

print(result_df)

print('\nBest Model:')
print('Parameters: lr={}, momentum={}'.format(best_model.model.lr, best_model.model.momentum))
test_accuracy = Accuracy()
for batch in test_batches:
    prediction = best_model.model.predict(batch.data)
    test_accuracy.update(prediction, batch.label)
print('Test Accuracy = {}'.format(test_accuracy.accuracy()))