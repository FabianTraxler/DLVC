
from .model import Model
from .batches import BatchGenerator

import numpy as np

from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass

    @abstractmethod
    def __lt__(self, other) -> bool:
        '''
        Return true if this performance measure is worse than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass

    @abstractmethod
    def __gt__(self, other) -> bool:
        '''
        Return true if this performance measure is better than another performance measure of the same type.
        Raises TypeError if the types of both measures differ.
        '''

        pass


class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self):
        '''
        Ctor.
        '''

        self.reset()

    def reset(self):
        '''
        Resets the internal state.
        '''

        self.correct_predictions = 0
        self.total_predictions = 0

    def update(self, prediction: np.ndarray, target: np.ndarray):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        def get_prediction(row):             
            return np.argmax(row)

        prediction = np.apply_along_axis(get_prediction, 1, prediction)

        correct_predictions = (prediction == target)
        self.correct_predictions += sum(correct_predictions)
        self.total_predictions += len(prediction)

    def __str__(self):
        '''
        Return a string representation of the performance.
        '''

        accuracy = self.accuracy()

        return "accuracy: {}".format(accuracy)

    def __lt__(self, other) -> bool:
        '''
        Return true if this accuracy is worse than another one.
        Raises TypeError if the types of both measures differ.
        '''

        try:
            return self.accuracy < other.accuracy 
        except:
            raise TypeError


        pass

    def __gt__(self, other) -> bool:
        '''
        Return true if this accuracy is better than another one.
        Raises TypeError if the types of both measures differ.
        '''
        try:
            return self.accuracy() > other.accuracy()
        except:
            raise TypeError

    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        if self.total_predictions != 0:
            accuracy = self.correct_predictions / self.total_predictions
        else:
            accuracy = 0

        return accuracy