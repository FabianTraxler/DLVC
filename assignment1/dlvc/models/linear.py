from ..model import Model

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

class LinearClassifier(Model):
    '''
    Linear classifier without bias.
    Returns softmax class scores (see lecture slides).
    '''

    def __init__(self, input_dim: int, num_classes: int, lr: float, momentum: float, nesterov: bool):
        '''
        Ctor.
        input_dim is the length of input vectors (> 0).
        num_classes is the number of classes (> 1).
        lr: learning rate to use for training (> 0).
        momentum: momentum to use for training (> 0).
        nesterov: training with or without Nesterov momentum.
        '''

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

        self.criterion = nn.CrossEntropyLoss()
        self.weights = torch.rand((num_classes, input_dim), dtype=torch.double, requires_grad=True)
        self.last_update = 0

    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple, which is (0, input_dim).
        '''

        return (0, self.input_dim)

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self.num_classes, )

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data are the input data, with shape (m, input_dim) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the current cross-entropy loss on the batch.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        outputs = self.__predict__(data)

        labels = torch.tensor(labels)

        loss = self.criterion(outputs, labels)

        self.weights.retain_grad() # include this tensor in the computation graph
        loss.backward() # compute gradients with backpropagation
        
        if not self.nestrov:
            # update the weights
            update = self.lr * self.weights.grad
            self.weights = self.weights - self.momentum * self.last_update - update
            self.last_update = update
        else:
            self.eta = 1 #it can be changed or dynaimcally adjusted if wanted
            #calculate velocity:
            if not self.v:
                self.v = -self.eta * self.weights.grad
            else: 
                self.v = self.lr * self.v - self.eta * self.weights.grad
            
            #update weights by the learning rule
            self.weights = self.weights + self.v
            self.update = self.v
        return float(loss)

    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        data = torch.tensor(data)
        output = torch.matmul(data, self.weights.T)
        output = nn.Softmax(-1)(output)
        return output.detach().numpy()
    
    def __predict__(self, data: np.ndarray) -> torch.tensor:
        '''
        Predict softmax class scores from input data and return as torch tensor to retain the Gradients.
        Data are the input data, with a shape compatible with input_shape().
        The label array has shape (n, output_shape()) with n being the number of input samples.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        data = torch.tensor(data)
        output = torch.matmul(data, self.weights.T)
        output = nn.Softmax(-1)(output)

        return output
