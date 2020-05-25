import numpy as np
import torch
import torch.nn as nn

from ..model import Model

class CnnClassifier(Model):
    '''
    Wrapper around a PyTorch CNN for classification.
    The network must expect inputs of shape NCHW with N being a variable batch size,
    C being the number of (image) channels, H being the (image) height, and W being the (image) width.
    The network must end with a linear layer with num_classes units (no softmax).
    The cross-entropy loss (torch.nn.CrossEntropyLoss) and SGD (torch.optim.SGD) are used for training.
    '''

    def __init__(self, net: nn.Module, input_shape: tuple, num_classes: int, lr: float, wd: float):
        '''
        Ctor.
        net is the cnn to wrap. see above comments for requirements.
        input_shape is the expected input shape, i.e. (0,C,H,W).
        num_classes is the number of classes (> 0).
        lr: learning rate to use for training (SGD with e.g. Nesterov momentum of 0.9).
        wd: weight decay to use for training.
        '''

        self.net = net

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.cuda = next(net.parameters()).is_cuda

        self.optimizer = torch.optim.SGD(
            net.parameters(), 
            lr=lr, 
            weight_decay=wd, 
            nesterov=True, 
            momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()


    def input_shape(self) -> tuple:
        '''
        Returns the expected input shape as a tuple.
        '''

        return self.input_shape

    def output_shape(self) -> tuple:
        '''
        Returns the shape of predictions for a single sample as a tuple, which is (num_classes,).
        '''

        return (self.num_classes, )

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        '''
        Train the model on batch of data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        Labels has shape (m,) and integral values between 0 and num_classes - 1.
        Returns the training loss.
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError("The given data is not ndarray")
        if not isinstance(labels, np.ndarray):
            raise TypeError("The given label is not ndarray")
        if data.shape[0] != labels.shape[0]:
            raise ValueError("The length of input data and label does not match") 
        try:
            self.net.train()
            self.optimizer.zero_grad()

            # Initialize the inputs and transfer them to gpu if necessary
            if self.cuda:
                inputs = torch.from_numpy(data).cuda()
                labels = torch.from_numpy(labels).long().cuda()
            else:
                inputs = torch.from_numpy(data).cpu()
                labels = torch.from_numpy(labels).long().cpu()

            outputs = self.net(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            return float(loss.detach())
        except:
            raise RuntimeError("Something went wrong in training")


    def predict(self, data: np.ndarray) -> np.ndarray:
        '''
        Predict softmax class scores from input data.
        Data has shape (m,C,H,W) and type np.float32 (m is arbitrary).
        The scores are an array with shape (n, output_shape()).
        Raises TypeError on invalid argument types.
        Raises ValueError on invalid argument values.
        Raises RuntimeError on other errors.
        '''

        if not isinstance(data, np.ndarray):
            raise TypeError("The given data is not ndarray")
        try:
            self.net.eval()
            if self.cuda:
                inputs = torch.from_numpy(data).cuda()
            else:
                inputs = torch.from_numpy(data).cpu()

            outputs = self.net(inputs)

            outputs = nn.Softmax(dim=1)(outputs)

            return outputs.cpu().detach().numpy()
        except:
            raise RuntimeError("Something went wrong in training")
