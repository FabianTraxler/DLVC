import os
import time
from collections import namedtuple

import cv2
import torch
import numpy as np

Vec2 = namedtuple('Vec2', ['x1', 'x2'])


class AutogradFn(torch.autograd.Function):
    '''
    This class wraps a Fn instance to make it compatible with PyTorch optimimzers
    '''
    @staticmethod
    def forward(ctx, fn, loc):
        ctx.fn = fn
        ctx.save_for_backward(loc)
        value = fn(Vec2(loc[0].item(), loc[1].item()))
        return torch.tensor(value)

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        loc, = ctx.saved_tensors
        grad = fn.grad(Vec2(loc[0].item(), loc[1].item()))
        print("grad out is : ", grad_output)
        return None, torch.tensor([grad.x1, grad.x2]) * grad_output


class Fn:
    '''
    A 2D function evaluated on a grid.
    '''

    def __init__(self, fpath: str, eps: float):
        '''
        Ctor that loads the function from a PNG file.
        Raises FileNotFoundError if the file does not exist.
        '''

        if not os.path.isfile(fpath):
            raise FileNotFoundError()

        self.fn = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        self.fn = self.fn.astype(np.float32)
        self.fn /= (2**16 - 1)
        self.eps = eps

    def visualize(self, scale=True) -> np.ndarray:
        '''
        Return a visualization as a color image. Use e.g. cv2.applyColorMap.
        Use the result to visualize the progress of gradient descent.
        '''

        # TODO implement

        # for the colour map to be sued the image pixels must be in range of 0-255,
        # so simple scaling as follows should suffice, if the original image is between 0 and 1:

        if scale:
            image = np.uint8((self.fn + 1) * 255 / 2)
        else:
            image = self.fn

        coloured = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        #cv2.imshow("colour window", coloured)
        #k = cv2.waitKey(5000)

        return coloured

    def __call__(self, loc: Vec2) -> float:
        '''
        Evaluate the function at location loc.
        Raises ValueError if loc is out of bounds.
        '''

        # TODO implement
        # You can simply round and map to integers. If so, make sure not to set eps and learning_rate too low
        # For bonus points you can implement some form of interpolation (linear should be sufficient)

        fn = self.fn.data.tolist()  # make a copy as a list for easier use

        # get the coordinates
        x = loc.x1
        y = loc.x2

        # here we check if the location is valid.
        # this will also raise valueError if we are on  the bottom or right border of fn
        if x < 0 or y < 0 or y > np.array(self.fn.data.tolist()).shape[1] - 1 or x > np.array(self.fn.data.tolist()).shape[0] - 1:
            raise ValueError

        # next we get the 4 bounding grid coordinates:
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        x2 = int(x1 + 1)
        y2 = int(y1 + 1)

        # now we get the fn-values at the 4 grid points of fn:

        q11 = fn[x1][y1]
        q12 = fn[x1][y2]
        q21 = fn[x2][y1]
        q22 = fn[x2][y2]

        # and finnaly we can get the interpolation with matrix vector multiplication:
        x_vec = [x2 - x, x - x1]
        y_vec = [y2 - y, y - y1]
        fn_matrix = [[q11, q12], [q21, q22]]

        interpolation = 1 / ((x2 - x1) * (y2 - y1)) * \
            np.linalg.multi_dot([x_vec, fn_matrix, y_vec])

        # uncomment to see the interpolation result and the neighbouring values:
        #print("here is the interpolation: ", interpolation)
        #print("while the surrounding gridpoints are : \n", q11, q12, q21, q22)

        return interpolation

    def grad(self, loc: Vec2) -> Vec2:
        '''
        Compute the numerical gradient of the function at location loc, using the given epsilon.
        Raises ValueError if loc is out of bounds of fn or if eps <= 0.
        '''

        # TODO implement one of the two versions presented in the lecture

        # here we will calculate the gradients, for that we will add eps to x and y
        # calculate the gradients per dimesnions and return them as a vector / tensor

        x = loc.x1
        y = loc.x2

        if x < 0 or y < 0 or y > np.array(self.fn.data.tolist()).shape[1] - 1 or x > np.array(self.fn.data.tolist()).shape[0] - 1 or self.eps <= 0:
            raise ValueError

        x_eps = x + self.eps
        y_eps = y + self.eps

        f_loc = self.__call__(loc=loc)  # get the interpolation for fn in loc

        # first for x coordinate of gradients:
        # location shifted with eps in x
        loc_x = Vec2(x_eps, y)

        # get the interpolation value int he shifted location
        f_loc_x = self.__call__(loc=loc_x)
        # get x coordinate of the gradient
        grad_x = (f_loc_x - f_loc) / self.eps

        # for y coordinate of gradients:
        # location shifted with eps in y
        loc_y = Vec2(x, y_eps)
        # get the interpolation value int he shifted location
        f_loc_y = self.__call__(loc=loc_y)
        # get y coordinate of the gradient
        grad_y = (f_loc_y - f_loc) / self.eps

        print("gradients for x and y: ")
        print(grad_x, grad_y)
        print()
        grad = Vec2(10 * grad_y, 10 * grad_x)

        return grad


if __name__ == '__main__':

    # remove this comment block to enable the argument parser
    """

    # Parse args
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform gradient descent on a 2D function.')
    parser.add_argument(
        'fpath', help='Path to a PNG file encoding the function', default="fn/beale.png")
    parser.add_argument('sx1', type=float, default=1.0,
                        help='Initial value of the first argument')
    parser.add_argument('sx2', type=float, default=1.0,
                        help='Initial value of the second argument')
    parser.add_argument('--eps', type=float, default=1.0,
                        help='Epsilon for computing numeric gradients')
    parser.add_argument('--learning_rate', type=float,
                        default=10.0, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0,
                        help='Beta parameter of momentum (0 = no momentum)')
    parser.add_argument('--nesterov', action='store_true',
                        help='Use Nesterov momentum')




    args = parser.parse_args()

    # Init
    fn = Fn(args.fpath, args.eps)

    vis = fn.visualize()
    loc = torch.tensor([args.sx1, args.sx2], requires_grad=True)

    """

# Init
    sx1 = 50.5
    sx2 = 50.5
    lr = 3000
    fn = Fn("fn/madsen.png", 1)

    vis = fn.visualize()
    loc = torch.tensor([sx1, sx2], requires_grad=True)

    # fn.__call__(loc)
    # fn.grad(loc)
    optimizer = torch.optim.SGD(
        [loc], lr=lr, momentum=0, nesterov=False)

    # Perform gradient descent using a PyTorch optimizer
    # See https://pytorch.org/docs/stable/optim.html for how to use it
    i = 0
    while True:
        i += 1
        # Visualize each iteration by drawing on vis using e.g. cv2.line()
        # Find a suitable termination condition and break out of loop once done
        optimizer.zero_grad()
        start_point = (loc.data[0], loc.data[1])

        value = AutogradFn.apply(fn, loc)

        # this block could be helpful if we want to calculate the loss manually
        # by manually taking the steps based on our gradiant values and comparing
        # it to the loc values updated by the optimizer
        # calcualte the expected output:
        grad = fn.grad(Vec2(loc.data[0], loc.data[1]))
        grad_x = grad.x2
        grad_y = grad.x1

        try:
            # for the first iteration x_calculated and y_calculated do not exist
            x_calculated += lr * grad_x
            y_calculated += lr * grad_y
        except:
            print("x and y calc not present, using loc")
            x_calculated = loc.data[0] + lr * grad_x
            y_calculated = loc.data[1] + lr * grad_y

        # we create our target value
        target = torch.tensor([x_calculated, y_calculated], requires_grad=True)

        # define loss and calculate it:
        loss = torch.nn.L1Loss()
        output = loss(loc, target)

        # here if we want to use the loss, simply change it to output.backward()
        # output.backward()
        value.backward()

        # some minor logs for the locations losses etc
        print("locations and should be locs are:", loc.data, target.data)
        print("loss is :", value.data)

        # fineally we do one iteration
        optimizer.step()

        # this below block is responsible for the plotting
        end_point = (loc.data[0], loc.data[1])
        color = (255, 0, 0)
        thickness = 3

        image = fn.visualize()
        image = cv2.line(image, start_point, end_point, color, thickness)
        cv2.imshow('Progress', image)
        cv2.waitKey(50)  # 20 fps, tune according to your liking

        # break conditions
        if (i == 100 or (np.abs(grad_x) < 0.0061 and np.abs(grad_y) < 0.0061)):
            print("we reached the iteration limit or stopping criterion")
            time.sleep(10)
            break
