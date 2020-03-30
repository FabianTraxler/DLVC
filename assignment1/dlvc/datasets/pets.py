
from ..dataset import Sample, Subset, ClassificationDataset
import pandas as pd
import numpy as np

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    def __init__(self, fdir: str, subset: Subset):
        '''
        Loads a subset of the dataset from a directory fdir that contains the Python version
        of the CIFAR-10, i.e. files "data_batch_1", "test_batch" and so on.
        Raises ValueError if fdir is not a directory or if a file inside it is missing.

        The subsets are defined as follows:
          - The training set contains all cat and dog images from "data_batch_1" to "data_batch_4", in this order.
          - The validation set contains all cat and dog images from "data_batch_5".
          - The test set contains all cat and dog images from "test_batch".

        Images are loaded in the order the appear in the data files
        and returned as uint8 numpy arrays with shape 32*32*3, in BGR channel order.
        '''

        # TODO implement
        # See the CIFAR-10 website on how to load the data files
        
        if subset == 1:
            #training set:

            fdir1 = fdir + "data_batch_1"
            self.data1 = self.unpickle(fdir1)
            if b'batch_label' in self.data1: del self.data1[b'batch_label']
            self.data1 = pd.DataFrame.from_dict(self.data1, orient='index').T
            self.data1 = self.data1[self.data1[b'labels'].isin([3,5])]


            fdir2 = fdir + "data_batch_2"
            self.data2 = self.unpickle(fdir2)
            if b'batch_label' in self.data2: del self.data2[b'batch_label']
            self.data2 = pd.DataFrame.from_dict(self.data2, orient='index').T
            self.data2 = self.data2[self.data2[b'labels'].isin([3,5])]

            fdir3 = fdir + "data_batch_3"
            self.data3 = self.unpickle(fdir3)
            if b'batch_label' in self.data3: del self.data3[b'batch_label']
            self.data3 = pd.DataFrame.from_dict(self.data3, orient='index').T
            self.data3 = self.data3[self.data3[b'labels'].isin([3,5])]

            fdir4 = fdir + "data_batch_4"
            self.data4 = self.unpickle(fdir4)
            if b'batch_label' in self.data4: del self.data4[b'batch_label']                        
            self.data4 = pd.DataFrame.from_dict(self.data4, orient='index').T
            self.data4 = self.data4[self.data4[b'labels'].isin([3,5])]

            #creating whole dataframe:
            self.data = self.data1.append(self.data2).append(self.data3).append(self.data4)

            #reshaping image arrayto 32*32*3
            self.data[b'data'] = self.data[b'data'].apply(lambda x: np.array(x).reshape((32,32,3)).astype(np.uint8))

            #changinglabels to 0 and 1
            self.data[b'labels'].replace({3: 0, 5: 1}, inplace=True)

        elif subset == 2:
            #validation set
            fdir5 = fdir + "data_batch_5"
            self.data5 = self.unpickle(fdir5)
            if b'batch_label' in self.data5: del self.data5[b'batch_label']                        
            self.data = pd.DataFrame.from_dict(self.data5, orient='index').T

            self.data = self.data[self.data[b'labels'].isin([3,5])]
            self.data[b'data'] = self.data[b'data'].apply(lambda x: np.array(x).reshape((32,32,3)).astype(np.uint8))
            self.data[b'labels'].replace({3: 0, 5: 1}, inplace=True)

        elif subset == 3:
            #test set
            fdir_test = fdir + "test_batch"
            self.data_test = self.unpickle(fdir_test)
            if b'batch_label' in self.data_test: del self.data_test[b'batch_label']                        
            self.data = pd.DataFrame.from_dict(self.data_test, orient='index').T

            self.data = self.data[self.data[b'labels'].isin([3,5])]
            self.data[b'data'] = self.data[b'data'].apply(lambda x: np.array(x).reshape((32,32,3)).astype(np.uint8))
            self.data[b'labels'].replace({3: 0, 5: 1}, inplace=True)
        pass

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        # TODO implement

        return(self.data.size)
        

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        # TODO implement
        try:
            return(self.data.iloc[idx])
        except IndexError as e:
            print("bad index")
            raise e
        

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        # TODO implement

        return(self.data[b'labels'].nunique())
        

obj = PetsDataset("cifar-10-batches-py/", 1)

import cv2 

cv2.imshow("example",obj.data[b'data'].values[1])
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
