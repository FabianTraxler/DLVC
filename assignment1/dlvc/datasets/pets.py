

from ..dataset import Sample, Subset, ClassificationDataset
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
            self.data1[b"data"] = np.swapaxes(self.data1[b"data"].reshape((len(self.data1[b"data"]), 3, 32, 32)), 1, 3)

            idx_to_remove = np.where((np.array(self.data1[b"labels"]) != 3) & (np.array(self.data1[b"labels"]) != 5))[0]
             
            self.data1[b"data"]=np.delete(self.data1[b"data"], idx_to_remove, axis=0)
            self.data1[b"labels"]=np.delete(self.data1[b"labels"], idx_to_remove, axis=0)
            self.data1[b"filenames"]=np.delete(self.data1[b"filenames"], idx_to_remove, axis=0)


            fdir2 = fdir + "data_batch_2"
            self.data2 = self.unpickle(fdir2)
            if b'batch_label' in self.data2: del self.data2[b'batch_label']
            self.data2[b"data"] = np.swapaxes(self.data2[b"data"].reshape((len(self.data2[b"data"]), 3, 32, 32)), 1, 3)

            idx_to_remove = np.where((np.array(self.data2[b"labels"]) != 3) & (np.array(self.data2[b"labels"]) != 5))[0]
             
            self.data2[b"data"]=np.delete(self.data2[b"data"], idx_to_remove, axis=0)
            self.data2[b"labels"]=np.delete(self.data2[b"labels"], idx_to_remove, axis=0)
            self.data2[b"filenames"]=np.delete(self.data2[b"filenames"], idx_to_remove, axis=0)


            fdir3 = fdir + "data_batch_3"
            self.data3 = self.unpickle(fdir3)
            if b'batch_label' in self.data3: del self.data3[b'batch_label']
            self.data3[b"data"] = np.swapaxes(self.data3[b"data"].reshape((len(self.data3[b"data"]), 3, 32, 32)), 1, 3)

            idx_to_remove = np.where((np.array(self.data3[b"labels"]) != 3) & (np.array(self.data3[b"labels"]) != 5))[0]
             
            self.data3[b"data"]=np.delete(self.data3[b"data"], idx_to_remove, axis=0)
            self.data3[b"labels"]=np.delete(self.data3[b"labels"], idx_to_remove, axis=0)
            self.data3[b"filenames"]=np.delete(self.data3[b"filenames"], idx_to_remove, axis=0)


            fdir4 = fdir + "data_batch_4"
            self.data4 = self.unpickle(fdir4)
            if b'batch_label' in self.data4: del self.data4[b'batch_label']                        
            self.data4[b"data"] = np.swapaxes(self.data4[b"data"].reshape((len(self.data4[b"data"]), 3, 32, 32)), 1, 3)

            idx_to_remove = np.where((np.array(self.data4[b"labels"]) != 3) & (np.array(self.data4[b"labels"]) != 5))[0]
             
            self.data4[b"data"]=np.delete(self.data4[b"data"], idx_to_remove, axis=0)
            self.data4[b"labels"]=np.delete(self.data4[b"labels"], idx_to_remove, axis=0)
            self.data4[b"filenames"]=np.delete(self.data4[b"filenames"], idx_to_remove, axis=0)
            


            #creating whole dataframe:
            self.data = {}
            self.data[b"data"] = np.concatenate((self.data1[b"data"],self.data2[b"data"],self.data3[b"data"],self.data4[b"data"]))
            self.data[b"labels"] = np.concatenate((self.data1[b"labels"],self.data2[b"labels"],self.data3[b"labels"],self.data4[b"labels"]))
            self.data[b"filenames"] = np.concatenate((self.data1[b"filenames"],self.data2[b"filenames"],self.data3[b"filenames"],self.data4[b"filenames"]))
            self.data[b"labels"] = np.where(self.data[b"labels"] == 3, 0, 1)

            #self.data = self.data1.append(self.data2).append(self.data3).append(self.data4)

            #reshaping image arrayto 32*32*3
            #self.data[b'data'] = self.data[b'data'].apply(lambda x: np.array(x).reshape((32,32,3)).astype(np.uint8))

            #changinglabels to 0 and 1
            #self.data[b'labels'].replace({3: 0, 5: 1}, inplace=True)

        elif subset == 2:
            #validation set
            fdir5 = fdir + "data_batch_5"
            self.data = self.unpickle(fdir5)
            if b'batch_label' in self.data: del self.data[b'batch_label']                        
            self.data[b"data"] = np.swapaxes(self.data[b"data"].reshape((len(self.data[b"data"]), 3, 32, 32)), 1, 3)

            idx_to_remove = np.where((np.array(self.data[b"labels"]) != 3) & (np.array(self.data[b"labels"]) != 5))[0]
             
            self.data[b"data"]=np.delete(self.data[b"data"], idx_to_remove, axis=0)
            self.data[b"labels"]=np.delete(self.data[b"labels"], idx_to_remove, axis=0)
            self.data[b"filenames"]=np.delete(self.data[b"filenames"], idx_to_remove, axis=0)
            self.data[b"labels"] = np.where(self.data[b"labels"] == 3, 0, 1)


        elif subset == 3:
            #test set
            fdir_test = fdir + "test_batch"
            self.data = self.unpickle(fdir_test)
            if b'batch_label' in self.data: del self.data[b'batch_label']                        
            self.data[b"data"] = np.swapaxes(self.data[b"data"].reshape((len(self.data[b"data"]), 3, 32, 32)), 1, 3)

            idx_to_remove = np.where((np.array(self.data[b"labels"]) != 3) & (np.array(self.data[b"labels"]) != 5))[0]
             
            self.data[b"data"]=np.delete(self.data[b"data"], idx_to_remove, axis=0)
            self.data[b"labels"]=np.delete(self.data[b"labels"], idx_to_remove, axis=0)
            self.data[b"filenames"]=np.delete(self.data[b"filenames"], idx_to_remove, axis=0)
            self.data[b"labels"] = np.where(self.data[b"labels"] == 3, 0, 1)

        pass

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        # TODO implement

        return(len(self.data[b"labels"]))
        

    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        # TODO implement
        try:
            return(Sample(idx, self.data[b"data"][idx], self.data[b"labels"][idx]))
        except IndexError as e:
            print("bad index")
            print(e)
            raise IndexError
        

    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        # TODO implement
        u, indices = np.unique(self.data[b"labels"], return_inverse=True)
        return(len(u))
        
"""
obj = PetsDataset("path/to/cifar-10-batches-py/", 1)


import cv2 

cv2.imshow("example",obj.data[b'data'].values[1])
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image





obj = PetsDataset("path/cifar-10-batches-py/",1)

data = obj.data

print((data[b"labels"]))

print((data[b"data"][0].shape))

print(len(data[b"filenames"]))



#used for debugging:


def unpickle( file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


data = unpickle("path/cifar-10-batches-py/data_batch_1")

if b'batch_label' in data: del data[b'batch_label'] 

self.data1[b"data"] = self.data1[b"data"].reshape((len(self.data1[b"data"]), 32, 32, 3))

idx_to_remove = np.where((np.array(self.data1[b"labels"]) != 3) & (np.array(self.data1[b"labels"]) != 5))[0]
 
self.data1[b"data"]=np.delete(self.data1[b"data"], idx_to_remove, axis=0)
self.data1[b"labels"]=np.delete(self.data1[b"labels"], idx_to_remove, axis=0)
self.data1[b"filenames"]=np.delete(self.data1[b"filenames"], idx_to_remove, axis=0)
self.data1[b"labels"] = np.where(self.data1[b"labels"] == 3, 0, 1)
"""
