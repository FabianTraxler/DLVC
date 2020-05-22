

from ..dataset import Sample, Subset, ClassificationDataset
import numpy as np

class PetsDataset(ClassificationDataset):
    '''
    Dataset of cat and dog images from CIFAR-10 (class 0: cat, class 1: dog).
    '''
      
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
        
        # initialize the dataset variables
        self.data = np.array([], dtype=np.uint8).reshape(0, 32, 32, 3)
        self.labels = np.array([], dtype=np.uint8)
        self.filenames = list()

        # load the specified dataset
        if subset.value == 1:
            # training set

            # loop over the first 4 files and create a concentated dataset
            for i in range(1,5):
                filename = fdir + "data_batch_" + str(i)
                new_data = self.__load_file__(filename)

                self.data = np.append(self.data, new_data["data"], axis=0)
                self.labels = np.append(self.labels ,new_data["labels"], axis=0)
                self.filenames.append(new_data["filenames"])

        elif subset.value == 2:
            #validation set
            filename = fdir + "data_batch_5"
            new_file = self.__load_file__(filename)
            self.data = new_file['data']
            self.labels = new_file['labels']
            self.filenames = new_file['filenames']

        elif subset.value == 3:
            #test set
            filename = fdir + "test_batch"
            new_file = self.__load_file__(filename)
            self.data = new_file['data']
            self.labels = new_file['labels']
            self.filenames = new_file['filenames']

 
    def __unpickle__(self, filename):
        """
        Read a pickled Version of the Dataset and return the resulting dict
        """
        import pickle
        with open(filename, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def __load_file__(self,filename):
        """
        Load a file from filepath and return labesl, data, and filenames
        """
        data_dict = self.__unpickle__(filename)

        # reshape the images
        data_dict[b'data'] = data_dict[b'data'].reshape((len(data_dict[b"data"]), 3, 32, 32))
        data_dict[b'data'] = data_dict[b'data'].transpose(0, 2,3,1)

        # From RGB to BGR
        data_dict[b'data'] = data_dict[b'data'][..., [2,1,0]]

        # keep only cats and dogs
        ## cat_label = 3, dog_label = 5
        idx_to_remove = np.where((np.array(data_dict[b"labels"]) != 3) & (np.array(data_dict[b"labels"]) != 5))[0]
        for key in data_dict.keys():
            data_dict[key] = np.delete(data_dict[key], idx_to_remove, axis=0)
                
        # rename labels
        ## cat_label = 0, dog_label = 1
        data_dict[b"labels"] = np.where(data_dict[b"labels"] == 3, 0, 1)

        
        return {
                "labels": data_dict[b"labels"], 
                "data": data_dict[b"data"], 
                "filenames": data_dict[b"filenames"]
                }

    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''

        return(len(self.labels))
        
    def __getitem__(self, idx: int) -> Sample:
        '''
        Returns the idx-th sample in the dataset.
        Raises IndexError if the index is out of bounds.
        '''

        if idx >= self.__len__():
            raise IndexError
        else:
            return(Sample(idx, self.data[idx], self.labels[idx]))
        
    def num_classes(self) -> int:
        '''
        Returns the number of classes.
        '''

        u, _ = np.unique(self.labels, return_inverse=True)
        return(len(u))
        
    def image_shape(self) -> tuple:
        return self.__getitem__(0).data.shape