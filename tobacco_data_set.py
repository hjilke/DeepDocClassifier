from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import glob2


ENCODING_CLASS = {
    'Note': 0,
    'Scientific': 1,
    'Form': 2,
    'Report': 3,
    'ADVE': 4,
    'Memo': 5, 
    'Resume': 6,
    'Email': 7,
    'News': 8,
    'Letter': 9
}

DECODING_CLASS = [
     'Note',
     'Scientific',
     'Form',
     'Report',
     'ADVE',
     'Memo', 
     'Resume',
     'Email',
     'News',
     'Letter' 
]

class SourceReader:

    def __init__(self, path_to_dir):
        """
        Reader for the Tobacco-3482 data set. The reader assumes that each class
        is a subfolder containing all the images that belong to this class.
        e.g. 
            source_dir/
                Note
                    *.img
                Scientific
                ...
                Letter

        Parameters
        ----------
        path_to_dir : str
            Path to directory

        References
        ----------
        D. Lewis, G. Agam, S. Argamon, O. Frieder, D. Grossman, and
        J. Heard, "Building a test collection for complex document information
        processing," in ACM SIGIR, New York, USA, 2006, pp. 665-666.
        """
        self.path_to_dir = path_to_dir

    def read_class_labels(self):
        """

        Returns
        -------
        tuple of size 2
            Return new x, y cooridnate.
        """
        return os.listdir(self.path_to_dir)

    def get_image_paths(self, class_labels):
        """
        For each class store all pathes to the images.
        
        Parameters
        ----------
        class_labels : list
            List of all class names
        
        Returns
        -------
        dict
            Return dictionary containing pairs of class name and the list of image pathes
            belonging to this class
        """
        data_dict = {}
        for label in class_labels:
            built_path = os.path.join(self.path_to_dir, label, "*.jpg")
            data_dict[label] = list(glob2.glob(built_path))
        return data_dict

    
class RandomPartitioner:

    def __init__(self, data_dict, num_images_per_class):
        """
        Implement the random partion scheme outlined in the experiment section of the 
        paper referenced below.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing pairs of class name and the list of image pathes
            belonging to this class
        num_images_per_class: int
            Number of fixed training images for each class

        References
        ----------
        Muhammad Zeshan Afzal et al. "DeepDocClassifier: Document Classification with
        Deep Convolutional Neural Network" in 2015 13th International Conference on Document 
        Analysis and Recognition (ICDAR).
        """
        self.num_images_per_class = num_images_per_class
        self.data_dict = data_dict

    def partition(self):
        """
        Partition the data according to scheme outlined in the experiment section of the 
        paper referenced below.
        
        Returns
        -------
        list of size 2
            Return the training partition and the test partition
        
        References
        ----------
        Muhammad Zeshan Afzal et al. "DeepDocClassifier: Document Classification with
        Deep Convolutional Neural Network" in 2015 13th International Conference on Document 
        Analysis and Recognition (ICDAR).
        """
        train_partition = {}
        test_partition = {}
        for label, data in self.data_dict.items():
            idx = np.arange(len(data))
            np.random.shuffle(idx)
            train_partition[label] = [data[i] for i in idx[:self.num_images_per_class]]
            test_partition[label] = [data[i] for i in idx[self.num_images_per_class + 1:]]
        return train_partition, test_partition
    

class TobaccoDataSet(Dataset):

    def __init__(self, data_dict, transform=None):
        """
        Implement the torch *Dataset* wrapper for the Tobacco-3482 data set.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing pairs of class name and the list of image pathes
            belonging to this class
        transform: torchvision.transforms
            Applied transformations

        References
        ----------
        D. Lewis, G. Agam, S. Argamon, O. Frieder, D. Grossman, and
        J. Heard, "Building a test collection for complex document information
        processing," in ACM SIGIR, New York, USA, 2006, pp. 665-666.
        """
        self.images = []
        for label in data_dict:
            num_samples = len(data_dict[label])
            self.images += list(zip([label] * num_samples, data_dict[label]))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label, img_name = self.images[idx]
        sample = Image.open(img_name).convert("RGB")
        if self.transform:
            sample = self.transform(sample)
        return sample, ENCODING_CLASS[label]


