from torchvision.models.alexnet import alexnet
from model import DeepDocClassifier, train
from tobacco_data_set import SourceReader, TobaccoDataSet, RandomPartitioner
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch

import numpy as np
import os
import uuid
import argparse

root = os.path.dirname(__name__)


def train_validation_split(data_size, train_frac=0.8):
    indices = list(range(data_size))
    train_size = int((0.8 * data_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:train_size], indices[train_size+1:]
    return SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)


def create_partition(data_dict, n_train_images):
    partitioner = RandomPartitioner(data_dict, n_train_images)
    train_partition, test_partition = partitioner.partition()
    return train_partition, test_partition


def create_set_of_partitions(data_dict, random_partitions, path_to_partitions):
    for n_train_images in random_partitions:
        train_partition, test_partition = create_partition(data_dict, n_train_images)

        train_data = TobaccoDataSet(train_partition, transform=transforms)
        test_data = TobaccoDataSet(test_partition, transform=transforms)

        file_name = "data-partition-{}-size_train-{}".format(uuid.uuid4(), n_train_images)
        file_path = os.path.join(path_to_partitions, file_name)
        data = [train_data, test_data]
        torch.save(data, file_path)


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((227,227))
]) 


def train_on_partitions(path_to_partitions, max_epochs, path_to_models):
    
    partitions = [partition for partition in os.listdir(path_to_partitions) if partition.startswith("data")]
    print("#######################################################")
    print("Start training on {} partitions".format(len(partitions)))
    
    for run_id, partition_id in enumerate(partitions):
        print("Partition #{} ".format(run_id))
        print("PartitionID #{} ".format(partition_id))
        
        partition_path = os.path.join(path_to_partitions, partition_id)
        train_data, _ = torch.load(partition_path)
        time_stamp = os.path.join(path_to_models, "model-partitionID-{}".format(partition_id))

        train_sampler, val_sampler = train_validation_split(len(train_data), train_frac=0.8)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_data, batch_size=10, sampler=val_sampler)

        print("Size of train-set {}".format(len(train_loader) * 10))
        print("Size of val-set {}".format(len(val_loader) * 10))
        
        model = DeepDocClassifier(alexnet, 10)
        
        if os.path.exists(time_stamp):
            print("Load Model, as model exists, PartitionID #{} ".format(partition_id))
            model.load_state_dict(torch.load(time_stamp))            
        
        train(model, train_loader, val_loader, epochs=max_epochs, lr=0.0001, momentum=0.9, decay=0.0005, 
            print_every=30, path_to_file=time_stamp)
        print("#######################################################\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--build_partitions', action='store_true', default=None)
    parser.add_argument('--train_models', action='store_true', default=None)
    parser.add_argument('--data_path', default=None, type=str)

    args = parser.parse_args()
    path_to_partitions = os.path.abspath("./partitions")
    path_to_models = os.path.abspath("./models")

    if args.data_path is None:
        path_to_tobacco_data = os.path.abspath(os.path.join(root, "data/Tobacco3482-jpg"))
    else:
        path_to_tobacco_data = args.data_path

    file_reader = SourceReader(path_to_tobacco_data)
    labels = file_reader.read_class_labels()
    data_dict = file_reader.get_image_paths(labels)

    if args.build_partitions:
        create_set_of_partitions(data_dict, [20, 40, 60, 80, 100], path_to_partitions)
        partitions = [partition for partition in os.listdir(path_to_partitions) if partition.startswith("data")]
        for partition in partitions:
            print("Created Partition {}".format(partition))
    

    if args.train_models:
        train_on_partitions(path_to_partitions=path_to_partitions, max_epochs=15, path_to_models=path_to_models)
        partitions = [partition for partition in os.listdir(path_to_partitions) if partition.startswith("model")]
        for partition in partitions:
            print("Created Models {}".format(partition))

