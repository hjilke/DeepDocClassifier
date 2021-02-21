from torch import nn
import numpy as np
import torch

class DeepDocClassifier(nn.Module):

    def __init__(self, alex_base_model, num_classes=10):
        """
        Implementation of the DeepDocClassifier outlined in the paper
        referenced below. Overall, the model is an extension of the AlexNet.

        The model uses the weights of AlexNet trained on imagenet.

        Parameters
        ----------
        alex_base_model : torchvision.models
            Pytorch implementation of the AlexNet
        num_classes: int
            Number of classes

        References
        ----------
        Muhammad Zeshan Afzal et al. "DeepDocClassifier: Document Classification with
        Deep Convolutional Neural Network" in 2015 13th International Conference on Document 
        Analysis and Recognition (ICDAR).
        """
        super().__init__()
        
        self.num_classes = num_classes
        base_model = alex_base_model(pretrained=True)

        self.conv_layer = list(base_model.children())[0]
        self.glob_avg_pooling = list(base_model.children())[1]
        self.fc_layers = nn.Sequential(
            list(base_model.children())[2][:-1],
            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.glob_avg_pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x



def train(net, train_set, val_set, epochs=10, lr=0.001, momentum=0.9, decay=0.0005, 
          print_every=10, n_epochs_stop=4, path_to_file=None):
    """
    Helper method to train AlexNet according to the scheme defined in the paper below.
    As such, optimizer and loss are fixed


    Parameters
    ----------
    net : torchvision.models
        Pytorch implementation of the AlexNet
    train_set: torch.utils.DataLoader
        Dataloader for the training set
    val_set: torch.utils.DataLoader
        Dataloader for the validation set
    epochs: int
        Number of epochs
    lr: float
        Learning Rate
    momentum: float
        Momentum for SGD
    decay: float
        Weight decay for SGD
    print_every: int
        Print/calc val loss all *print_every* batches
    n_epochs_stop: int
        Early stopping, based on no improvement on the validation set
    path_to_file: str
        File to store the model

    References
    ----------
    Muhammad Zeshan Afzal et al. "DeepDocClassifier: Document Classification with
    Deep Convolutional Neural Network" in 2015 13th International Conference on Document 
    Analysis and Recognition (ICDAR).
 
    """
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    criterion = nn.CrossEntropyLoss()
    min_val_loss = None
    epochs_no_improve = 0
    counter = 0

    for e_idx in range(epochs):

        for image, label in train_set:
            counter += 1
            net.zero_grad()
            output = net(image)
            loss = criterion(output, label)
            loss.backward()
            opt.step()
            
            if counter % print_every == 0:
                val_losses = []
                net.eval()
                for image, label in val_set:
                    output = net(image)
                    val_loss = criterion(output, label)
                    val_losses.append(val_loss.item())
                
                if min_val_loss and val_loss < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = val_loss
                else:
                    epochs_no_improve += 1

                net.train() 
                
                print("Epoch: {}/{}...".format(e_idx + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

                if path_to_file:
                        torch.save(net.state_dict(), path_to_file)
            
            if e_idx > 5 and epochs_no_improve == n_epochs_stop:
                print('Early stopping!' )
                break
       








