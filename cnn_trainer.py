import torch
import torchvision.transforms.functional as F
import numpy as np
import numbers
import os

from torchvision import datasets, transforms
from torchvision.transforms.functional import pad
from sklearn.model_selection import train_test_split


# reference for padding function and class: https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/2
def get_padding(image):
  w, h = image.size
  max_wh = np.max([w, h])
  h_padding = (max_wh - w) / 2
  v_padding = (max_wh - h) / 2
  l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
  t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
  r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
  b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
  padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
  return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


img_h, img_w = 512, 512

# paths to the test and train folders
train_path = '/home/rhys/ml/train'
val_path = '/home/rhys/ml/val'
test_path = '/home/rhys/ml/test'

train_transform_rotation = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomRotation((-175, 175)),
     transforms.Resize((img_h, img_w))
])
train_transform_affine = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomAffine(175, (0.1, 0.3), (0.8, 1.2)),
     transforms.Resize((img_h, img_w))
])
train_transform_flipx = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomAffine(175, (0.1, 0.3), (0.8, 1.2)),
     transforms.Resize((img_h, img_w)),
     transforms.RandomVerticalFlip(1),
])
train_transform_flipy = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomAffine(175, (0.1, 0.3), (0.8, 1.2)),
     transforms.Resize((img_h, img_w)),
     transforms.RandomHorizontalFlip(1),
])
train_transform_norm = transforms.Compose([
     NewPad(),
     transforms.ToTensor(),
     transforms.Resize((img_h, img_w)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# we don't want to perform any augmentations to the test set
test_transform = transforms.Compose([
     NewPad(),
     transforms.ToTensor(),
     transforms.Resize((img_h, img_w)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# load our train and test data sets
train1 = datasets.ImageFolder(train_path, transform=train_transform_norm)
train2 = datasets.ImageFolder(train_path, transform=train_transform_rotation)
train3 = datasets.ImageFolder(train_path, transform=train_transform_affine)
train4 = datasets.ImageFolder(train_path, transform=train_transform_flipy)
train5 = datasets.ImageFolder(train_path, transform=train_transform_flipx)

train_dataset = torch.utils.data.ConcatDataset([train1, train2, train3, train4, train5])
val_dataset = datasets.ImageFolder(val_path, transform=test_transform)
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

# use DataLoader which batches the data to preserve RAM
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

# Model creation
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=1),
            #nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining a 2D convolution layer
            nn.Conv2d(3, 3, kernel_size=7, stride=1, padding=1),
            #nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(11163,75),
            nn.Linear(75, 4),
            nn.LogSoftmax(dim=1)
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

model = Net()
# weight balancing (manually calculated)
weights = torch.tensor([1.1618, 0.9677, 0.7947, 1.1798], dtype=torch.float32)
loss_function = nn.CrossEntropyLoss(weight=weights)
# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_ar = []
val_loss_ar = []
# Train the model
epochs = 10
for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = loss_function(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item()*inputs.size(0)

        # Since our model outputs a LogSoftmax, find the real 
        # percentages by reversing the log function
        output = torch.exp(output)
        # Get the top class of the output
        top_p, top_class = output.topk(1, dim=1)
        # See how many of the classes were correct?
        equals = top_class == labels.view(*top_class.shape)
        # Calculate the mean (get the accuracy for this batch)
        # and add it to the running accuracy for this epoch
        train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Print the progress of our training
        counter += 1
        print(counter, "/", len(train_loader))

    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = loss_function(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epochopen('svm_model.pkl', 'wb')
            val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(val_loader))
    
    
    torch.save(model.state_dict(), './all_augmentations_weighted.pt')
    # Get the average loss for the entire epoch
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = val_loss/len(val_loader.dataset)
    train_loss_ar.append(train_loss)
    val_loss_ar.append(valid_loss)
    # Print out the information
    print('Training Accuracy: {:.6f} \tValidation Accuracy: {:.6f}'.format(train_accuracy/len(train_loader), val_accuracy/len(test_loader)))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

import pickle

pickle.dump(val_loss_ar, open('val_loss.array', 'wb'))
pickle.dump(train_loss_ar, open('train_loss.array', 'wb'))
