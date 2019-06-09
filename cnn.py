#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I

import torch.optim as optim

from models import Net

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #input image 1x224x224
        
        self.pool = nn.MaxPool2d(2, 2) #all pooling layers will be the same

        self.conv1 = nn.Conv2d(1, 32, 4)    #32x221x221 -> #32x110x110 after pooling
       
        self.conv2 = nn.Conv2d(32, 64, 3)   #64x108x108 -> #64x54x54 after pooling

        self.conv3 = nn.Conv2d(64, 128, 2)  #128x53x53 -> #128x26x26 after pooling

        self.conv4 = nn.Conv2d(128, 256, 1) #256x26x26 -> #256x13x13 after pooling

        self.fc1 = nn.Linear(256*13*13,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,68*2)

        #initializations found in research paper
        # I.uniform_(self.conv1.weight)
        # I.uniform_(self.conv2.weight)
        # I.uniform_(self.conv3.weight)
        # I.xavier_uniform_(self.fc1.weight)
        # I.xavier_uniform_(self.fc2.weight)
        # I.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):

        dropout_1 = nn.Dropout(0.1)
        dropout_2 = nn.Dropout(0.2)
        dropout_3 = nn.Dropout(0.3)
        dropout_4 = nn.Dropout(0.4)
        dropout_5 = nn.Dropout(0.5)
        dropout_6 = nn.Dropout(0.6)
                
        x = dropout_1(self.pool(F.elu(self.conv1(x))))
        x = dropout_2(self.pool(F.elu(self.conv2(x))))
        x = dropout_3(self.pool(F.elu(self.conv3(x))))
        x = dropout_4(self.pool(F.elu(self.conv4(x))))
        
        x = x.view(x.size(0), -1) # flatten
        
        x = dropout_5(F.elu(self.fc1(x)))
        x = dropout_6(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
        

#print the net to make sure it looks ok    
net = Net()
print(net)

#transform the data for the cnn
data_transform = transforms.Compose([Rescale(250),
                                     RandomCrop(224),
                                     Normalize(),
                                     ToTensor()])

# testing that you've defined a transform
assert(data_transform is not None), 'Define a data_transform'

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                             root_dir='data/training/',
                                             transform=data_transform)

#torch tensor number of images
print('Number of images: ', len(transformed_dataset))

# iterate through the transformed dataset and print some stats about the first few samples
#make sure it looks ok
for i in range(4):
    sample = transformed_dataset[i]
    print(i, sample['image'].size(), sample['keypoints'].size())
    #print(sample['keypoints'])


# load training data in batches
batch_size = 128

train_loader = DataLoader(transformed_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)


# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                             root_dir='data/test/',
                                             transform=data_transform)

# load test data in batches
batch_size = 10

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=0)

#do a first test of the model on a batch of test images
def net_sample_output():    
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)
        
        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts




# call the above function
# returns: test images, test predicted keypoints, test ground truth keypoints
test_images, test_outputs, gt_pts = net_sample_output()

# print out the dimensions of the data to see if they make sense
print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


# visualize the output
# by default this shows a batch of 10 images
def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):

    for i in range(batch_size):
        plt.figure(figsize=(20,10))
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*50.0+100
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*50.0+100
        
        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
            
        plt.axis('off')

    plt.show()
    
# call it
#visualize_output(test_images, test_outputs, gt_pts)

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters(), lr=0.001)

def train_net(n_epochs):
    # prepare the net for training
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            #print(key_pts)
        
            
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            
            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if True: #batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0

    print('Finished Training')

# train your network
n_epochs = 50

train_net(n_epochs)    


#get a sample of test data again
test_images, test_outputs, gt_pts = net_sample_output()

print(test_images.data.size())
print(test_outputs.data.size())
print(gt_pts.size())

visualize_output(test_images, test_outputs, gt_pts)

## TODO: change the name to something uniqe for each new model
model_dir = 'saved_models/'
model_name = 'test_new.pt'

# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)

