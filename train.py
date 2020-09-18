import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_load import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from data_load import Rescale, RandomCrop, Normalize, ToTensor
from torchvision import transforms, utils

#from models import Net, ModelInitilizer
import models.base_model as modelib
import torch.optim as optim
from torch.autograd import Variable


def train_net(train_loader, net, criterion, optimizer, n_epochs):
    
    # prepare the net for training
    net.train()
    training_loss = []

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
            key_pts = key_pts.type(torch.FloatTensor).cuda()
            images = images.type(torch.FloatTensor).cuda()

            # forward pass to get outputs
            preds = net(images)
            
            loss = criterion(preds, key_pts)
            loss.backward()
            # accumulate gradient
            if batch_i % 2 ==0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            if batch_i % 10 == 9:    # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/10))
                running_loss = 0.0
        training_loss.append(running_loss)

    print('Finished Training')
    return training_loss

def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    plt.figure(figsize=(20,5))
    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i+1)

        # un-transform the image data
        image = test_images[i].data   # get the image from it's Variable wrapper
        image = image.cpu().numpy()   # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))   # transpose to go from torch to numpy image
        image = np.uint8(255*image)
        
        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu().numpy()
        # undo normalization of keypoints  
        predicted_key_pts = predicted_key_pts*112+112
        
        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]         
            ground_truth_pts = ground_truth_pts*112.0+112
        
        # call show_all_keypoints
        #show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
        show_all_keypoints(image, predicted_key_pts, ground_truth_pts)
        plt.axis('off')
    plt.show()

def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    plt.imshow(image)
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')    

def net_sample_output(test_loader, net):
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):
        
        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']
        
        # convert images to FloatTensors
        images = images.type(torch.FloatTensor).cuda()
        key_pts.to("cuda")

        # forward pass to get net output
        preds = net(images)

        # reshape to batch_size x 68 x 2 pts
        preds = preds.view(preds.size()[0], 68, -1)
        
        # break after first image is tested
        if i == 0:
            return images, preds, key_pts


# start to train 
#-------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    data_transform = transforms.Compose([Rescale(250),
                                        RandomCrop(224),
                                        Normalize(),
                                        ToTensor()])

    train_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                        root_dir='./data/training/',
                                        transform=data_transform) 

    batch_size = 10
    train_loader = DataLoader(train_dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=1)

    test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                        root_dir='./data/test/',
                                        transform=data_transform) 
    
    test_loader = DataLoader(test_dataset, 
                             batch_size=batch_size,
                             shuffle=True, 
                             num_workers=1)

    net = modelib.Net()
    print(net)
    net.apply(modelib.ModelInitilizer.weights_init)
    net.load_state_dict(torch.load('./saved_models/facial_keypoints_model_res0.pt'))
    net.to("cuda")
    
    # test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)
    # #print out the dimensions of the data to see if they make sense
    # print("test_imgs_size", test_images.data.size())
    # print("test_output", test_outputs.data.size())
    # print("gt_output", gt_pts.size())
    # image = test_images[0].data   # get the image from it's Variable wrapper
    # image = image.cpu().numpy()   # convert to numpy array from a Tensor
    # image = np.transpose(image, (1, 2, 0))

    # visualize_output(test_images, test_outputs, gt_pts)
    
    n_epochs = 20
    criterion = nn.MSELoss()
    #criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss()
    optimizer = optim.RMSprop(net.parameters(),lr=2.5e-5)
    #optimizer = optim.Adam(net.parameters(), lr=2.5e-4)
    # #optimizer = optim.SGD(net.parameters(), lr =1e-4)
    # # adjust learning rate
    
    training_loss = \
    train_net(train_loader, net, criterion, optimizer, n_epochs)

    plt.figure()
    plt.semilogy(training_loss)
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Loss');

    test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)

    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())    
    visualize_output(test_images, test_outputs, gt_pts)

    ## change the name to something uniqe for each new model
    model_dir = './saved_models/'
    model_name = 'facial_keypoints_model_res1.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir+model_name)