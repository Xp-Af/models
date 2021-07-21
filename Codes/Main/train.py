#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import pandas as pd
import argparse
import os
import glob
import datetime

img_dir = 'path/to/dir'
csv_path = 'path/to/groundtruth_csv'

# Check categories
classes = pd.read_csv(csv_path)['Label'].nunique()
print('Preparing ' + str(classes) + ' categories classification')

# Dataset Preparation
class LoadDataSet(Dataset):
    def __init__(self, csv_path, split):
        self.df_original = pd.read_csv(csv_path)
        self.df = self.df_original[self.df_original['Split'] == split]
        self.label_index = self.df.columns.get_loc('Label')
        self.id_index = self.df.columns.get_loc('Img_name')
        # The values of normalization depends on the target dataset.
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.8941926, 0.8941926, 0.8941926],std=[0.25469009, 0.25469009, 0.25469009])])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # Load img path and labels from dataframe
        label = self.df.iat[idx, self.label_index]
        #print(label) #For debag
        img_path = os.path.join(img_dir, self.df.iat[idx, self.id_index])
        #print(img_path) #For debag
        # Read images
        image = Image.open(img_path).convert('RGB')
        # Process image
        if self.transform:
            image = self.transform(image)
        return image, label

# Load data
train_data = LoadDataSet(csv_path, 'train')
val_data   = LoadDataSet(csv_path, 'val')

print('train_data =', len(train_data))
print('val_data =', len(val_data))

# Define model
net = resnet18(num_classes=classes)

# Define optimizer
optimizer = optim.Adam(net.parameters())

# Set data loader
train_loader = DataLoader(
      dataset=train_data,          # set dataset
      batch_size=64,  # set batch size
      shuffle=True,                # shuffle or not
      num_workers=0)               # set number of cores

valid_loader = DataLoader(
      dataset=val_data,
      batch_size=64,
      shuffle=True,
      num_workers=0)

# Set GPU IDs
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
if len(gpu_ids) > 0: # If args.gpu_ids = '-1' then gpu_ids=[]
        torch.cuda.set_device(gpu_ids[0])

# Get device name: CPU or GPU if use CPU, this line is needed.
#device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
device = torch.device('cuda') if gpu_ids else torch.device('cpu')

# Use multi-GPU
if len(gpu_ids) > 0:
    assert(torch.cuda.is_available())
    net.to(gpu_ids[0])
    net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPU

# Optimizing
criterion = nn.CrossEntropyLoss()

###  Training
num_epochs = 100

# Initialize list for plot graph after training
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
# val_loss_best, epoch_val_loss_best = 0, 0
val_loss_best = 0

print ('Training is starting ...')
for epoch in range(num_epochs):
    # Initialize each epoch
    train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
    # ====== Train mode ======
    net.train()
    for i, (images, labels) in enumerate(train_loader):  # Iteration the number of mini-batches
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                      # Reset gradient
        outputs = net(images)                      # Feedforward
        loss = criterion(outputs, labels)          # Calculate loss
        train_loss += loss.item()                  # Summation of losses train_loss
        acc = (outputs.max(1)[1] == labels).sum()  # The number of correct predictions
        train_acc += acc.item()                    # Summation of the number of correct predictions
        loss.backward()   # Backpropagation
        optimizer.step()  # Update weights
 
    avg_train_loss = train_loss / len(train_loader.dataset)  # Average of losses
    avg_train_acc = train_acc / len(train_loader.dataset)    # Average of acc
    # Print log. Note that python starts from 0.

    # ====== Valid mode ======
    net.eval()
    with torch.no_grad(): # Stop unneeded calculations
        for i, (images, labels) in enumerate(valid_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            acc = (outputs.max(1)[1] == labels).sum()
            val_acc += acc.item()

    avg_val_loss = val_loss / len(valid_loader.dataset)
    avg_val_acc = val_acc / len(valid_loader.dataset)
    # Print log. Note that python starts from 0.
    print ('epoch [{}/{}], train_loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'.format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))

    # Append list for polt graph after training
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

    # Keep the best weight when loss it the lowest
    if(val_loss_best == 0):
        # Initialize
        val_loss_best = avg_val_loss
        epoch_best = epoch + 1 # Python starts from 0
        weight_best = net.state_dict()
    else:
        if(val_loss_best > avg_val_loss):
            val_loss_best = avg_val_loss
            epoch_best = epoch + 1 # Python starts from 0
            weight_best = net.state_dict()

            # For debug
            print('Updated val_loss_best!')
        else:
            pass

print ('Training finished !')
