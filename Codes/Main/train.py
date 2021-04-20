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

# Arguments
parser = argparse.ArgumentParser(description= 'PyTorch Training')
parser.add_argument('--model', default='ResNet18', help='model selection(default: ResNet18)')
parser.add_argument('--optimizer', default='SGD', help='specify optimzer (default: SGD)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--val_batch_size', type=int, default=64, metavar='N', help='input batch size for val (default: 1000)')
parser.add_argument('--image_size', type=int, default=256, help='input image size for traning (default:256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#parser.add_argument('--target', default='CIFAR', help='target data selection(default: CIFAR)')
args = parser.parse_args()

# Choose datasets
#target_data = args.target
#target_data = 'Pascal_HS'

# Image dir
#img_dir = '../../../models/Datasets/CIFAR/Images'
#target_dir = '../../../data/extracted/png-flat-using-crop0.8-small128/'
#img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-256-8bit/'
#img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-512-8bit/'

if args.image_size == 512:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-512-8bit/'
elif args.image_size == 128:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-128-8bit/'
else:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-256-8bit/'

# Load csv
#csvs_path = glob.glob('../../Datasets/pascal_image_corpus/v0.2/data-cut/split/*.csv')
#csvs_path = glob.glob(csv_dir + '*.csv')
#csv_path = sorted(csvs_path, key=lambda f: os.stat(f).st_mtime, reverse=True)[0]

#csv_path = '../../../models/Datasets/CIFAR/Split/split_2021-01-14-23-12.csv'
csv_path = '../../../data/cleansed/documents/echo_merged_split.csv'
# csv_path = '../../../data/cleansed/documents/echo_merged_split_downsampling.csv'


# Check categories
classes = pd.read_csv(csv_path)['Label'].nunique()
print('Preparing ' + str(classes) + ' categories classification')
# Define savename
#datetime = csv_path.rsplit('_', 1)[1].rsplit('.', 1)[0]


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
if args.model == 'ResNet':
    print('ResNet50 is selected for the model')
    from torchvision.models import resnet50
    net = resnet50(num_classes=classes)
elif args.model == 'Inception':
    print('Inceptionv3 is selected for the model')
    from torchvision.models import inception_v3
    net = inception_v3(num_classes=classes)
elif args.model == 'DenseNet':
    print('DenseNet121 is selected for the model')
    from torchvision.models import densenet121
    net = densenet121(num_classes=classes)
elif args.model == 'SqueezeNet':
    print('SqueezeNet is selected for the model')
    from torchvision.models import squeezenet1_0
    net = squeezenet1_0(num_classes=classes)
elif args.model == 'VGG':
    print('VGG is selected for the model')
    from torchvision.models import vgg16
    net = vgg16(num_classes=classes)
else:
    print('Use default ResNet18.')
    from torchvision.models import resnet18
    net = resnet18(num_classes=classes)


# Define optimizer
if args.optimizer == 'Adadelta':
    print('Adadelta is a selected for the optimizer')
    optimizer = optim.Adadelta(net.parameters())
elif args.optimizer == 'Adam':
    print('Adam is a selected for the optimizer')
    optimizer = optim.Adam(net.parameters())
elif args.optimizer == 'RMSprop':
    print('RMSprop is a selected for the optimizer')
    optimizer = optim.RMSprop(net.parameters())
else:
    print('Use default SGD.')
    optimizer = optim.SGD(net.parameters(), lr=0.01) 


# Set data loader
train_loader = DataLoader(
      dataset=train_data,          # set dataset
      batch_size=args.batch_size,  # set batch size
      shuffle=True,                # shuffle or not
      num_workers=0)               # set number of cores

valid_loader = DataLoader(
      dataset=val_data,
      batch_size=args.val_batch_size,
      shuffle=True,
      num_workers=0)


# Select device
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#net = net.to(device)

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
# optimizer = optim.Adam(net.parameters())

###  Training
num_epochs = args.epochs

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
    #print('epoch [{}/{}], avg_train_loss: {avg_val:.4f}, avg_train_acc: {avg_acc:.4f}'.format(epoch+1, num_epochs, avg_val=avg_train_loss, avg_acc=avg_train_acc))

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
    #print('epoch: [{}/{}], avg: train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}'.format(epoch+1, num_epochs, train_loss=avg_train_loss, train_acc=avg_train_acc, val_loss=avg_val_loss, val_acc=avg_val_acc))

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

dt_now = datetime.datetime.now()
dt_name = dt_now.strftime('%Y-%m-%d-%H-%M')

# Save weights
save_weight_dir = '../Weights/Af/'
os.makedirs(save_weight_dir, exist_ok=True)
# weights_path = save_weight_dir + args.model + args.optimizer + '_epoch_' + str(epoch_best) + '_val-loss_' + f'{val_loss_best:.4f}' + '_Af_' + dt_name  + '.pt'

device_name = 'GPU-' + '-'.join(map(str, gpu_ids)) if gpu_ids else 'CPU'

basename = args.model + '_' + \
           args.optimizer + '_' + \
           'batch-size-' + str(args.batch_size) + '_' + \
           'image-size-' + str(args.image_size) + '_' + \
           'epochs-' + str(args.epochs) + '_' + \
           'val-best-epoch-' + str(epoch_best) + '_' + \
           'val-loss-' + f'{val_loss_best:.4f}' + '_' + \
           device_name + '_' + \
           'Af_' + dt_name


weights_path = save_weight_dir + basename + '.pt'
torch.save(weight_best, weights_path)


# Save learning curve
save_lc_dir = '../../Results/Af/LC/'
os.makedirs(save_lc_dir, exist_ok=True)

lc_path = save_lc_dir + basename + '.csv' 
df_lc = pd.DataFrame([train_loss_list, train_acc_list, val_loss_list, val_acc_list], index=['train_loss','train_acc','val_loss','val_acc']).T
# df_lc.to_csv(save_lc_dir + args.model + args.optimizer + '_Af_' + dt_name + '.csv', index=False)
df_lc.to_csv(lc_path, index=False)
