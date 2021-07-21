#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import argparse
import glob
import os

img_dir = 'path/to/img'

# Model loading
weights_path = 'path/to/weights'

# Load csv
csv_path = 'path/to/groundtruth'
classes = pd.read_csv(csv_path)['Label'].nunique()

class LoadTestDataSet(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)[(pd.read_csv(csv_path)['Split'] == 'test') | (pd.read_csv(csv_path)['Split'] == 'val')]
        self.label_index = self.df.columns.get_loc('Label')
        self.id_index = self.df.columns.get_loc('Img_name')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        # Read img path and labels from dataframe
        label = self.df.iat[idx, self.label_index]
        #print(label) #For debag
        img_path = os.path.join(img_dir, self.df.iat[idx, self.id_index])
        #print(img_path) #For debag
        img_name = self.df.iat[idx, self.id_index]
        #print(img_name) #For debag
        # Read images
        image = Image.open(img_path).convert('RGB')
        # Process image
        if self.transform:
            image = self.transform(image)
        return image, label, img_name

# Load data
test_data = LoadTestDataSet(csv_path)

print('test/val data =', len(test_data))

net = resnet18(num_classes=classes)

# Select device

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
device = torch.device('cuda') if gpu_ids else torch.device('cpu')

# Use multi-GPU
if len(gpu_ids) > 0:
    assert(torch.cuda.is_available())
    net.to(gpu_ids[0])
    net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPU

# Load weight
net.load_state_dict(image_save_point)

# Set data loader
test_loader = DataLoader(
      dataset=test_data,
      batch_size=64,
      shuffle=False,
      num_workers=0)


# ====== Test mode ======
print ('inference is starting ...')
net.eval()
with torch.no_grad():
    total = 0
    test_acc = 0
    class_names = [str(i) for i in list(range(classes))]
    for i, (images, labels, img_names) in enumerate(test_loader): # process by all_images/batch_size
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # Predict with values between 0-1
        likelihood_ratio = nn.functional.softmax(outputs, dim=1)
        test_acc += (outputs.max(1)[1] == labels).sum().item()
        total += labels.size(0)

        labels = labels.to('cpu').detach().numpy().copy()
        likelihood_ratio = likelihood_ratio.to('cpu').detach().numpy().copy()

        s_img_names = pd.Series(img_names, name='Img_name')
        s_labels = pd.Series(labels, name='Label')
        df_img_name_label = pd.concat([s_img_names, s_labels], axis=1)
        df_likelihood_ratio = pd.DataFrame(likelihood_ratio, columns=class_names)

        if i == 0:
            df_concat = pd.concat([df_img_name_label, df_likelihood_ratio], axis=1)
        else:
            df_add = pd.concat([df_img_name_label, df_likelihood_ratio], axis=1)
            df_concat = df_concat.append(df_add, ignore_index=True)

    print('inference_accuracy: {} %'.format(100 * test_acc / total))
print ('inference finished !')
