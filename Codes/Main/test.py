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


#target_data = 'Pascal_HS'
# target_data = 'CIFAR'
target_data = 'Af'


# Arguments
parser = argparse.ArgumentParser(description= target_data + 'PyTorch Test')
parser.add_argument('--model', default='ResNet18', help='model selection(default: ResNet18)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
parser.add_argument('--image_size', type=int, default=256, help='input image size for traning (default:256)')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
args = parser.parse_args()




# Image dir
# img_dir = '../../Datasets/CIFAR/Images/'
# img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-256-8bit/'
# img_dir = '../../../models/Datasets/CIFAR/Images'


if args.image_size == 512:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-512-8bit/'
elif args.image_size == 128:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-128-8bit/'
else:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-256-8bit/'


# Model loading
weights_dir = '../Weights/' + target_data + '/*'
weights_path = glob.glob(weights_dir)

# Choose recent one.
weights_sorted = sorted(weights_path, key=lambda f: os.stat(f).st_mtime, reverse=True)
image_save_point = torch.load(weights_sorted[0])
datetime = weights_sorted[0].rsplit('_',1)[1].rsplit('.',1)[0]
#modelname = weights_sorted[0].split('_',1)[0]

# Load csv
# csv_path = '../../Datasets/CIFAR/Split/split_' + datetime + '.csv'
# csv_path = '../../../models/Datasets/CIFAR/Split/split_2021-01-14-23-12.csv'
csv_path = '../../../data/cleansed/documents/echo_merged_split.csv'
#csv_path = '../../../data/cleansed/documents/echo_merged_split_downsampling.csv'


#results_csv_dir = '../../Results/' + target_data + '/csv'
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
elif args.model == 'ResNet18':
    print('Use default ResNet18.')
    from torchvision.models import resnet18
    net = resnet18(num_classes=classes)
else:
    print('Error occured. No model.')


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


# Load weight
net.load_state_dict(image_save_point)

# Set data loader
test_loader = DataLoader(
      dataset=test_data,
      batch_size=args.batch_size,
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


df_val_test = pd.read_csv(csv_path)[['Img_name', 'Split']]
df_merge = pd.merge(df_concat, df_val_test, on = 'Img_name', how = 'left')


# Save inference results
save_dir = '../../Results/' + target_data + '/Likelihood/'
os.makedirs(save_dir, exist_ok=True)

basename = os.path.splitext(os.path.basename(weights_sorted[0]))[0]

# df_merge.to_csv(save_dir + modelname + '_' + target_data + '_' + datetime + '.csv', index=False)
df_merge.to_csv(save_dir + basename + '.csv', index=False)

