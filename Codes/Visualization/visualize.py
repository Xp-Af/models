import os
import PIL
import argparse
import numpy as np
import datetime
import pandas as pd
import glob
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#target_data = 'Pascal_HS'
target_data = 'Af'

# Model loading
weights_dir = '../Weights/' + target_data + '/*'
weights_path = glob.glob(weights_dir)
# Choose recent one
weights_sorted = sorted(weights_path, key=lambda f: os.stat(f).st_mtime, reverse=True)
image_save_point = torch.load(weights_sorted[0])
datetime = weights_sorted[0].rsplit('_',1)[1].rsplit('.',1)[0]

# Load csv
#csv_path = '../../Datasets/pascal_image_corpus/v0.2/data-cut/split/split_' + datetime + '.csv'
#csv_path = '../../Datasets/CIFAR/Split/split_' + datetime + '.csv'
csv_path = '../../../data/cleansed/documents/echo_merged_split.csv'
df = pd.read_csv(csv_path)
classes = df['Label'].nunique()

# Arguments
parser = argparse.ArgumentParser(description=target_data + 'PyTorch Visualization')
parser.add_argument('--model', default='resnet18', help='model selection(default: resnet18)')
parser.add_argument('--image_size', type=int, default=256, help='input image size for traning (default:256)')
args = parser.parse_args()

if args.image_size == 512:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-512-8bit/'
else:
    img_dir = '../../../../TRPG/data/cleansed/radiographs/png-crop0.8-256-8bit/'

# Define model
if args.model == 'ResNet':
    print('ResNet50 is selected for the model')
    from torchvision.models import resnet50
    net = resnet50(num_classes=classes)
    configs = [dict(model_type='resnet', arch=net, layer_name='layer4')]
elif args.model == 'Inception':
    print('Inceptionv3 is selected for the model')
    from torchvision.models import inception_v3
    net = inception_v3(num_classes=classes)
    configs = [dict(model_type='inception', arch=net, layer_name='layer4')]
elif args.model == 'DenseNet':
    print('DenseNet121 is selected for the model')
    from torchvision.models import densenet121
    net = densenet121(num_classes=classes)
    configs = [dict(model_type='densenet', arch=net, layer_name='features_norm5')]
elif args.model == 'SqueezeNet':
    print('SqueezeNet is selected for the model')
    from torchvision.models import squeezenet1_0
    net = squeezenet1_0(num_classes=classes)
    configs = [dict(model_type='squeezenet', arch=net, layer_name='features_12_expand3x3_activation')]
elif args.model == 'VGG':
    print('VGG is selected for the model')
    from torchvision.models import vgg16
    net = vgg16(num_classes=classes)
    configs = [dict(model_type='vgg', arch=net, layer_name='features_29')]
else:
    print('Use default ResNet18')
    from torchvision.models import resnet18
    net = resnet18(num_classes=classes)
    configs = [dict(model_type='resnet', arch=net, layer_name='layer4')]

# Target data
df_val_test = df[(df['Split'] == 'val') | (df['Split'] == 'test')]

# Model load
net.load_state_dict(image_save_point)

for img_file in df_val_test['Img_name']:
    img_path = os.path.join(img_dir, img_file)

    pil_img = PIL.Image.open(img_path).convert('RGB')
    pil_img

    torch_img = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])(pil_img).to(device)
    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]

    for config in configs:
        config['arch'].to(device).eval()

    cams = [
        [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
        for config in configs
    ]

    images = []
    for gradcam, gradcam_pp in cams:
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask, torch_img)

        mask_pp, _ = gradcam_pp(normed_torch_img)
        heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

        images.extend([torch_img.cpu(), heatmap, heatmap_pp, result, result_pp])

    grid_image = make_grid(images, nrow=5)
    #img_path.split('/', 7)[7]
    basename = os.path.splitext(os.path.basename(weights_sorted[0]))[0]
    save_dir = '../../Results/' + target_data + '/GradCAM/' + basename + '/'
    os.makedirs(save_dir, exist_ok=True)
    # Save images
    save_path = save_dir + img_file
    transforms.ToPILImage()(grid_image).save(save_path)
