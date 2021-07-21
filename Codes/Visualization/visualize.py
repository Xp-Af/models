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


# Model loading
weights_dir = 'path/to/weight'


csv_path = 'path/to/ground_truth'
df = pd.read_csv(csv_path)
classes = df['Label'].nunique()


img_dir = 'path/to/dir'


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
    
    basename = os.path.splitext(os.path.basename(weights_sorted[0]))[0]
    save_dir = 'path/to/save_dir'
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = save_dir + img_file
    transforms.ToPILImage()(grid_image).save(save_path)
