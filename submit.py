import configs
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 1'

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torchvision.transforms as T

from models.pspnet import PSPNet

from utils.rle2img import rle_encode, rle_decode

import matplotlib.pyplot as plt
import albumentations as A


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

subm = []

# net = FCN8s(nclass= 1)
# net = UNet(n_channels=3, n_classes=1, bilinear=True)
net = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=1, zoom_factor=8, use_ppm=True,criterion=nn.BCEWithLogitsLoss(), pretrained=True)
net.load_state_dict(torch.load("exps/exp_2021Y_10M_06D_17H_58M/checkpoints/epoch_99.pth"))
net.to(device)
net.eval()

test_mask = pd.read_csv('data/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
test_mask['name'] = test_mask['name'].apply(lambda x: 'data/test_a/' + x)

trans= T.Compose([
    T.ToPILImage(),
    T.Resize(configs.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize([0.40464398, 0.42690134, 0.39236653],
                [0.20213476, 0.18353915, 0.17596193])
])

for idx, name in enumerate(test_mask['name'].iloc[:]):
    image = cv2.imread(name)
    image = trans(image)
    image = torch.unsqueeze(image, dim=0).float()

    with torch.no_grad():
        image = image.to(device)
        # print(image.shape)
        # FCN
        # score = net(image)[0]
        score, _, _ = net(image, torch.zeros((configs.IMAGE_SIZE,configs.IMAGE_SIZE)).to(device))
        score_sigmoid = score.sigmoid().cpu().numpy()
        score_sigmoid = (score_sigmoid >= 0.5).astype(np.uint8)
        score_sigmoid = A.resize(score_sigmoid, 512,512, interpolation=cv2.INTER_NEAREST)
        print(score_sigmoid.sum())
        # print(score_sigmoid.shape)

    subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])

subm = pd.DataFrame(subm)
subm.to_csv('results/PSPNet.csv', index=None, header=None, sep='\t')


plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(cv2.imread('data/test_a/' + subm[0].iloc[0]))
plt.subplot(122)
plt.imshow(rle_decode(subm[1].fillna('').iloc[0]), cmap='gray')
plt.savefig('results/PSPNet.png')
plt.show()
