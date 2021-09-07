import configs
import sys


import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torchvision.transforms as T

from models.fcn import FCN8s

from utils.rle2img import rle_encode, rle_decode

import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

subm = []

net = FCN8s(nclass= 1)
net.load_state_dict(torch.load("exps/exp_2021Y_09M_03D_11H_01M/checkpoints/epoch_99.pth"))
net.to(device)
net.eval()

test_mask = pd.read_csv('data/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
test_mask['name'] = test_mask['name'].apply(lambda x: 'data/test_a/' + x)

trans= T.Compose([
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
        score = net(image)[0]
        score_sigmoid = score.sigmoid().cpu().numpy()
        score_sigmoid = (score_sigmoid >= 0.5).astype(np.uint8)

    subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])

subm = pd.DataFrame(subm)
subm.to_csv('data/submission.csv', index=None, header=None, sep='\t')


plt.figure(figsize=(16,8))
plt.subplot(121)
plt.imshow(cv2.imread('data/test_a/' + subm[0].iloc[0]))
plt.subplot(122)
plt.imshow(rle_decode(subm[1].fillna('').iloc[0]), cmap='gray')
plt.show()