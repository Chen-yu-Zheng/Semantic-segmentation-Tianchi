import pandas as pd
from utils.rle2img import *
import matplotlib.pyplot as plt

train_mask = pd.read_csv('./data/train_mask.csv', sep='\t', names=['name', 'mask'])
print(train_mask.shape)
print(train_mask.head())

# 读取第一张图，并将对于的rle解码为mask矩阵
img = plt.imread('data/train/'+ train_mask['name'].iloc[0])
mask = rle_decode(train_mask['mask'].iloc[0])

plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(mask)
plt.show()

print(rle_encode(mask) == train_mask['mask'].iloc[0])
# 结果为True