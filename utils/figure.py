import matplotlib.pyplot as plt
import numpy as np

def figureImgandMask(img, mask, i):
    img, mask = img.numpy(), mask.numpy()
    img = np.transpose(img, axes= [1,2,0])
    plt.figure(i, figsize=(16,8))
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask, cmap= 'gray')
    plt.show()
