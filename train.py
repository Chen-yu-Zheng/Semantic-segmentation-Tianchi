from torch.utils import data
from torch.utils.data import dataset
from dataset import get_TianchiDataset
from utils.figure import *

if __name__ == '__main__':
    dataset = get_TianchiDataset()
    print(dataset[0][0], dataset[0][1])
    j = 1
    for i in range(10):
        figureImgandMask(dataset[i][0], dataset[i][1], j)
        j += 1
