import torch
import cv2
from torch.utils.data import dataset
from utils.rle2img import *
#from utils.figure import *
import torchvision.transforms as T
from torch.utils import data as D
import configs

class TianChiDataset(D.Dataset):
    def __init__(self, paths, rles, test_mode=False):
        self.paths = paths
        self.rles = rles
        self.test_mode = test_mode
        
        self.len = len(paths)
        self.transform_img = T.Compose([
            T.ToPILImage(),
            T.Resize(configs.IMAGE_SIZE),
            T.ToTensor(),
            # T.Normalize([0.625, 0.448, 0.688],
            #             [0.131, 0.177, 0.101]),
        ])

        self.transform_mask = torch.tensor
        
    # get data operation
    def __getitem__(self, index):
        img = cv2.imread(self.paths[index])
        if not self.test_mode:
            mask = rle_decode(self.rles[index])
            #though mask = '', it can be decoded to 512*512 matrix
            img = self.transform_img(img)
            mask = self.transform_mask(mask)
            return img, mask
        else:
            return self.as_tensor(img), ''        
    
    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    

def get_TianchiDataset():
    train_mask = pd.read_csv(configs.ROOT, sep='\t', names=['name', 'mask'])
    train_mask['name'] = train_mask['name'].apply(lambda x: 'data/train/' + x)
    train_mask['mask'] = train_mask['mask'].fillna('')

    dataset = TianChiDataset(
        train_mask['name'].values,
        train_mask['mask'].values,
        False
    )
    return dataset


'''
def get_mean_std():
    dataset = get_TianchiDataset()
    mean = torch.zeros(3)
    std = torch.zeros(3)

    
    print(means)
    print(stds)
'''


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for X, _ in train_data:
        for d in range(3):
            mean[d] += X[d, :, :].mean()
            std[d] += (X[d, :, :].pow(2)).sum()
    mean.div_(len(train_data))
    std -= len(train_data) * configs.IMAGE_SIZE * configs.IMAGE_SIZE * mean.pow(2)
    std.div_(len(train_data) * configs.IMAGE_SIZE * configs.IMAGE_SIZE - 1)
    std.pow_(0.5)
    return list(mean.numpy()), list(std.numpy())

def main():
    mean, std = getStat(get_TianchiDataset())
    print(mean)
    print(std)


if __name__ == '__main__':
    main()
'''
[0.40464398, 0.42690134, 0.39236653]
[0.20213476, 0.18353915, 0.17596193]
'''