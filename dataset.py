import torch
import cv2
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



def get_mean_std():
    dataset = get_TianchiDataset()
    means = []
    stds = []
    sum = 0
    std = 0
    for channel in range(3):
        for i in range(len(dataset)):
            img = dataset[i][0][channel]
            sum += img.sum().item()
        mean = sum / (len(dataset)*configs.IMAGE_SIZE*configs.IMAGE_SIZE)
        means.append(mean)

        for i in range(len(dataset)):
            img = dataset[i][0][channel]
            std += ((img - mean) * (img - mean)).sum().item()
        std = std / ((len(dataset)*configs.IMAGE_SIZE*configs.IMAGE_SIZE) - 1)
        std = std ** 0.5
        stds.append(std)
    
    print(means)
    print(stds)

def main():
    get_mean_std()

if __name__ == '__main__':
    main()
