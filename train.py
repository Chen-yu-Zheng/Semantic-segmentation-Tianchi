import configs
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import get_TianchiDataset
import torch.backends.cudnn as cudnn
from utils.figure import figureImgandMask


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0002')
    parser.add_argument('--cuda'  , action='store_false', help='enables cuda')
    parser.add_argument('--seed'  , type=int, default=configs.RANDOMSEED, help='input random seed')
    parser.add_argument('--save', default='./log', help='folder to store log files, model checkpoints')

    opt = parser.parse_args()
    print(opt)


    torch.manual_seed(opt.seed)
    if torch.cuda.is_available() and opt.cuda:
        device = torch.device('cuda')
        torch.cuda.manual_seed(opt.seed)
        cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        device = torch.device('cpu')
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        print("WARNING: You use a CPU device, so you should probably run with --cuda")


    dataset = get_TianchiDataset()
    train1_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train1_size
    train1_dataset, _ = random_split(dataset, [train1_size, test_size])
    train_size = int(0.8 * train1_size)
    val_size = train1_size - train_size
    train_dataset, val_dataset = random_split(train1_dataset, [train_size, val_size])
    print(len(train_dataset))
    j = 1
    for i in range(10):
        figureImgandMask(train_dataset[i][0], train_dataset[i][1], j)
        j += 1

if __name__ == '__main__':
    main()