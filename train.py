import configs
import argparse
import yaml
import logging
import time
import os
import sys
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from models.fcn import FCN8s

from dataset import get_TianchiDataset

from utils.loss import DiceLoss
from utils.figure import figureImgandMask
from utils.logger import create_exp_dir


#train
def train(epoch, net, train_loader, criterion, opt, device, optimizer, writer):
    net.train()
    score_epoch = 0.0
    loss_epoch = 0

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logging.info("Epoch:%d, LR: %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step= cur_step)

    step_loss = 0

    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = net(images)[0]
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_epoch = loss_epoch + loss.item()
        step_loss = step_loss + loss.item()
        #score_epoch = score_epoch + compute_score(output.data, labels.data)

        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)

        if step % opt.log_frequency == 0 or step == len(train_loader) - 1:
            logging.info(
                "Train: [{:3d}/{}], Step, {:03d}/{:03d} Loss {:3f} "
                .format(epoch + 1, opt.epochs, step, len(train_loader) - 1, loss))
        cur_step += 1

    loss_epoch = loss_epoch / 50000
    score_epoch = score_epoch / 50000

    print('[%d/%d][%d] train_loss: %.4f err: %.4f'
         % (epoch, opt.niter, len(train_loader), loss_epoch, score_epoch))
    return loss_epoch, score_epoch

#test network
def val(net, val_loader, criterion, device):
    net.eval()
    score_epoch = 0.0
    loss_epoch = 0
    for _, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        loss = criterion(output, labels)

        loss_epoch = loss_epoch + loss.item()
        # score_epoch = score_epoch + compute_score(output.data, labels.data)

    loss_epoch = loss_epoch / 10000
    score_epoch = score_epoch / 10000
    print('Test error: %.4f' % (score_epoch))
    return loss_epoch, score_epoch


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100')
    parser.add_argument('--dataroot', default=configs.ROOT, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=configs.WORKERS)
    parser.add_argument('--batchSize', type=int, default=configs.BATCHSIZE, help='input batch size')
    parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--log_frequency', type=int, default=200, help='number of steps to print log in an epoch')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate, default=0.0002')
    parser.add_argument('--cuda'  , action='store_false', help='enables cuda')
    parser.add_argument('--seed'  , type=int, default=configs.RANDOMSEED, help='input random seed')
    #parser.add_argument('--save', default='exps/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), help='folder to store log files, model checkpoints')
    parser.add_argument('--save', default='exp1', help='folder to store log files, model checkpoints')
    opt = parser.parse_args()
    print(opt)

    # default `log_dir` is "runs" - we'll be more specific here
    opt.exp_name = opt.save + "_" + time.strftime("%YY_%mM_%dD_%HH_%MM", time.localtime())

    # 新建实验文件夹
    if not os.path.exists(os.path.join("exps", opt.exp_name)):
        os.makedirs(os.path.join("exps", opt.exp_name))


    # 日志文件
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m/%d %I:%M:%S %p")

    fh = logging.FileHandler(os.path.join("exps", opt.exp_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(opt)

    # 配置文件
    with open(os.path.join("exps", opt.exp_name, "config.yml"), "w") as f:
        yaml.dump(opt, f)

    # Tensorboard文件
    writer = SummaryWriter("exps/%s/runs/%s-%05d" %
                        (opt.exp_name, time.strftime("%m-%d-%H-%M", time.localtime())))

    # 文件备份
    create_exp_dir(os.path.join("exps", opt.exp_name),
                scripts_to_save=glob.glob('*.py'))

    #确定device为gpu/cpu
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

    #导入数据集，划分数据集
    dataset = get_TianchiDataset()
    train1_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train1_size
    train1_dataset, _ = random_split(dataset, [train1_size, test_size])
    train_size = int(0.8 * train1_size)
    val_size = train1_size - train_size
    train_dataset, val_dataset = random_split(train1_dataset, [train_size, val_size])

    # j = 1
    # for i in range(10):
    #     figureImgandMask(train_dataset[i][0], train_dataset[i][1], j)
    #     j += 1

    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize,\
                                                shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batchSize,\
                                                shuffle=False, num_workers=int(opt.workers))  
    
    net = FCN8s(nclass= 2)
    net.to(device)
    print(net)

    criterion = DiceLoss()

    for epoch in range(1,opt.niter+1):
        if epoch > 120:
            optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
        elif epoch > 80:
            optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
        else:
            optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

        
        train_loss, train_error = train(epoch, net, train_loader, criterion, opt, optimizer, writer)
        test_loss, test_error = val(net, val_loader, criterion)

        if epoch % 10 == 0:
            # do checkpointing
            torch.save(net.state_dict(), "exps/%s/checkpoints/epoch_%d.pth" %
                        (opt.exp_name, epoch))

if __name__ == '__main__':
    main()
