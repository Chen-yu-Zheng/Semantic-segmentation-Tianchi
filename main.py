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

from models.unet import UNet

from dataset import get_TianchiDataset

from utils.loss import DiceLoss
from utils.figure import figureImgandMask
from utils.logger import create_exp_dir


#train
def train(epoch, net, train_loader, criterion, opt, device, optimizer, log, writer):
    net.train()
    loss_epoch = 0

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    log.info("Epoch:%d, LR: %.6f", epoch, cur_lr)

    writer.add_scalar("lr", cur_lr, global_step= cur_step)


    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # FCN
        # output = net(images)[0].squeeze()
        output = net(images).squeeze()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_epoch = loss_epoch + loss.item() / opt.batchSize

        writer.add_scalar("loss/train", loss.item() / opt.batchSize, global_step=cur_step)

        if step % opt.log_frequency == 0 or step == len(train_loader) - 1:
            log.info(
                "Train: [{:d}/{:d}], Step: {:d}/{:d}, Loss: {:.4f} "
                .format(epoch + 1, opt.niter, step, len(train_loader) - 1, loss.item() / opt.batchSize))
        cur_step += 1

    loss_epoch = loss_epoch / len(train_loader)
    writer.add_scalar("loss/train_epoch", loss_epoch, global_step=epoch)
    log.info("Train: [{:d}/{}], Epoch_avg_loss: {:.4f}".format(epoch + 1, opt.niter, loss_epoch))
    return loss_epoch


#test network
def val(opt, epoch, net, val_loader, criterion, device, log, writer, cur_step):
    net.eval()
    loss_epoch = 0
    for step, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)

        # FCN
        # output = net(images)[0].squeeze()
        output = net(images).squeeze()
        loss = criterion(output, labels)

        loss_epoch = loss_epoch + loss.item() / opt.batchSize

        if step % opt.log_frequency == 0 or step == len(val_loader) - 1:
                log.info(
                    "Valid: [{:d}/{:d}], Step {:d}/{:d} Loss {:.4f}".format(
                        epoch + 1, opt.niter, step, len(val_loader) - 1, loss.item() / opt.batchSize))

    loss_epoch = loss_epoch / len(val_loader)
    writer.add_scalar("loss/val", loss_epoch, global_step=cur_step)
    log.info("Valid: [{:d}/{:d}], Loss {:.4f}".format(epoch + 1, opt.niter, loss_epoch))

    return loss_epoch


def test(opt, net, test_loader, criterion, device, log, writer):
    net.eval()
    loss_epoch = 0
    cur_step = 0
    for step, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # FCN
        #output = net(images)[0].squeeze()
        output = net(images).squeeze()
        loss = criterion(output, labels)

        loss_epoch = loss_epoch + loss.item() / opt.batchSize

        if step % opt.log_frequency == 0 or step == len(test_loader) - 1:
                log.info(
                    "Test: Step {:d}/{:d} Loss {:.4f}".format(
                      step, len(test_loader) - 1, loss.item() / opt.batchSize))
        cur_step += 1

    loss_epoch = loss_epoch / len(test_loader)
    writer.add_scalar("loss/test", loss_epoch, global_step=cur_step)
    log.info("Test: Loss {:.4f}".format(loss_epoch))

    return loss_epoch

def loss_fn(y_pred, y_true):
    bce = nn.BCEWithLogitsLoss()(y_pred, y_true)
    dice = DiceLoss()(y_pred, y_true)
    return 0.8*bce+ 0.2*dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tianchi', help='cifar10 | cifar100')
    parser.add_argument('--dataroot', default=configs.ROOT, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=configs.WORKERS)
    parser.add_argument('--batchSize', type=int, default=configs.BATCHSIZE, help='input batch size')
    parser.add_argument('--niter', type=int, default=configs.NITER, help='number of epochs to train for')
    parser.add_argument('--log_frequency', type=int, default=configs.LOG_FREQUENCY, help='number of steps to print log in an epoch')
    parser.add_argument('--lr', type=float, default=configs.LR, help='learning rate, default=0.0002')
    parser.add_argument('--cuda'  , action='store_false', help='enables cuda')
    parser.add_argument('--seed'  , type=int, default=configs.RANDOMSEED, help='input random seed')
    #parser.add_argument('--save', default='exps/' + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()), help='folder to store log files, model checkpoints')
    parser.add_argument('--save', default='exp', help='folder to store log files, model checkpoints')
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
    log = logging.getLogger()
    log.addHandler(fh)
    log.info(opt)

    # 配置文件
    with open(os.path.join("exps", opt.exp_name, "config.yml"), "w") as f:
        yaml.dump(opt, f)

    # Tensorboard文件
    writer = SummaryWriter("exps/%s/runs/%s" %
                        (opt.exp_name, time.strftime("%m-%d-%H-%M", time.localtime())))

    # 文件备份
    create_exp_dir(os.path.join("exps", opt.exp_name),
                scripts_to_save=glob.glob('*.py'))

    #checkpoints
    if not os.path.exists(os.path.join("exps", opt.exp_name, 'checkpoints')):
            os.mkdir(os.path.join("exps", opt.exp_name, 'checkpoints'))

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
    train1_dataset, test_dataset = random_split(dataset, [train1_size, test_size])
    train_size = int(0.8 * train1_size)
    val_size = train1_size - train_size
    train_dataset, val_dataset = random_split(train1_dataset, [train_size, val_size])

    #show figures
    # j = 1
    # for i in range(10):
    #     figureImgandMask(train_dataset[i][0], train_dataset[i][1], j)
    #     j += 1

    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize,\
                                                shuffle=True, num_workers=int(opt.workers))

    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batchSize,\
                                                shuffle=False, num_workers=int(opt.workers))
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batchSize,\
                                                shuffle=False, num_workers=int(opt.workers)) 
    
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    net.to(device)
    print(net)

    # x = torch.rand((1,3,configs.IMAGE_SIZE, configs.IMAGE_SIZE)).to(device)
    # out = net(x)[0]
    # print(out)
    # print(out.shape)
    # print(train_dataset[0][1].shape)
    # sys.exit()

    criterion = loss_fn

    for epoch in range(opt.niter):
        if epoch > 90:
            optimizer = optim.SGD(net.parameters(), lr=configs.LR / 25, momentum=0.9, weight_decay=0.0005)
        elif epoch > 60:
            optimizer = optim.SGD(net.parameters(), lr=configs.LR / 5, momentum=0.9, weight_decay=0.0005)
        else:
            optimizer = optim.SGD(net.parameters(), lr=configs.LR, momentum=0.9, weight_decay=0.0005)

        #train
        train_loss = train(epoch, net, train_loader, criterion, opt, device, optimizer, log, writer)

        #valid
        cur_step = (epoch + 1) * len(train_loader)
        val_loss = val(opt, epoch, net, val_loader, criterion, device, log, writer, cur_step)

        if epoch % 10 == 0 or epoch == opt.niter - 1:
            # do checkpointing
            torch.save(net.state_dict(), "exps/%s/checkpoints/epoch_%d.pth" %
                        (opt.exp_name, epoch))
        
    #test
    test_loss = test(opt, net, test_loader, criterion, device, log, writer)

if __name__ == '__main__':
    main()
