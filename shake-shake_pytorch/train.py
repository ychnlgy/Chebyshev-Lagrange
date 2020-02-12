# -*- coding: utf-8 -*-

import argparse, sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from tqdm import tqdm

import utils
from datasets import load_dataset
from models import ShakeResNet, ShakeResNeXt

def main(args):
    train_loader, test_loader, channels = load_dataset(args.label, args.batch_size, args.mnist)
    model = ShakeResNet(args.depth, args.w_base, args.label, args.use_shakeshake, args.act_type, channels)
    
    params = sum(torch.numel(p) for p in model.parameters() if p.requires_grad)
    print("Parameters: %d" % params)
    
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benckmark = True

    opt = optim.SGD(model.parameters(),
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov)
    loss_func = nn.CrossEntropyLoss().cuda()

    headers = ["Epoch", "LearningRate", "TrainLoss", "TestLoss", "TrainAcc.", "TestAcc."]
    logger = utils.Logger(args.checkpoint, headers)
    for e in range(args.epochs):
        lr = utils.cosine_lr(opt, args.lr, e, args.epochs)
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        bar = tqdm(total=len(train_loader), leave=False)
        for x, t in train_loader:
            x, t = Variable(x.cuda()), Variable(t.cuda())
            y = model(x)
            loss = loss_func(y, t)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_acc += utils.accuracy(y, t).item()
            train_loss += loss.item() * t.size(0)
            train_n += t.size(0)
            bar.set_description("Loss: {:.4f}, Accuracy: {:.2f}".format(
                train_loss / train_n, train_acc / train_n * 100), refresh=True)
            bar.update()
        bar.close()

        model.eval()
        test_loss, test_acc, test_n = 0, 0, 0
        for x, t in tqdm(test_loader, total=len(test_loader), leave=False):
            with torch.no_grad():
                x, t = Variable(x.cuda()), Variable(t.cuda())
                y = model(x)
                loss = loss_func(y, t)
                test_loss += loss.item() * t.size(0)
                test_acc += utils.accuracy(y, t).item()
                test_n += t.size(0)
        logger.write(e+1, lr, train_loss / train_n, test_loss / test_n,
                     train_acc / train_n * 100, test_acc / test_n * 100)
    
    final_score = test_acc/test_n*100
    
    with open("history.txt", "a") as f:
        command = " ".join(sys.argv)
        f.write("%s\nParameters: %d\nFinal test accuracy: %.5f\n" % (command, params, final_score))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint")
    parser.add_argument("--mnist", type=int)
    
    # For Networks
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--w_base", type=int, default=32)
    parser.add_argument("--cardinary", type=int, default=4)
    parser.add_argument("--use_shakeshake", type=int)
    parser.add_argument("--act_type", type=str)
    
    # For Training
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--nesterov", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    main(args)
