#Filename:	main.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sab 05 Des 2020 03:06:09  WIB

import argparse
import copy
from util.loader import *
from util.mnist import *
from util.train import *
import torch.optim as optim

def parse_param():
    """
    parse the parameters
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type = str, default = "./util/files.txt", help = "the training directory")
    parser.add_argument("-dtype", type = str, default = "countertriplet", help = "the types: query, counterquery, counertriplet")
    parser.add_argument("-lamb", type = float, default = 0.1, help = "the lambda factor")
    parser.add_argument("-margin", type = float, default = 0.1, help = "the margin factor")
    parser.add_argument("-pernum", type = int, default = 5, help = "the perturb num")
    parser.add_argument("-batch", type = int, default = 128, help = "batch size")
    parser.add_argument("-epoch", type = int, default = 100, help = "epoch number")
    parser.add_argument("-lr", type = float, default = 1e-4, help = "learning rate")
    parser.add_argument("-gpu", type = bool, default = True, help = "use gpu?")
    args = parser.parse_args()
    return args

def print_param(args):

    print("dataset:\t", args.dataset)
    print("dtype:\t", args.dtype)
    print("lambda:\t", args.lamb)
    print("margin:\t", args.margin)
    print("perturbation number:\t", args.pernum)
    print("batch size\t", args.batch)
    print("epoch\t", args.epoch)
    print("learning rate\t", args.lr)

def run(model, train_loader, test_loader, optimizer, loss_func, epoch, dtype):

    best_acc = 0
    best_epoch = 0
    best_model = model
    best_iters = 0

    if dtype == "query":
        train = train_query
    elif dtype == "counterquery":
        train = train_counterquery
    elif dtype == "countertriplet":
        train = train_countertriplet
    else:
        return

    for i in range(epoch):

        model, train_loss, train_acc, time_elapsed = train(model, train_loader, loss_func, optimizer)
        print("Training set: Epoch {}, Loss {}, Accuracy {}, Time Elapsed {}".format(i, train_loss / len(train_loader.dataset), train_acc / len(train_loader.dataset), time_elapsed))
        test_loss, test_acc, time_elapsed = test(model, test_loader, loss_func, True)
        print("Test set: Epoch {}, Loss {}, Accuracy {}, Time Elapsed {}".format(i, test_loss / len(test_loader.dataset), test_acc / len(test_loader.dataset), time_elapsed))

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = copy.deepcopy(model)
            best_iters = i

        train_scheduler.step()
    
    return best_model, best_acc, best_iters

if __name__ == "__main__":
    args = parse_param()
    print_param(args)
    
    file_list = load_file_list(args.dataset)
    train_loader = train_loader(file_list, args.batch, args.dtype)
    test_loader = test_loader(file_list, args.batch)

    model = LeNet()
    optimizer = optim.Adam(model.parameters(), args.lr)

    if args.gpu:
        device = torch.device("cuda", 0)
        model = model.to(device)

    if args.dtype == "query":
        loss_func = nn.CrossEntropyLoss()
    elif args.dtype == "counterquery":
        loss_func = CELoss()
    elif args.dtype == "countertriplet":
        loss_func = JointLoss(args.margin, args.lamb)

    if args.gpu:
        loss_func = loss_func.to(device)
    
    best_model, best_acc, best_iters = run(model, train_loader, test_loader, optimizer, loss_func, args.epoch, args.dtype)

