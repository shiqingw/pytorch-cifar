'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugmentPolicy

from torchsummary import summary

import os
import argparse
import time
import platform

from models import *
from utils import progress_bar, format_time, save_dict, prepare_data, plot_loss_and_acc
from test_cases import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--exp_num', default=0, type=int, help='test case number')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    torch.manual_seed(0)
    exp_num = args.exp_num
    result_dir = './results/exp_{:03d}'.format(exp_num)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    test_case = Test_case(exp_num)
    net = test_case.net
    lr = test_case.lr
    optimizier_type = test_case.optimizier_type
    use_data_augmentation = test_case.use_data_augmentation

    if platform.system() == 'Darwin':
        if not torch.backends.mps.is_available():
            device = 'cpu'
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
        else:
            device = 'mps'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> device: ', device)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    trainloader, testloader = prepare_data(use_data_augmentation)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Evaluating model..')
    if device == "mps":
        summary(net, input_size=(3, 32, 32))
        net = net.to(device)
    else:
        net = net.to(device)
        summary(net, input_size=(3, 32, 32))

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(result_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(result_dir, "ckpt.pth"))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    if optimizier_type == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr,
                            momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif optimizier_type == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    else: raise ValueError('To be done')

    # Training
    def train(epoch, training_loss, training_acc):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        train_start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        train_stop_time = time.time()
        print('Epoch: %03d | Loss: %.3f | Acc: %.3f%% (%d/%d) | Training Time: %s'
                        % (epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total,
                         format_time(train_stop_time - train_start_time)))

        training_loss += [train_loss/(batch_idx+1)]
        training_acc += [correct/total]
        

    def test(epoch, testing_loss, testing_acc):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        test_start_time = time.time()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_stop_time = time.time()
        print('Epoch: %03d | Loss: %.3f | Acc: %.3f%% (%d/%d) | Testing Time: %s'
                        % (epoch, test_loss/(batch_idx+1), 100.*correct/total, correct, total,
                         format_time(test_stop_time - test_start_time)))
   
        testing_loss += [test_loss/(batch_idx+1)]
        testing_acc += [correct/total]

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(result_dir, "ckpt.pth"))
            best_acc = acc
        
    training_loss = []
    training_acc = []
    testing_loss = []
    testing_acc = []
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch, training_loss, training_acc)
        test(epoch, testing_loss, testing_acc)
        scheduler.step()
    stop_time = time.time()
    print("Total Time: %s" % format_time(stop_time - start_time))
    
    print("==> Saving training loss/acc and testing loss/acc...")
    training_info = {"training_loss": training_loss, "training_acc": training_acc,\
         "testing_loss": testing_loss, "testing_acc": testing_acc}
    save_dict(training_info, os.path.join(result_dir, "training_info.npy"))

    print("==> Drawing loss and acc...")
    loss_path = os.path.join(result_dir, "loss.png")
    acc_path = os.path.join(result_dir, "acc.png")
    plot_loss_and_acc(training_loss, training_acc, testing_loss, testing_acc, loss_path, acc_path)
    
    print("==> Process finished.")
