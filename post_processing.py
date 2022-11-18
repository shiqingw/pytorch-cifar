from utils import *
import os
import torch

for exp_num in range(13,21):

    result_dir = "./results/exp_{:03d}".format(exp_num)

    # plotting
    # training_info = load_dict(os.path.join(result_dir, "training_info.npy"))
    # training_loss = training_info["training_loss"]
    # training_acc = training_info["training_acc"]
    # testing_loss = training_info["testing_loss"]
    # testing_acc = training_info["testing_acc"]

    # loss_path = os.path.join(result_dir, "loss.png")
    # acc_path = os.path.join(result_dir, "acc.png")
    # plot_loss_and_acc(training_loss, training_acc, testing_loss, testing_acc, loss_path, acc_path)

    # print highest acc
    checkpoint = torch.load(os.path.join(result_dir, "ckpt.pth"), map_location=torch.device('cpu'))    
    best_acc = checkpoint['acc']
    print("Test case: {:03d}, highest acc: {:.3f}".format(exp_num, best_acc))