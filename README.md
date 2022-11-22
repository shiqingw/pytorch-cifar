# ResNet Model with 5M Parameters on CIFAR-10

This GitHub repository contains the codebase of mini-project 1 of the Deep Learning course at NYU. In this project, we try to propose a modified residual network (ResNet) architecture with the highest possible test accuracy on the CIFAR-10 image classification dataset under the constraint that the model has no more than 5 million parameters.

## Prerequisites
The code is tested on
- Python 3.8.15
- PyTorch 1.13.0
- Torchvision 0.14.0

## Code Usage
1. Step 1: Define your test case in `test_cases.py`. 
    - You can specify the network structure (must be defined in the `models` folder), the optimizer (`SGD` or `Adam`), the learning rate (a float number), and the data augmentation method (`'None'`, `'Naive'`, or `'Auto'`). 
    - Here, we have defined three data augmentation methods for the project: `'None'` means that we use the original training data as it is (except for a normalization); `'Naive'` refers to the first data augmentation method (RandomCrop + RandomHorizontalFlip + Normalization); `'Auto'` refers to the second data augmentation method (RandomCrop + RandomHorizontalFlip + [AutoAugment](https://pytorch.org/vision/main/generated/torchvision.transforms.AutoAugment.html) + [Cutout](https://github.com/uoguelph-mlrg/Cutout) + Normalization); 

2. Step 2a: Start training with the following command. The trained weights will be saved to the folder `results/exp_<exp_num>`.
```
python main.py --exp_num <exp_num>
```

3. Step 2b: Alternatively, you can manually resume the training with the following command. Note that the folder `results/exp_<exp_num>` must exist and the previously trained weights must be named `ckpt.pth` inside the folder.
```
python main.py --exp_num <exp_num> --resume 
```

## Test Accuracy and Trained Weights
Below is a brief summary of the models with more than 96% test accuracy on the CIFAR-10 dataset. 

| Model       | Test Acc. (%)  | Number of Parameters (M) |Exp_num   |
| ----------- | -------------- | ----------- | ----------- |
| Net 2       | 96.00   | 4.98 |021       |
| Net 6       | 96.03   | 2.48 |022       |
| Net 8       | 96.47   | 2.78 |005       |
| Net 9       | 96.53   | 3.96 |024       |
| Net 10      | 96.22   | 3.89 |025       |
| Net 11      | 96.31   | 3.66 |026       |
| Net 12      | 96.37   | 4.84 |027       |
| Net 20      | 96.61   | 4.29 |040       |
| Net 21      | 96.76   | 4.94 |041       |

You can find the training log (`00output.out`), trained weights (`ckpt.pth`), training and testing losses/accuracies (`training_info.npy`, `acc.png`, `loss.png`) in the `results/exp_<exp_num>` folder. For example, for Net 21, these documents can be found at `results/exp_041`.

You will have to use the `pickle` package to load the `training_info.npy`. A helper function called `load_dict` can be found in `utils.py`.

