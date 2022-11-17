from models import *
from torchsummary import summary

net = ResNet_four_layer_1()
summary(net, input_size=(3, 32, 32))