from models import *
from torchsummary import summary

net = ResNet_three_layer_8()
# net = ResNet_four_layer_1()
summary(net, input_size=(3, 32, 32))