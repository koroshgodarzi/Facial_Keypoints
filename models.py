## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net_GAP1(nn.Module):

    def __init__(self):
        super(Net_GAP1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x



class Net_GAP2(nn.Module):

    def __init__(self):
        super(Net_GAP2, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def _output_size(self, layers, input_size):
        """
        Compute output size after input going through convolution and pooling layers
        """
        output_size = input_size
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                output_size = (output_size + 2*layer.padding[0] - layer.kernel_size[0] + 1)
            elif isinstance(layer, nn.MaxPool2d):
                output_size = (output_size + 2*layer.padding - layer.kernel_size)/layer.stride + 1
        return output_size
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        output_size = self._output_size([self.conv1, self.pool, self.conv2, self.pool], 224)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*output_size*output_size, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def _output_size(self, layers, input_size):
        """
        Compute output size after input going through convolution and pooling layers
        """
        output_size = input_size
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                output_size = (output_size + 2*layer.padding[0] - layer.kernel_size[0] + 1)
            elif isinstance(layer, nn.MaxPool2d):
                output_size = (output_size + 2*layer.padding - layer.kernel_size)/layer.stride + 1
        return int(output_size)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x



class Net2(nn.Module):

    def __init__(self, conv_layer_num=3):
        super(Net2, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 =  nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 =  nn.Conv2d(128, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        output_size = self._output_size([self.conv1, self.pool, self.conv2, self.pool, self.conv3, self.pool, self.conv4, self.pool], 224)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*output_size*output_size, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def _output_size(self, layers, input_size):
        """
        Compute output size after input going through convolution and pooling layers
        """
        output_size = input_size
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                output_size = (output_size + 2*layer.padding[0] - layer.kernel_size[0] + 1)
            elif isinstance(layer, nn.MaxPool2d):
                output_size = (output_size + 2*layer.padding - layer.kernel_size)/layer.stride + 1
        return int(output_size)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    

class Net3(nn.Module):

    def __init__(self, conv_layer_num=3):
        super(Net3, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 =  nn.Conv2d(16, 64, 3, padding=1)
        self.conv4 =  nn.Conv2d(64, 32, 3, padding=1)
        self.conv5 =  nn.Conv2d(32, 128, 3, padding=1)
        self.conv6 =  nn.Conv2d(128, 64, 3, padding=1)
        self.conv7 =  nn.Conv2d(64, 512, 3, padding=1)
        self.conv8 =  nn.Conv2d(512, 256, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        output_size = self._output_size([self.conv1, self.pool, self.conv2, self.pool, self.conv3, self.pool, self.conv4, self.pool], 224)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*output_size*output_size, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def _output_size(self, layers, input_size):
        """
        Compute output size after input going through convolution and pooling layers
        """
        output_size = input_size
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                output_size = (output_size + 2*layer.padding[0] - layer.kernel_size[0] + 1)
            elif isinstance(layer, nn.MaxPool2d):
                output_size = (output_size + 2*layer.padding - layer.kernel_size)/layer.stride + 1
        return int(output_size)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool(F.relu(self.conv6(F.relu(self.conv5(x)))))
        x = self.pool(F.relu(self.conv8(F.relu(self.conv7(x)))))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x


class Net4(nn.Module):

    def __init__(self, conv_layer_num=3):
        super(Net4, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        output_size = self._output_size([self.conv1, self.pool, self.conv2, self.pool], 224)

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(32*output_size*output_size, 150)
        self.fc2 = nn.Linear(150, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def _output_size(self, layers, input_size):
        """
        Compute output size after input going through convolution and pooling layers
        """
        output_size = input_size
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                output_size = (output_size + 2*layer.padding[0] - layer.kernel_size[0] + 1)
            elif isinstance(layer, nn.MaxPool2d):
                output_size = (output_size + 2*layer.padding - layer.kernel_size)/layer.stride + 1
        return int(output_size)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x