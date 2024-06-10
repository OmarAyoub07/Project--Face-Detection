## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I



## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


# class Net(nn.Module):

#     def __init__(self, image_width, image_height):
#         super(Net, self).__init__()
        
#         ## TODO: Define all the layers of this CNN, the only requirements are:
#         ## 1. This network takes in a square (same width and height), grayscale image as input
#         ## 2. It ends with a linear layer that represents the keypoints
#         ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
#         # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
#         # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
#         self.conv1 = nn.Conv2d(1, 32, 5)
#         # 1. Add aditional Conv layer 
#         self.conv2 = nn.Conv2d(32, 64, 5)
#         self.conv3 = nn.Conv2d(64, 128, 5)

        
#         ## Note that among the layers to add, consider including:
#         # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
#         # 2. Define max pooling layer
#         self.pool = nn.MaxPool2d(2,2)
        
#         # 3. Define fully connected layers
#         fc1_input_size = self._get_flatten_neurons_size(image_width, image_height)
#         self.fc1 = nn.Linear(fc1_input_size, 1000)
#         self.fc2 = nn.Linear(1000, 1000)
#         self.fc3 = nn.Linear(1000, 136)


#         # 4. Dropout layer to avoid overfitting
#         self.dropout = nn.Dropout(0.5)
        
        
#         # 5. Initialize weights
#         self._initialize_weights()
        
        
#     def forward(self, x):
#         ## TODO: Define the feedforward behavior of this model
#         ## x is the input image and, as an example, here you may choose to include a pool/conv step:
#         ## x = self.pool(F.relu(self.conv1(x)))
        
#         # 1. Apply convolutional layers with ReLU activation and pooling
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
        
#         # 2. Flatten the tensor for fully connected layers
#         x = x.view(x.size(0), -1)
        
#         # 3. Apply fully connected layers with ReLU activation and dropout
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
                
#         # a modified x, having gone through all the layers of your model, should be returned
#         return x
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.constant_(m.bias, 0)
    
#     def _get_flatten_neurons_size(self, width, height):
#         x = torch.randn(1, 1, width, height) 
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)
#         print("Flattened tensor size:", x.size())
#         return x.size(1)
    
    
    
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 68, 3)
        # 1. Add aditional Conv layers 
        self.conv2 = nn.Conv2d(68, 136, 3)
        self.conv3 = nn.Conv2d(136, 272, 3)
        self.conv4 = nn.Conv2d(272, 544, 3)
        self.conv5 = nn.Conv2d(544, 1088, 3)
        self.conv6 = nn.Conv2d(1088, 2176, 3)


        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # 2. Define max pooling layer
        self.pool = nn.MaxPool2d(2,2)
        
        # 3. Define fully connected layers
        self.fc1 = nn.Linear(2176, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 136)
        
        # 4. Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(0.4)
        

        
        # 5. Initialize weights
        self._initialize_weights()
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # 1. Apply convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        
        # 2. Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)
        
        # 3. Apply fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)


                
        # a modified x, having gone through all the layers of your model, should be returned
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    # def _get_flatten_neurons_size(self, width, height):
    #     x = torch.randn(1, 1, width, height) 
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = self.pool(F.relu(self.conv3(x)))
    #     x = self.pool(F.relu(self.conv4(x)))
    #     x = self.pool(F.relu(self.conv5(x)))
    #     x = self.pool(F.relu(self.conv6(x)))

    #     x = x.view(x.size(0), -1)
    #     print("Flattened tensor size:", x.size())
    #     return x.size(1)
    