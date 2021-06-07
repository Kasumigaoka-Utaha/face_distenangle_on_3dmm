import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Distangler(nn.Module):
    def __init__(self,input_channel=79,output_channel_1=128,output_channel_2=64):
        super(Distangler,self).__init__()
        # shared fully connect layers 
        self.relu = nn.ReLU(inplace = False)
        self.fc1 = nn.Linear(input_channel,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,512)
        # branch 1 to feature parameters for other features
        self.branch1 = nn.Linear(512,output_channel_1)
        # branch 2 to feature parameters for mouths
        self.branch2 = nn.Linear(512,output_channel_2)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        out1 = self.branch1(x)
        out1 = F.normalize(out1,dim=1)
        out2 = self.branch2(x)
        out2 = F.normalize(out2,dim=1)
        return out1,out2

class Concatenater(nn.Module):
    def __init__(self,input_channel_1=128,input_channel_2=64,output_channel=79):
        super(Concatenater,self).__init__()
        self.channel = input_channel_1 + input_channel_2
        self.relu = nn.ReLU(inplace = False)
        self.fc1 = nn.Linear(self.channel,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,output_channel)
    def forward(self,x1,x2):
        # concanate the two vectors
        x = torch.cat((x1,x2),dim=1)
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        out = F.normalize(out,dim=1)
        return out

#class MouthNet(nn.Module):
    #def __init__(self,input_channel=64,output_channel=64):
        #super(MouthNet,self).__init__()
        

#class ExpressionNet(nn.Module):
    #def __init__(self,input_channel=79,output_channel=79):
       # super(ExpressionNet,self).__init__()
       # self.distangle = Distangler(input_channel)
       # self.concatenate = Concatenater(output_channel=output_channel)
        #self.mouth = MouthNet()
        #self.gan = GAN_net()
    #def forward(self,x):
       # out1,out2 = self.distangle(x)
        #out1 = self.mouth(out1)
        #out2 = self.gan(out2)
       # out = self.concatenate(out1,out2)
       # out += x
       # return out
