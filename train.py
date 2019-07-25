import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

import matplotlib.pyplot as plt 
from PIL import Image
import time
import random
import os
import argparse

from util import sv_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description = "Model Training")
    parser.add_argument('--data_dir',action='store')
    parser.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'vgg13'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='10')
    parser.add_argument('--gpu', action="store_true", default=True)
    return parser.parse_args()

def train(model,criterion,optimizer,dataloaders,epochs,gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model.to(device);

    steps = 0
    running_loss = 0
    accuracy = 0

    start = time.time()
    print('Training Started')

    for epoch in range(epochs):
        train_mode = 0
        valid_mode = 1
    
        for mode in [train_mode, valid_mode]:   
            if mode == train_mode:
                model.train()
            else:
                model.eval()
            
            pass_count = 0
            for data in dataloaders[mode]:
                pass_count += 1
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
            
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)
                
                if mode == train_mode:
                    loss.backward()
                    optimizer.step()
                   
                running_loss += loss.item()
            
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
            
            if mode == train_mode:
                print("\nEpoch: {}/{} ".format(epoch+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                      "Accuracy: {:.4f}".format(accuracy))
            
            running_loss = 0
    time_elapsed = time.time() - start
    print("\nTotal time: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60))          

    
    
def main():
    args = parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    Train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
                                           
    Test_transform = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])       

# TODO: Load the datasets with ImageFolder
    Train_image_datasets = datasets.ImageFolder(train_dir,transform = Train_transform)
    Test_image_datasets = datasets.ImageFolder(test_dir,transform = Test_transform)
    Valid_image_datasets = datasets.ImageFolder(valid_dir,transform = Test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
    Trainloader = torch.utils.data.DataLoader(Train_image_datasets,batch_size=64,shuffle=True)
    Testloader = torch.utils.data.DataLoader(Test_image_datasets,batch_size=64)
    Validloader = torch.utils.data.DataLoader(Valid_image_datasets,batch_size=64)

    image_datasets = [Train_image_datasets, Valid_image_datasets, Test_image_datasets]
    dataloaders = [Trainloader, Validloader, Testloader]
    
    #choose the model arch and assign it
    model = getattr(models, args.arch)(pretrained = True)
    
    
    #sort out the gpu bit
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for param in model.parameters():
        param.requires_grad = False

    feats = model.classifier[0].in_features
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(feats, 4096)),
                            ('relu', nn.ReLU()),
                            ('Drop',nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(4096,102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    gpu = args.gpu
    
    
    class_index = image_datasets[0].class_to_idx
    train(model,criterion,optimizer,dataloaders,epochs,gpu)
    model.class_to_idx = class_index
    
    sv_checkpoint(model,optimizer,args,classifier)
    
if __name__ == "__main__":
    main()
    
