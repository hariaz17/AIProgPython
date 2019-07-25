import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json

import time
import random
import os
import argparse

def sv_checkpoint(model, optimizer, args, classifier):
    checkpoint = {'arch': args.arch,
                 'model' : model,
                 'epochs': args.epochs,
                 'learning_rate': args.learning_rate,
                 'hidden_units':args.hidden_units,
                 'classifier': classifier,
                 'state_dict': model.state_dict(),
                 'optimizer':optimizer.state_dict(),
                 'batch_size':64,
                 'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')


def ld_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


    
def ld_labels(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

