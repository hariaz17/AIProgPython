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
import PIL
import time
import random
import os
import argparse

from util import ld_checkpoint, ld_labels

def parse_args():
    parser = argparse.ArgumentParser(description = "Model Prediction")
    parser.add_argument('--filepath',dest='filepath',default=None)
    parser.add_argument('--gpu', action="store_true", default=True)
    parser.add_argument('--checkpoint', action="store", default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--cat_names', dest='cat_names', default='cat_to_name.json')
    return parser.parse_args()


def process_image(images):
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    basewidth = 256
    mean =np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    im = Image.open(images)
    wpercent = (basewidth / float(im.size[0]))
    hsize = int((float(im.size[1]) * float(wpercent)))
    img = im.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    out = img.crop((16,16,240,240))
    np_image = np.array(out)
    np_image = np.array(out)
    norm = (np_image - np.min(np_image))/np.ptp(np_image)

    res = (norm-mean)/std
    res = res.transpose((2,0,1))
    
    
    return res


def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    model.eval()
    model.to(device)
    
    #pre-process the image
    inp = process_image(image_path)
    
    #convert
    image = torch.from_numpy(np.array([inp])).float()
    image = Variable(image)
    
    image = image.to(device)
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data
    
    # obtain the topk
    # 0 -> probabilities
    # 1 -> index
    prob = torch.topk(probabilities, topk)[0].tolist()[0] # probabilities
    index = torch.topk(probabilities, topk)[1].tolist()[0] # index
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    # transfer index to label
    label = []
    for i in range(5):
        label.append(ind[index[i]])

    return prob, label


def main(): 
    args = parse_args()
    gpu = args.gpu
    model = ld_checkpoint(args.checkpoint)
    cat_to_name = ld_labels(args.cat_names)
    
    if args.filepath == None:
        img_num = random.randint(1, 102)
        image = random.choice(os.listdir('./flowers/test/' + str(img_num) + '/'))
        img_path = './flowers/test/' + str(img_num) + '/' + image
        prob, classes = predict(img_path, model, int(args.top_k), gpu)
        print('Image selected: ' + str(cat_to_name[str(img_num)]))
        print(prob)
        print(classes)
        print([cat_to_name[x] for x in classes])
    else:    
    
        im_path = args.filepath
        prob, classes = predict(im_path, model, int(args.top_k), gpu)
        print('File selected: ' + im_path)
        print(prob)
        print(classes)
        print([cat_to_name[x] for x in classes])

if __name__ == "__main__":
    main()
    
 





