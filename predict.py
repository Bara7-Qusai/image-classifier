import os
import numpy as np
import json
import argparse

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data

import torchvision
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

from collections import OrderedDictش
from PIL import Image


def get_input_args():
    parser = argparse.ArgumentParser(description="Arguments for the model prediction script")

    parser.add_argument('--checkpoint_path', help="checkpoint file path", default='checkpoint.pth')
    parser.add_argument('--image_path', help="This is an image file that you want to classify", default='flowers/test/49/image_06213.jpg')
    parser.add_argument('--category_names', help="json file to categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="number of top k likely classes to predict, default is 5", default=5, type=int)
    parser.add_argument('--gpu', help="Input device you want to use (gpu or cpu)", type=str, default='cpu', choices=['gpu', 'cpu'])

    return parser.parse_args()


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    
    if checkpoint['arch'] == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
    else:
        model = models.resnet152(pretrained=True)
        model.fc = checkpoint['classifier']  
     
    for param in model.parameters():
        param.requires_grad = False

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    image_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.open(image_path)
    transformed_image = image_transforms(pil_image)
    return transformed_image


def predict(image_path, model, topk, category_names, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f,strict=False)
    
    image = process_image(image_path)
    image = image.unsqueeze(0).float().to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_ps, top_classes = ps.topk(topk, dim=1)
        idx_to_class = {value: cat_to_name[key] for key, value in model.class_to_idx.items()}
        predicted_flowers = [idx_to_class[i] for i in top_classes[0].tolist()]
        predicted_probabilities = top_ps[0].tolist()
        classes = top_classes[0].tolist()
        
        print(f'The top {topk} predicted probabilities are {predicted_probabilities} for the classes {classes} with their associated flower names {predicted_flowers}')
        return predicted_probabilities, classes, predicted_flowers

if __name__ == "__main__":
 
    args = get_input_args()

    checkpoint_path = args.checkpoint_path
    image_path = args.image_path
    category_names = args.category_names
    top_k = args.top_k
    gpu = args.gpu

    device = torch.device("cuda" if gpu == 'gpu' and torch.cuda.is_available() else "cpu")

    loaded_model = load_checkpoint(checkpoint_path)

    print(predict(image_path, loaded_model, top_k, category_names, device))
