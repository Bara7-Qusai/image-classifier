import os
import numpy as np
import argparse
import json
from time import time

import torch
from torch import nn , optim
import torch.nn.functional as F
from torch.utils import data

import torchvision
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

from collections import OrderedDict
from PIL import Image


def get_input_args():

    parser =  argparse.ArgumentParser(description="Arguments for the model training script")

    parser.add_argument('--dir',type=str, help='directory fof the images',default='/flowers')
    parser.add_argument('--arch',type=str, help='choose training model architecture',default='vgg16', choices=['vgg16', 'resnet152'])
    parser.add_argument('--save_dir',type=str, help='directory for model checkpoint',default='checkpoint.pth')
    parser.add_argument ('--learning_rate', help = 'learning rate', type = float, default = 0.001)
    parser.add_argument ('--hidden_units', help = 'hidden units ', type = int, default = 1000)
    parser.add_argument ('--epochs', help = 'number of epochs', type = int, default = 5)
    parser.add_argument ('--gpu',dest='gpu', help = "specify if you want to choose gpu or not", action='store_true')

    return parser.parse_args()

def train_transform(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])
    train_dataset = ImageFolder(train_dir,transform=train_transforms)
    return train_dataset

def valid_transform(valid_dir):
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])
    valid_dataset = ImageFolder(valid_dir,transform=validation_transforms)
    return valid_dataset
def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                     ])
    test_dataset =  ImageFolder(test_dir,test_transforms)
    return test_dataset

def train_loader(train_dataset,batch_size=64,shuffle=True):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader
def valid_loader(valid_dataset,batch_size=32):
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size)
    return valid_dataloader
def test_loader(test_dataset,batch_size=32):
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)
    return test_dataloader
def create_model(arch, hidden_units, gpu, learning_rate):
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_units = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_units, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, int(hidden_units / 2))),
            ('relu', nn.ReLU()),
            ('Dropout', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(int(hidden_units / 2), 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    elif arch == "resnet152":
        model = models.resnet152(pretrained=True)
        input_units = model.fc.in_features
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_units, hidden_units)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, int(hidden_units / 2))),
            ('relu', nn.ReLU()),
            ('Dropout', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(int(hidden_units / 2), 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
     
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Training on GPU...")
    else:
        device = torch.device("cpu")
        print("GPU not available, training on CPU...")
    
    criterion = nn.NLLLoss()
    model.to(device)
    
    return model, optimizer, device

def training(epochs,model,optimizer,train_dataloader,valid_dataloader,device):
    criterion = nn.NLLLoss()
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 25
    
    start_time = time()
    print('training started....')
    
    for epoch in range(epochs):
        for images, labels in train_dataloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                              f"Train loss: {running_loss/print_every:.3f}.. "
                              f"Validation loss: {test_loss/len(valid_dataloader):.3f}.. "
                              f"Validation accuracy: {accuracy/len(valid_dataloader):.3f}")
                running_loss = 0
                model.train()
                
    end_time = time()    
    tot_time = end_time - start_time
    print("\n** Total time for training is: ",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    

    
def save_checkpoint(model,arch,optimizer,save_dir,lr,epochs,train_dataset):
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'arch':arch,
                  'output_units': 102,
                  'learning_rate': lr,
                  'classifier': model.classifier,
                  'epochs': epochs,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer' : optimizer.state_dict()
                 }
    if save_dir=='checkpoint.pth':
        torch.save(checkpoint, 'checkpoint.pth')
    else:
        torch.save(checkpoint, save_dir+'/checkpoint.pth')
        
 
def main():
    current_dir_path= os.getcwd()
    args = get_input_args()
    data_dir = args.dir
    train_dir = current_dir_path+data_dir + '/train'
    valid_dir = current_dir_path+data_dir + '/valid'
    test_dir = current_dir_path+data_dir + '/test'

    train_dataset = train_transform(train_dir=train_dir)
    valid_dataset = valid_transform(valid_dir=valid_dir)
    test_dataset = test_transform(test_dir=test_dir)

    train_dataloader = train_loader(train_dataset=train_dataset,batch_size=64,shuffle=True)
    valid_dataloader = valid_loader(valid_dataset=valid_dataset,batch_size=32)
    test_dataloader = test_loader(test_dataset=test_dataset,batch_size=32)

    arch = args.arch
    gpu = args.gpu
    hidden_units = args.hidden_units
    epochs = args.epochs
    lr = args.learning_rate
    save_dir = args.save_dir
 
    model , optimizer, device= create_model(arch,hidden_units,gpu,lr)
    print(model)
    print(device)


    training(epochs,model,optimizer,train_dataloader,valid_dataloader,device)


    save_checkpoint(model,arch,optimizer,save_dir,lr,epochs,train_dataset)
    
if __name__=="__main__":
    main()

