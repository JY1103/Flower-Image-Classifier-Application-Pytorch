import argparse
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from DataPreprocess import *

#get defined parameters
def get_input_para():
    parser = argparse.ArgumentParser(
        description='Image Classifier Training Parameters',
    )

    parser.add_argument('data_dir', action="store")
    parser.add_argument('--save_dir', action="store")
    parser.add_argument('--arch', action="store",default="vgg16")
    parser.add_argument('--learning_rate', action="store",default=0.001,type=float)
    parser.add_argument('--hidden_units', action="store",default=4096,type=int)
    parser.add_argument('--epochs', action="store",default=4,type=int)
    parser.add_argument('--gpu', action="store_true", default=True)

    return parser.parse_args()

#get predefined model
def get_pre_model(arch,learning_rate,hidden_units,epochs,gpu):
    model=getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    from collections import OrderedDict
    classifier=nn.Sequential(OrderedDict([
        ('fc1',nn.Linear(25088,hidden_units)),
        ('relu1',nn.ReLU()),
        ('dropout1',nn.Dropout(0.5)),
        ('fc2',nn.Linear(hidden_units,102)),
        ('output',nn.LogSoftmax(dim=1))]))
    model.classifier=classifier

    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr=learning_rate)

    return model,optimizer,criterion

#calculate loss and accuracy
def validation(model,testloader,criterion,gpu):
    if gpu:
        model.to("cuda")
    test_loss=0
    accuracy=0
    for images, labels in testloader:
        if gpu:
            images, labels = images.to("cuda"), labels.to("cuda")
        outputs =  model.forward(images)
        test_loss += criterion(outputs, labels).item()    
        ps=torch.exp(outputs)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


####Train Model
def train_model(model, train_data, valid_data, criterion, optimizer, epochs, gpu):
    print_every = 40
    steps = 0
    if gpu:
        model.to("cuda")

    for e in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in train_data:
            steps += 1
            if gpu:
                images, labels = images.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss,accuracy = validation(model,valid_data,criterion,gpu)
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.3f}".format(running_loss/print_every),
                          "Valid Loss: {:.3f}".format(valid_loss/len(valid_data)),
                          "Valid Accuracy: {:.3f}".format(accuracy/len(valid_data)))

                    running_loss = 0
                    model.train()
    return model

#save model to checkpoint
def save_model(model, epochs,hidden_units, arch, class_to_idx, optimizer, save_dir=None): 
    if save_dir is not None:
        state = {
        'epochs': epochs,
        'hidden_units':hidden_units,
        'arch':arch,
        'class_to_idx':class_to_idx,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
        torch.save(state, save_dir)
        print ("The model has been saved in "+save_dir)
        pass

def main():
    parameters = get_input_para()
    data_dir = parameters.data_dir
    save_dir = parameters.save_dir
    arch = parameters.arch
    learning_rate = parameters.learning_rate
    hidden_units = parameters.hidden_units
    epochs = parameters.epochs
    gpu = parameters.gpu

    ####Load Data
    train_data, valid_data, test_data, train_datasets = loadData(data_dir)
    class_to_idx = train_datasets.class_to_idx

    pre_model,optimizer,criterion = get_pre_model(arch,learning_rate,hidden_units,epochs,gpu)
    trained_model = train_model(pre_model, train_data, valid_data, criterion, optimizer, epochs, gpu)
    test_loss, accuracy = validation(trained_model,test_data,criterion,gpu)
    print( "Test Loss: {:.3f}".format(test_loss/len(test_data)),"Test Accuracy: {:.3f}".format(accuracy/len(test_data)))
    save_model(trained_model, epochs,hidden_units, arch, class_to_idx, optimizer, save_dir=None)
    pass

if __name__ == '__main__':
    main()



    