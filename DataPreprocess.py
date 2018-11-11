import numpy as np
import time
import torch
from torchvision import datasets, transforms, models

def loadData(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms={'train':transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
                     'valid':transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
                     'test':transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])}

    image_datasets={'train':datasets.ImageFolder(train_dir,transform=data_transforms['train']),
                    'valid':datasets.ImageFolder(valid_dir,transform=data_transforms['valid']),
                    'test':datasets.ImageFolder(test_dir,transform=data_transforms['test'])}

    dataloaders={'train':torch.utils.data.DataLoader(image_datasets['train'],batch_size=64,shuffle=True),
                 'valid':torch.utils.data.DataLoader(image_datasets['valid'],batch_size=32),
                 'test':torch.utils.data.DataLoader(image_datasets['test'],batch_size=32)}
    
    train_data=dataloaders['train']
    valid_data=dataloaders['valid']
    test_data=dataloaders['test']
    train_datasets=image_datasets['train']
    
    return train_data,valid_data,test_data,train_datasets


def process_image(image):
    # Process a PIL image for use in a PyTorch model
    image=image.resize([256,256])
    image=image.crop((0,0,224,224))
    np_image = np.array(image)
    np_image = np_image/255
    mean=np.array([0.485, 0.456, 0.406])
    std=np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image