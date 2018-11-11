import argparse
import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from DataPreprocess import *

##Set Parameter 
def get_input_para():  
    parser = argparse.ArgumentParser(
        description='Image Classifier Prediction Parameters',
    )

    parser.add_argument('image_path', action="store")
    parser.add_argument('input', action="store")
    parser.add_argument('--top_k', action="store",default=1,type=int)
    parser.add_argument('--category_names', action="store")
    parser.add_argument('--gpu', action="store_true", default=True)

    return parser.parse_args()


##Load Model
def loadModel(filePath):
    state = torch.load(filePath)
    hidden_units=state['hidden_units']
    arch=state['arch']
    class_to_idx=state['class_to_idx']
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

    state = torch.load(filePath)
    model.load_state_dict(state['state_dict'])
    model.class_to_idx=class_to_idx
    return model

## Predict
def predict(image_path, model, topk,gpu,category_names=None):
    # predict the class from an image file
    from PIL import Image
    if gpu:
        model.to("cuda")
    class_to_idx=model.class_to_idx
    idx_to_class={y:x for x,y in class_to_idx.items()}
    
    im = Image.open(image_path)
    np_image=process_image(im)
    image=torch.from_numpy(np_image).type(torch.FloatTensor)
    image=image.resize_(1,np_image.shape[0],np_image.shape[1],np_image.shape[2])
    if gpu:
        image = image.to("cuda")
    
    with torch.no_grad():
        output=model.forward(image)
    
    ps=torch.exp(output)
    probs=list(ps[0].topk(topk)[0].cpu().numpy())
    classes=list(ps[0].topk(topk)[1].cpu().numpy())
    classes_map=[idx_to_class[ele] for ele in classes]
    
    if category_names is not None:
        import json
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes_name=[cat_to_name[ele] for ele in classes_map]
        return classes_name, probs
    else:
        return classes_map, probs    
 

def main():
    parameters=get_input_para()
    image_path=parameters.image_path
    checkpoint_path=parameters.input
    top_k=parameters.top_k
    category_names=parameters.category_names
    gpu=parameters.gpu

    model=loadModel(checkpoint_path)
    classes,probs=predict(image_path, model, top_k,gpu,category_names)
    print (classes,probs)
    pass

if __name__ == '__main__':
    main()