import pandas as pd
import argparse
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import glob, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
#user def functions 
def process_image(image="/home/workspace/aipnd-project/flowers/train/10/image_07086.jpg"):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    size = 224, 224
    im=im.resize(size)
    im = np.asarray(im)
    im=im/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im=(im-mean)/std
    im = im.transpose((2, 0, 1))
    img=torch.FloatTensor(im)
    return(img)
def predict(image_path, model, k=5,device='cuda'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''    
    #
    inputs=image_path
    inputs=inputs.unsqueeze_(0)
    inputs= inputs.to(device)
    outputs = model.forward(inputs)
    ps = torch.exp(outputs)
    prob=ps.cpu().detach().numpy()
    mio=pd.DataFrame({'probability':prob[0],'id':np.arange(len(prob[0]))},columns=['probability','id'])
    mioOrdenado=mio.sort_values(by=['probability'],ascending=False)
    classes=mioOrdenado.iloc[0:k,0]
    probs=mioOrdenado.iloc[0:k,1]
    #
    
    return probs, classes
def imshow(image=process_image(), ax=None, title=" "):
#    if ax is None:
#        fig, ax = plt.subplots()
    fig, ax = plt.subplots()
    #----
    image_path=image
    image_path=image_path.numpy()
    image = image_path.transpose((1, 2, 0)) 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean 
    image = np.clip(image, 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title(title)
    plt.show()
#end user def functions

parser = argparse.ArgumentParser()
parser.add_argument('--path_checkpoint', action='store',
                    dest='path_checkpoint',default="/home/workspace/checkpoint20.pth",
                    help='path_checkpoint')
parser.add_argument('--path_image', action='store',
                    dest='path_image',default="/home/workspace/aipnd-project/flowers/train/10/image_07086.jpg",
                    help='path_image')
parser.add_argument('--category_names', action='store',
                    dest='category_names',
                    help='category_names',default="/home/workspace/aipnd-project/cat_to_name.json")
parser.add_argument('--top_k', action='store',
                    dest='top_k',
                    help='top_k',type=int,default=3)
parser.add_argument('-gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='gpu')
results = parser.parse_args()
print('path_checkpoint     = {!r}'.format(results.path_checkpoint))
print('path_image     = {!r}'.format(results.path_image))
print('category_names     = {!r}'.format(results.category_names))
print('top_k     = {!r}'.format(results.top_k))
print('gpu     = {!r}'.format(results.gpu))



path_checkpoint=results.path_checkpoint
path_image=results.path_image
category_names=results.category_names
top_k=results.top_k

gpu=results.gpu





with open(category_names, 'r') as f:
    cat_to_name = json.load(f)


checkpoint = torch.load(path_checkpoint)

modelo=checkpoint['modelo']

if gpu==False:
    print("cpu is used")
    device='cpu'
else:
    print("gpu is used")
    device='cuda'
print(device)

ima=path_image
imagen=process_image(image=ima)
cla,prob=predict(image_path=imagen, model=modelo,k=top_k,device=device)
print(pd.DataFrame(prob,cla))

