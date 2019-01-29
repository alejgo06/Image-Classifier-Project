import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import argparse
#user def functions




parser = argparse.ArgumentParser()
parser.add_argument('--path_checkpoint', action='store',
                    dest='path_checkpoint',default='checkpoint20.pth',
                    help='path_checkpoint')
parser.add_argument('-test', action='store_true',
                    default=False,
                    dest='test',
                    help='test')
results = parser.parse_args()
print('path_checkpoint     = {!r}'.format(results.path_checkpoint))
print('test     = {!r}'.format(results.test))
checkpoint=results.path_checkpoint
test=results.test
def mi_validation(model, loader, criterion):
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy
#end user def 
data_dir = '/home/workspace/aipnd-project/flowers'#cambiar
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
testloader =  torch.utils.data.DataLoader(test_data, batch_size=32)



with open('/home/workspace/aipnd-project/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
criterion = nn.NLLLoss()

checkpoint = torch.load('/home/workspace/'+checkpoint)
modelo=checkpoint['modelo']
model=modelo

if test==True:
    with torch.no_grad():
        test_loss, accuracy = mi_validation(modelo, testloader, criterion)
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
else:
    with torch.no_grad():
        valid_loss, accuracy = mi_validation(modelo, validloader, criterion)
    print("Valid Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
          "Valid Accuracy: {:.3f}".format(accuracy/len(validloader)))
    