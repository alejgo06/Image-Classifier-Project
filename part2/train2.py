import argparse
import json
from collections import OrderedDict
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', action='store',
                    dest='data_directory',
                    help='data_directory')
parser.add_argument('--save_dir', action='store',
                    dest='save_dir',
                    help='save_dir')
parser.add_argument('--arch', action='store',
                    dest='arch',
                    help='arch',default="vgg16")
parser.add_argument('--learning_rate', action='store',
                    dest='learning_rate',
                    help='learning_rate',type=float,default=0.01)
parser.add_argument('--hidden_units', action='store',
                    dest='hidden_units',
                    help='hidden_units',type=int,default=512)
parser.add_argument('--epochs', action='store',
                    dest='epochs',
                    help='epochs',type=int,default=20)
parser.add_argument('-gpu', action='store_true',
                    default=False,
                    dest='gpu',
                    help='gpu')
results = parser.parse_args()
print('data_directory     = {!r}'.format(results.data_directory))
print('save_dir     = {!r}'.format(results.save_dir))
print('arch     = {!r}'.format(results.arch))
print('learning_rate     = {!r}'.format(results.learning_rate))
print('hidden_units     = {!r}'.format(results.hidden_units))
print('epochs     = {!r}'.format(results.epochs))
print('gpu     = {!r}'.format(results.gpu))


data_directory=results.data_directory
save_dir=results.save_dir
arch=results.arch
learning_rate=results.learning_rate
hidden_units=results.hidden_units
epochs=results.epochs
gpu=results.gpu

# change to cuda
print(gpu)
if gpu==False:
    print("se usa la cpu")
    device='cpu'
else:
    print("se usa la gpu")
    device='cuda'
print(device)

data_dir = data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid' 
test_dir = data_dir + '/test'

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






print(arch)
if(arch=="vgg16"):
    print("se entrena vgg16")
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
elif(arch=="alexnet"):
    print("se entrena alexnet")
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(9216, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
print(model)


print_every = 40
steps = 0


criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

def mi_validation(model, loader, criterion):
    test_loss = 0
    accuracy = 0
    for ii, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy




model.to(device)
running_loss = 0
for e in range(epochs):

    model.train()
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:

            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            model.eval()

            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = mi_validation(model, validloader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

            running_loss = 0


            model.train()

checkpoint = {'input_size': 50176,
              'output_size': 102,
              'class_to_idx':  train_data.class_to_idx,
              'nepochs':epochs,
              'state_dict': model.state_dict(),
              'clasificador':model.classifier,
             'modelo':model}

torch.save(checkpoint, save_dir+'.pth')

