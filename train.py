### Import necessary libraries
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image 
import numpy as np
import matplotlib.pyplot as plt
import json

class empty_generic:
    pass
program_arguments = empty_generic()
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', help = 'directory with training dataset')
parser.add_argument('--arch', default = 'vgg16', help = 'accepts vgg16 or densenet161')
parser.add_argument('--epochs', default = 10, type = int, help = 'number of epochs')
parser.add_argument('--save_dir', help = 'save directory')
parser.add_argument('--learning_rate', default = 0.003, type = float, help = 'learning rate')
parser.add_argument('--device', default = 'cpu', help = 'allows cuda or cpu')
args = parser.parse_args(namespace = program_arguments)
data_dir = program_arguments.data_dir
learnrate = program_arguments.learning_rate
dev = program_arguments.device
arch = program_arguments.arch
epoch_count = program_arguments.epochs
#print("the directory you entered was {}".format(data_dir))

def main():
    ### define transformations for the data
    train_transforms = transforms.Compose([transforms.RandomRotation(60),
                                     transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.RandomHorizontalFlip(30),
                                     transforms.ColorJitter(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    ### define paths to the train, validation, and test data sets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    ### load in the datasets
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    ### set up dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)

    ### define processor
    device = torch.device(dev)
    print("using device '{}'".format(device))
    ### define model architecture and optimizer
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        #class_in = 25088
    else:
        model = models.densenet161(pretrained = True)
        #class_in = 2208
    class_in = model.classifier.in_features   
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(class_in, 2000),
                                nn.ReLU(),
                                nn.Dropout(p = 0.2),
                                nn.Linear(2000, 512),
                                nn.ReLU(),
                                nn.Dropout(p = 0.2),
                                nn.Linear(512,102),
                                nn.LogSoftmax(dim = 1))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learnrate)
    model = model.to(device);
    
    ### train the network
    epochs = epoch_count
    training_losses = []
    validation_losses = []
    model.train()    
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
        
            images = images.to(device) 
            labels = labels.to(device)
            #print("image shape: '{}'".format(images.shape))
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            #print("loss: {}".format(loss.item()))
            running_loss += loss.item()
        
        else:
            valid_loss = 0 
            accuracy = 0
       
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device) 
                    logps = model.forward(images)
                    valid_loss +=  criterion(logps, labels)
                    #print("step: {}, valid_loss: {}".format(e, valid_loss))
           
                
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                
            model.train()        
            training_losses.append(running_loss/len(trainloader))
            validation_losses.append(valid_loss/len(validloader))
        
            print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(training_losses[-1]),
              "Test Loss: {:.3f}.. ".format(validation_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
    ### map from integer values to flower names
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    ### attach map as a parameter to the model
    model.class_to_idx = train_data.class_to_idx
    
    ### save model parameters
    checkpoint = {'input size': 25088,
             'output size': 102,
             'epochs': epochs,
             'model': model,
             'classifier': nn.Sequential(nn.Linear(class_in, 2000),
                                nn.ReLU(),
                                nn.Dropout(p = 0.2),
                                nn.Linear(2000, 512),
                                nn.ReLU(),
                                nn.Dropout(p = 0.2),
                                nn.Linear(512,102),
                                nn.LogSoftmax(dim = 1)),
             #'classifier': model.classifier(),     
             'optimizer': optimizer.state_dict(),     
             'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
    
    # save the state dict 
    torch.save(model.state_dict(), 'state_dict.pth')
   

if __name__ == "__main__":
    main()



