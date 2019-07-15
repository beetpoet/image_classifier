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

### get picture from command line input
class empty_generic:
    pass
program_arguments = empty_generic()
parser = argparse.ArgumentParser()

parser.add_argument('pic_path', help = 'path to a picture to classify')
parser.add_argument('checkpoint', help = 'path to saved trained model parameters')
parser.add_argument('state_dict', help = 'path to saved state dictionary')
parser.add_argument('--top_k', type = int, default = 3, help = 'return top how many predictions')
#parser.add_argument('--category_names', default = cat_to_name.json, help = 'map integer values to category names')
parser.add_argument('--device', default = 'cpu', help = 'accepts cuda or cpu')
args = parser.parse_args(namespace = program_arguments)
img_path = program_arguments.pic_path
checkpoint = program_arguments.checkpoint
state_dict = program_arguments.state_dict
top_k = program_arguments.top_k
#cat_names_map = program_arguments.category_names
dev = program_arguments.device

### function to build the model from saved checkpoint
def build_model(model_filepath, state_dict_filepath):
    device = torch.device(dev)
    #model = models.vgg16(pretrained=True)
    model_params = torch.load(model_filepath)
    model = model_params['model']
    model.classifier = model_params['classifier']
    model.load_state_dict(torch.load(state_dict_filepath))
    model.class_to_idx = model_params['class_to_idx']
    optimizer = model_params['optimizer']
    model.to(device)
    return model

### process the image provided in the command line
def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        '''
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])
    
        img = Image.open(image)
        img = preprocess(img).float()
        img = np.array(img)
        #img = img / 255
    
        means = [0.485, 0.456, 0.406]
        st_dev = [0.229, 0.224, 0.225]
    
        img = (np.transpose(img, (1, 2, 0)) - means)/st_dev
        img = np.transpose(img, (2, 0, 1))
        return img
    
def predict(image, model, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(model.classifier.parameters(), lr = 0.003)
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        #model = build_model(checkpoint)
        #model = model.to(device)

        image = process_image(img_path)
        
        image = torch.FloatTensor(image).to(device)
        #image = image.to(device)
   

        with torch.no_grad():
            image.unsqueeze_(0)
            model.eval()
            optimizer.zero_grad()
        
            logps = model.forward(image)
        
            ps = torch.exp(logps)
            probs, classes  = ps.topk(topk, dim = 1)
            idx_to_class = {a: b for b, a in model.class_to_idx.items()}
            probs=probs.cpu().numpy()[0].tolist()
            classes=classes.cpu().numpy()[0].tolist()   
            pred_labels = [idx_to_class[x] for x in classes]
            pred_class = [cat_to_name[str(x)] for x in pred_labels]
        return probs, pred_class
### build the model from train.py
def main():
    ''' 
    Load a saved, trained model and use it to classify an image
    '''   
    ### build the model
    model = build_model('checkpoint.pth', 'state_dict.pth')
    ### classify the picture (here come dat boi)
    prediction = predict(img_path, model, top_k)
    print(prediction)
    
if __name__ == "__main__":
    main()