import torch
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms


import numpy as np

import PIL
from PIL import Image
import matplotlib.pyplot as plt
 

from math import ceil 
import time
import json
import copy
import seaborn as sns
import argparse

# Functions arg_parser() parses command line arguments
# Define  parser
def arg_parser():
    parser = argparse.ArgumentParser("Predict network settings")
    parser.add_argument('--image', default='/home/workspace/ImageClassifier/flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str) 
    parser.add_argument('--checkpoint', default = '/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action = 'store', type= str)
    parser.add_argument('--topk', default='3', dest ='topk', action='store', type =int,   help='TopK matches as int')
    parser.add_argument('--category_names', dest='category_names',  action= 'store', default= '/home/workspace/ImageClassifier/cat_to_name.json', help='mapping categories to names')
#    parser.add_argument('--labels', type=str, help='cat_to_name.json')
    parser.add_argument('--gpu',  default = 'gpu', action='store', dest='gpu', help='Use GPU + Cuda calculations')
    
    return parser.parse_args()
        
# Use command line values when specified
# Starts the code to predict the class from image file
#    ''' Predicting the class of an image using a trained deep learning model.
#    '''

   
# Loading the checkpoint file
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load('checkpoint.pth')

    if checkpoint['architecture'] == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
    else: 
        exec("model = models.{}(pretrained=True)".checkpoint['architecture'])
        model.name = checkpoint['architecture']
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
          
    #Loads defaults #arch = checkpoint_dict['arch']

    #Loads the checkpoint model
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
    
    
# Checks for GPU if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")        
        model.cuda()
        
    was_training = model.training    
    model.eval()
    

    #Predicts classes of an image using trained learning model 
def process_image (image_path):

    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    
    np_image = np.array(pil_image)/255    
    
    # Normalize each color channel
    normalize_means = [0.485, 0.456, 0.406],
    normalized_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalize_means) / normalize_std
    
    np_image = np_image.transpose (2, 0, 1)
    

    return np_image
    #Device GPU and Cuda is available: predicts top K classes
def predict(image_tensor, model, device, cat_to_name, topk):
# Predict the class (or classes) of an image using a trained deep learning model.
    
     # No need for GPU on this part (just causes problems)
    model.to("cpu")
    
    
    if type(topk) == type(None):
        topk = 5
        print("TopK = 5")
    
    
    # Set model to evaluate
    model.eval();

    # Convert image from numpy to torch
    torch_image = torch.from_numpy(np.expand_dims(image_tensor, 
                                                  axis=0)).type(torch.FloatTensor)
    
    model=model.cpu()
    
    

    # Find probabilities (results) by passing through the function (note the log softmax means that its on a log scale)
    log_probs = model.forward(torch_image)

    # Convert to linear scale
    linear_probs = torch.exp(log_probs)

    # Find the top 5 results
    top_probs, top_labels = linear_probs.topk(top_k)
    
    # Detatch all of the details
    top_probs = np.array(top_probs.detach())[0] # This is not the correct way to do it but the correct way isnt working thanks to cpu/gpu issues so I don't care.
    top_labels = np.array(top_labels.detach())[0]
    
    # Convert to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]
    
    return top_probs, top_labels, top_flowers

#display image and top 5 classes
#creates two lists in a dictionary to print on screen

def print_probs(probs, flowers):
  
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, Percent Chance: {}%".format(j[1], ceil(j[0]*100)))
    
def main():
    
    # Get Keyword Args for Prediction
    args = arg_parser()
    
    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)
    
    # Process Image
    image_tensor = process_image(args.image)
    
    # Check for GPU
    device= gpu(args.gpu)
    
    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model, 
                                                 device, cat_to_name,
                                                 args.topk)
    
    # Print out probabilities
    print_probs(top_flowers, top_probs)


# Run Program
#==========================
if __name__ == '__main__': main()
