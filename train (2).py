# AI Python Programming 2020
#Image Classifier Part2 - train.py
#lourdes wellington

#Import Libraries

import torch
import torch.nn as nn
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torchvision import datasets, transforms, models

import torch.optim as optim
import torch.utils.data as data

import torchvision.datasets as datasets
import argparse

# Command line arguments
def arg_parser():
    
#define paraser
    parser = argparse.ArgumentParser()
    #Path to the directory with images
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    #Argument : Use GPU for training
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    #Argument : Number of passes of the training data
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    #Argument :Architecture model to use for image classification
    parser.add_argument('--arch', type=str, help='Model architecture')
    # Argument : Model learning rate
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    #Argument : Units in hidden layer pre-classifier
    parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
    #Argument : Directory to save  checkpoint model file
    parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')

#Parser args    
    args = parser.parse_args()
    return args

# This method loads in a model

# Define Functions
# ------------------------------------------------------------------------------- #
 

# Function train_transformer(train_dir) performs training transformations on a dataset
def train_transformer(train_dir):
   # Define transformation
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
# Load the Data
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

# Function test_transformer(test_dir) performs test transformations on a dataset
def valid_transformer(valid_dir):
# Define transformation
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
# Load the Data
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_data

#Function  test_transformer(test_dir) performs validation transformations on a dataset 
def test_transformer(test_dir):
    #Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
# Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data


# Function data_loader(data, train=True) creates a dataloader from dataset imported
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader


# Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA not found on device, using CPU.")
    return device

# primaryloader_model(architecture="vgg13") downloads model (primary) from torchvision
def primaryloader_model(architecture="vgg13"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg13(pretrained=True)
        model.name = "vgg13"
        print("Network architecture is vgg13.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    return model

# Function initial_classifier(model, hidden_units) creates a classifier with the corect number of input layers
def initial_classifier(model, hidden_units):
    # Check that hidden layers are inputted
    if type(hidden_units) == type(None): 
        hidden_units = 512 #hyperparamters
        print("Number of Hidden Layers is 512.")
    
    # Find Input Layers
    input_features = model.classifier[0].in_features
    
    # Define Classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# Function validation(model, testloader, criterion, device) validates training against testloader to return loss and accuracy
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

# Function network_trainer represents the training of the network model
def network_trainer(Model, Trainloader, Testloader, Device, 
                  Criterion, Optimizer, Epochs, Print_every, Steps):
    
    # Check Model args
    if type(Epochs) == type(None):
        Epochs = 20
        print(" Epochs number is 5")    
 
    print("Training process initializing ...\n")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        
        Model.train() # initialize Model
        print_every = 5
        
        for ii, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            Optimizer.zero_grad()
            
            # Forward passes
            outputs = Model.forward(inputs)
            loss = Criterion(outputs, labels)
            # Backward pass
            loss.backward()
            Optimizer.step()
        
            running_loss += loss.item()
        
            if Steps % print_every == 0:
                valid_loss =0
                accuracy =0
                Model.eval()
                with torch.no_grad():
                                                                
                    print("Epoch: {}/{} | ".format(e+1, Epochs),
                         "Loss: {:.3f} | ".format(running_loss/print_every),
                         "Validation Loss: {:.3f} ".format(valid_loss/len(Testloader)),
                         "Validation Accuracy:{:.3f}".format(accuracy/len(Testloader)))

                    running_loss = 0
                    Model.train()

    return Model

#Function validate_model(Model, Testloader, Device) validate  above model on test data images
def validate_model(Model, Testloader, Device):
   # This does validation on test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on test images: %d%%' % (100 * correct / total))

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves model to this checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Need Model checkpoint directory, model not saved.")
    else:
        if isdir(Save_Dir):
            # Create  class_to_idx  attribute in model
            Model.class_to_idx = Train_data.class_to_idx
            
            # Create checkpoint dictionary
            checkpoint = {'architecture': Model.name,
                          'classifier': Model.classifier,
                          'class_to_idx': Model.class_to_idx,
                          'state_dict': Model.state_dict()}
            
            # Save checkpoint
            torch.save(checkpoint, "checkpoint.pth")

        else: 
            print("Directory not found, model  not saved.")



# Main Function
# ******************************************************************#

# Function main() all functions used above are called and executed 
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = valid_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = initial_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.01
        print("Learning rate is 0.01")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 30
    steps = 0
    

    
    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining is now complete")
    
    # Quickly Validate the model
    validate_model(trained_model, testloader, device)
    
    # Save the model
    initial_checkpoint(trained_model, args.save_dir, train_data)


# Run Program
# =============================================================================
if __name__ == '__main__': main()
    
