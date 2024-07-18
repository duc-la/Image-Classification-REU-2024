import os
import platform
import subprocess
import importlib.util
import sys
from packaging import version
import numpy as np
import sklearn
from sklearn.metrics import classification_report

from tqdm import tqdm

import matplotlib.pyplot as plt
# %matplotlib inline

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torchvision.transforms as torch_transforms

# medmnist imports
import medmnist

# monai imports
import monai
import monai.data as monai_data
import monai.transforms as monai_transforms
import monai.networks.nets as monai_nn

import argparse
import copy


# print monai configuration
monai.config.print_config()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)

# Set device to GPU
my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("THE DEVICE" + str(my_device))
dtype = torch.float32

#### Function to print a given dictionary in a json-like format ####
def print_dict(results):
    for key, value in results.items():
        print(f"{key}: {value}")

##### Install necessary libraries if they are not already installed and import them #####
def check_and_install_package(package_name):
    if importlib.util.find_spec(package_name) is not None: # Check if the package is installed
        print(f"{package_name} is already installed.")
    else:                                                  # If not installed, install it using pip
        print(f"{package_name} is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} has been successfully installed.")
    
##### Check torch library requirement #####
def check_pytorch_access():
    
    # Check if PyTorch is installed and accessible
    pytorch_installed     = torch.__version__ is not None
    my_torch_version      = torch.__version__
    minimum_torch_version = '2.0'
    
    if not pytorch_installed:
        print('Warning!!! Your Torch is not installed. Please install Torch.')
        return {
            "PyTorch Installed": pytorch_installed,
            "PyTorch Version": "None"
        }
        
    if version.parse(my_torch_version) < version.parse(minimum_torch_version):
        print('Warning!!! Your Torch version %s does NOT meet the minimum requirement!\
                Please update your Torch library\n' %my_torch_version)

    return {
        "PyTorch Installed": pytorch_installed,
        "PyTorch Version": my_torch_version
    }

##### Check PyTorch has access to GPU #####
def check_gpu_access():
    # Check the current operating system
    os_name = platform.system()

    device = 'cpu'

    # Switch case based on the operating system
    if os_name == 'Windows':
        # Windows specific GPU check
        if torch.cuda.is_available():
            device = 'cuda'

    elif os_name == 'Linux':
        # Linux specific GPU check
        if torch.cuda.is_available():
            device = 'cuda'

    elif os_name == 'Darwin':
        # Mac specific GPU check
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'

    return {
        "Operating System": os_name,
        "Device": device
    }


##### Check if the dataset has been downloaded before #####
def check_medmnist_downloaded(dataset_dir):
    # Check if the dataset directory exists
    if not os.path.exists(dataset_dir):
        return False
    
    # Check if there are any files in the directory
    if not os.listdir(dataset_dir):
        return False

    return True

## Set up the download flag and create directory if they haven't existed yet 
def set_medmnist_download_flag_and_dir(dataset_dir):
    is_downloaded = check_medmnist_downloaded(dataset_dir)

    if not is_downloaded:
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)    
        download = True
    else:
        download = False

    return download

##### Implement a CNN, named model_ex1, with the architecture specified above #####
### Your code starts here ###
class Ex1Net(nn.Module):
    def __init__(self, n_channels, num_classes):
        super(Ex1Net, self).__init__()
        #    Block #1
        #        Convolution layer: input channels = n_channels, kernel size = 3; output features = 16; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 16
        #        Relu activation
        self.conv1 = nn.Conv2d(in_channels = n_channels, kernel_size=3, out_channels=16, padding=0, stride=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features= 16)
        self.relu = nn.ReLU()

        #    Block #2
        #        Convolution layer: input channels = 16, kernel size = 3; output features = 16; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 16
        #        Relu activation
        #        Max pooling layer: kernel size = 2; stride = 2
        self.conv2 = nn.Conv2d(in_channels=16, kernel_size=3, out_channels=16, padding=0, stride=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=16)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #    Block #3
        #       Convolution layer: input channels = 16, kernel size = 3; output features = 64; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 64
        #        Relu activation
        self.conv3 = nn.Conv2d(in_channels=16, kernel_size=3, out_channels=64, padding=0, stride=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)
        
        #    Block #4
        #        Convolution layer: input channels = 64, kernel size = 3; output features = 64; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 64
        #        Relu activation
        self.conv4 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, padding=0, stride=1)
        self.batch_norm4 = nn.BatchNorm2d(num_features=64)
        
        #    Block #5
        #        Convolution layer: input channels = 64, kernel size = 3; output features = 64; padding = 1; stride = 1; containing bias
        #        Batch normalization layer: features = 64
        #        Relu activation
        #        Max pooling layer: kernel size = 2; stride = 2
        self.conv5 = nn.Conv2d(in_channels=64, kernel_size=3, out_channels=64, padding=1, stride=1)
        self.batch_norm5 = nn.BatchNorm2d(num_features=64)
        self.maxPool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #    Block #6
        #        Fully-connected Layer: input features = 1024; output features = 128; containing bias
        #        Relu activation
        #        Fully-connected Layer: input features = 128; output features = 128; containing bias
        #        Relu activation
        #        Fully-connected Layer: input features = 128; output features = n_classes; containing bias
        self.FC1 = nn.Linear(in_features=1024, out_features=128)
        self.FC2 = nn.Linear(in_features=128, out_features=128)
        self.FC3 = nn.Linear(in_features=128, out_features=num_classes)




    def forward(self,x):
        #    Block #1
        #        Convolution layer: input channels = n_channels, kernel size = 3; output features = 16; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 16
        #        Relu activation
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        # print("11111111111")
        # print(x.size())

        #    Block #2
        #        Convolution layer: input channels = 16, kernel size = 3; output features = 16; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 16
        #        Relu activation
        #        Max pooling layer: kernel size = 2; stride = 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.maxPool2(x)
        # print("22222222222")
        # print(x.size())

        #    Block #3
        #    Convolution layer: input channels = 16, kernel size = 3; output features = 64; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 64
        #        Relu activation
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        # print("22222222222")
        # print(x.size())

        #    Block #4
        #        Convolution layer: input channels = 64, kernel size = 3; output features = 64; padding = 0; stride = 1; containing bias
        #        Batch normalization layer: features = 64
        #        Relu activation
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        #
        # print("22222222222")
        # print(x.size())

        #    Block #5
        #        Convolution layer: input channels = 64, kernel size = 3; output features = 64; padding = 1; stride = 1; containing bias
        #        Batch normalization layer: features = 64
        #        Relu activation
        #        Max pooling layer: kernel size = 2; stride = 2
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.relu(x)
        x = self.maxPool5(x)
        #
        # print("22222222222")
        # print(x.size())

        # Flatten the tensor before the Fully Connected Layers
        x = x.view(x.size(0), -1)  # Flatten the tensor
        #    Block #6
        #        Fully-connected Layer: input features = 1024; output features = 128; containing bias
        #        Relu activation
        #        Fully-connected Layer: input features = 128; output features = 128; containing bias
        #        Relu activation
        #        Fully-connected Layer: input features = 128; output features = n_classes; containing bias
        x = self.FC1(x)
        x = self.relu(x)
        x = self.FC2(x)
        x = self.relu(x)
        x = self.FC3(x)
        #
        # print("22222222222")
        # print(x.size())

        return x





### Your code ends here ###


def evaluation(args,model,data_loader):
    ### This function accepts args, model and data_loader(can be validation or train)
    ### Puts model in eval mode.
    ### return accuracy, average loss and auc scores for the data loader
    
    ### Your code starts here ###
    model.eval()

    with torch.no_grad():
        # run through several batches, does inference for each and store inference results
        # and store both target labels and inferenced scores
        acc = 0.0
        for image, target in data_loader:
            #image = temp['x']
            #target = temp['label']
            #image = torch.unsqueeze(image, 1)

            ## Send inputs to GPU
            image = image.to(dtype=dtype, device=my_device)
            target = target.to(dtype=torch.long, device=my_device)
            probs = model(image)
            preds = torch.argmax(probs, 1)
            acc += torch.count_nonzero(preds == target)

        return acc / len(data_loader.dataset)

    ### Your code ends here ###



def train(args,model,train_loader,val_loader):
    ### This function accepts args, model, train_loader, val_loader
    ### Set up the training process using data from train_loader
    ### after each epoch calls the evaluation function to evaluate metrics for both train and validation dataset
    ### saves the best model based on validation accuracy
    ### Note:- If you put metrics in the list, you can return the list and use matplotlib for further analysis
    ### Your code starts here ###

    # parser.add_argument('-batchSize', type=int, action="store", dest='batchSize', default=64)
    # parser.add_argument('-lr', type=float, action="store", dest='learningRate', default=0.0001)
    # parser.add_argument('-nepoch', type=int, action="store", dest='epochs', default=30)

    #Pick Cross Entropy loss for multiple classification
    criterion = torch.nn.CrossEntropyLoss()
    train_loader1 = train_loader#torch.utils.data.DataLoader(train_loader, batch_size=args.batchSize, num_workers=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learningRate, momentum=0.9)

    full_loss = []
    training_accuracy_values = []
    validation_accuracy_values = []
    best_val_acc = 0

    for e in range(args.epochs):
        print("Running Epoch ", e)
        for image, target in train_loader1:
            #image = temp['x']
            #target = temp['target']
            #image = torch.unsqueeze(image, 1)

            ## Put Model in Training Mode
            model.train()

            image = image.to(dtype=dtype, device=my_device)
            target = target.to(dtype=torch.long, device=my_device)
            model = model.to(dtype=dtype, device=my_device)

            ## Forward pass
            output = model(image)
            #print(output.size())
            #print(target.size())
            target = target.squeeze()
            #print(target.size())
            classification_loss = criterion(output, target)

            ## Set Gradients to Zero
            optimizer.zero_grad()

            ## Backward pass
            classification_loss.backward()

            ## Optimizer Step
            optimizer.step()

            full_loss.append(classification_loss.item())

        val_accu = evaluation(args, model, val_loader)
        train_accu = evaluation(args, model, train_loader)
        training_accuracy_values.append(train_accu)
        validation_accuracy_values.append(val_accu)

        print('Validation Accuracy ', val_accu.cpu().numpy())
        if (val_accu > best_val_acc):
            best_val_acc = val_accu
            best_trained_model = copy.deepcopy(model.state_dict())
            torch.save(best_trained_model, './best_trained_model.pt')
    
    ### Your code ends here ###


def main(args):
    # Execute the function and print the results
    check_pytorch_access_result = check_pytorch_access()
    print_dict(check_pytorch_access_result)

    gpu_access = check_gpu_access()
    print_dict(gpu_access)
    
    device = gpu_access['Device']
    
    device = torch.device(device)
    print(f"Using device: {device}")

    ##### Specify which dataset to use #####
    data_flag = 'pathmnist'     # available 2d datasets: {'pathmnist', 'octmnist', 'pneumoniamnist', 
                                #                         'chestmnist', 'dermamnist', 'retinamnist', 
                                #                         'breastmnist', 'bloodmnist', 'tissuemnist',
                                #                         'organamnist', 'organcmnist', 'organsmnist'}
    
    ## getting the dataset information
    data_info  = medmnist.INFO[data_flag]
    print_dict(data_info)
    
    ## getting the classification task and number of image channles and classes of the considered dataset
    task       = data_info['task']
    n_channels = data_info['n_channels']
    n_classes  = len(data_info['label'])
    
    ## getting the object of the Python class that represents the dataset
    DataClass  = getattr(medmnist, data_info['python_class'])
    
    ##### Set up the data directory #####
    dataset_dir = './data/' + data_flag + '/'
    print(f"{data_flag} dataset folder: {dataset_dir}")
    
    ## Set up the download flag and create directory if they haven't existed yet     
    download = set_medmnist_download_flag_and_dir(dataset_dir)
    print(f"MedMNIST {data_flag} dataset downloaded: {not download}")


    # preprocessing
    data_transform = torch_transforms.Compose([
                                                torch_transforms.ToTensor(),
                                                torch_transforms.Normalize(mean=[0.0], std=[1])
                                            ])
    
    # Download (if needed) and load the dataset
    train_data = DataClass(root = dataset_dir, split = 'train', transform = data_transform,  download = download)
    val_data   = DataClass(root = dataset_dir, split = 'val',   transform = data_transform,  download = download)
    test_data  = DataClass(root = dataset_dir, split = 'test',  transform = data_transform,  download = download)
    
    print("========= Train Dataset ==========")
    print(train_data)
    print("========= Val Dataset ==========")
    print(val_data)
    print("========= Test Dataset ==========")
    print(test_data)

    ##### Encapsulate data into dataloader form #####
    batch_size = args.batchSize # Modify this accoding to your GPU RAM size
    
    train_loader = torch_data.DataLoader(dataset = train_data, batch_size = batch_size,   shuffle = True)
    val_loader   = torch_data.DataLoader(dataset = val_data,   batch_size = 2*batch_size, shuffle = False)
    test_loader  = torch_data.DataLoader(dataset = test_data,  batch_size = 2*batch_size, shuffle = False)


    ## Fill the Above Ex1Net Code ##
    model = Ex1Net(n_channels=n_channels, num_classes=n_classes)
    
    
    ##### Training Process #####
    train(args,model,train_loader,val_loader)

    ##### Testing Process #####
    ### Reload the best saved model from training and report testing accuracy.
    ### Your code starts here ###


    evaluation(args,model,test_loader)

    ### Your code ends here ###


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Short sample app')
    ## Model Arguments
    parser.add_argument('-batchSize'      ,type=int  , action="store", dest='batchSize'                , default=64            )
    parser.add_argument('-lr'             ,type=float, action="store", dest='learningRate'             , default=0.0001        )
    parser.add_argument('-nepoch'         ,type=int  , action="store", dest='epochs'                   , default=30            )
    args = parser.parse_args()
    main(args)

