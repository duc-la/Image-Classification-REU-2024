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
%matplotlib inline

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


# print monai configuration
monai.config.print_config()

# Set deterministic training for reproducibility
monai.utils.set_determinism(seed=0)


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


def evaluation(args,model,data_loader):
    ### This function accepts args, model and data_loader(can be validation or train)
    ### Puts model in eval mode.
    ### return accuracy, average loss and auc scores for the data loader
    
    ### Your code starts here ###
    
    
    ### Your code ends here ###



def train(args,model,train_loader,val_loader):
    ### This function accepts args, model, train_loader, val_loader
    ### Set up the training process using data from train_loader
    ### after each epoch calls the evaluation function to evaluate metrics for both train and validation dataset
    ### saves the best model based on validation accuracy
    ### Note:- If you put metrics in the list, you can return the list and use matplotlib for further analysis
    ### Your code starts here ###
        
    
    
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


    if(args.model==0):
        ## Instantiate the ResNet101 model from monai with spatial_dims, n_input_channels and num_classes parameters ##
    elif(args.model==1):
        ## Instantiate the DenseNet121 model from monai with spatial_dims, in_channels, and out_channels parameters ##
    elif(args.model==2):
        ## Instantiate the ViT model from monai with spatial_dims, n_input_channels, img_size,patch_size, num_classes,classification parameters ##
                          
    
    #### You can reuse the code from previous exercise.

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
    parser.add_argument('-model'          ,type=int  , action="store", dest='model'                    , default=0             )
    parser.add_argument('-batchSize'      ,type=int  , action="store", dest='batchSize'                , default=64            )
    parser.add_argument('-lr'             ,type=float, action="store", dest='learningRate'             , default=0.0001        )
    parser.add_argument('-nepoch'         ,type=int  , action="store", dest='epochs'                   , default=30            )
    args = parser.parse_args()
    main(args)

