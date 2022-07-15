import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import gc


def add_noise(noise_type, image ,variance = 1e-3, seed =None ):
    if noise_type == "gauss":
        mean = 0
        var = variance  
        sigma = var**0.5
        np.random.seed(seed)
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(image.shape)
        img_noisy = image + gauss
        print(f'SNR: {np.mean(image)**2 / np.std(gauss)**2}')
    return img_noisy

"""def add_noise(noise_type, image):
    if noise_type == "gauss":
        mean = 0
        var = 1e-3  ####  1e-4 2e-02 
        sigma = var**0.5
        np.random.seed(seed=2)
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(image.shape)
        img_noisy = image + gauss
        print(f'SNR: {np.mean(image)**2 / np.std(gauss)**2}')
    return img_noisy"""

def timeit(f):
    """ Decorator to time Any Function """

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info("   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour" %
                                        (f.__name__, seconds, seconds / 60, seconds / 3600))
        return result

    return timed


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    call(["nvcc", "--version"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))

def count_parameters_by_model(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print(f'Layer: {name} {param} elements')
        total_params+=param
    print(f"Total Trainable Params: {total_params}\n")
    return total_params

def repeat(arr, count, dim=0):
    return torch.stack([arr for _ in range(count)], dim=dim)

#function to returns all the statistics of recon image
def getEvalCritStats(target, output, energy, dim=None):

    '''
        Input: targets, output, energy
        output: RMSE, MAE, weighted RMSE, weighted MAE
    '''

    mse = np.mean(np.abs((output - target) ** 2),axis=dim)
    rmse = np.sqrt(mse)
    mae = np.mean((np.abs(output - target)),axis=dim)
    mse_weighted = np.mean(np.abs(((output - target) ** 2) * energy), axis=dim)
    rmse_weighted = np.sqrt(mse_weighted)
    mae_weighted = np.mean(np.abs((output - target) * np.sqrt(energy)), axis=dim)

    return rmse, mae, rmse_weighted, mae_weighted  

def getEvalCritStats_torch(target, output, energy):

    '''
        Itorchut: targets, output, energy
        output: RMSE, MAE, weighted RMSE, weighted MAE
    '''

    mse = torch.mean(torch.abs((output - target) ** 2))
    rmse = torch.sqrt(mse)
    mae = torch.mean((torch.abs(output - target)))
    mse_weighted = torch.mean(torch.abs(((output - target) ** 2) * energy))
    rmse_weighted = torch.sqrt(mse_weighted)
    mae_weighted = torch.mean(torch.abs((output - target) * torch.sqrt(energy)))

    return rmse, mae, rmse_weighted, mae_weighted 

# Early stopping https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb

def saveTrainPlot(train_losses,validation_losses, loss_plot):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    print(fig)
    plt.plot(range(1,len(train_losses)+1),train_losses, label='Training Loss')
    plt.plot(range(1,len(validation_losses)+1),validation_losses,label='Validation Loss')

    # find position of lowest validation loss
    minposs = validation_losses.index(min(validation_losses))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(loss_plot, bbox_inches='tight')
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)
    gc.collect()

def saveTrainPlotOnly(train_losses ,loss_plot):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    print(fig)
    plt.plot(range(1,len(train_losses)+1),train_losses, label='Training Loss')
    # find position of lowest train loss
    minposs = train_losses.index(min(train_losses))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Lowest Train point')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(loss_plot, bbox_inches='tight')
    # Clear the current axes.
    plt.cla() 
    # Clear the current figure.
    plt.clf() 
    # Closes all the figure windows.
    plt.close('all')   
    plt.close(fig)
    gc.collect()

