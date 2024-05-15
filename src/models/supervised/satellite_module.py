import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer

class ESDSegmentation(pl.LightningModule):
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}):
        '''
        Constructor for ESDSegmentation class.
        '''
        # call the constructor of the parent class
        
        # use self.save_hyperparameters to ensure that the module will load
        
        # store in_channels and out_channels

        # if the model type is segmentation_cnn, initalize a unet as self.model
        
        # if the model type is unet, initialize a unet as self.model
        
        # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
        
        # initialize the accuracy metrics for the semantic segmentation task

        raise NotImplementedError       
    
    def forward(self, X):
        # evaluate self.model
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        
        # evaluate batch
        
        # calculate cross entropy loss

        # return loss
        raise NotImplementedError
    
    
    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch

        # evaluate batch for validation

        # get the class with the highest probability

        # evaluate each accuracy metric and log it in wandb

        # return validation loss 
        raise NotImplementedError
    
    def configure_optimizers(self):
        # initialize optimizer

        # return optimizer
        raise NotImplementedError