import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer


class ESDSegmentation(pl.LightningModule):
    def __init__(
        self,
        model_type,
        in_channels,
        out_channels,
        learning_rate=1e-3,
        model_params: dict = {},
    ):
        """
        Constructor for ESDSegmentation class.
        """
        # CALL THE CONSTRUCTOR OF THE PARENT CLASS
        super(ESDSegmentation, self).__init__()

        # use self.save_hyperparameters to ensure that the module will load
        self.save_hyperparameters()

        # store in_channels and out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        # if the model type is segmentation_cnn, initalize a unet as self.model
        # if the model type is unet, initialize a unet as self.model
        # if the model type is fcn_resnet_transfer, initialize a fcn_resnet_transfer as self.model
        if model_type == "segmentation_cnn":
            self.model = SegmentationCNN(in_channels, out_channels, **model_params)
        elif model_type == "unet":
            self.model = UNet(in_channels, out_channels, **model_params)
        elif model_type == "fcn_resnet_transfer":
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)
        else:
            print(f"model_type not found: {model_type}")

        # initialize the accuracy metrics for the semantic segmentation task
        self.train_accuracy_metrics = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=out_channels,
            average="macro",
            multidim_average="samplewise",
        )  # not sure the parameters are correct
        self.eval_accuracy_metrics = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=out_channels,
            average="macro",
            multidim_average="samplewise",
        )  # not sure the parameters are correct

    def forward(self, X):
        # evaluate self.model
        return self.model(X)

    def training_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch

        # evaluate batch
        eval = self(sat_img)

        # calculate cross entropy loss
        loss = nn.CrossEntropyLoss()(eval, mask)

        # return loss
        self.log(f"train_loss_{batch_idx}: ", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # get sat_img and mask from batch
        sat_img, mask = batch

        # evaluate batch for validation
        eval = self(sat_img)

        # get the class with the highest probability
        loss = nn.CrossEntropyLoss()(eval, mask)
        eval = torch.argmax(eval, dim=1)

        # evaluate each accuracy metric and log it in wandb
        self.eval_accuracy_metrics(eval, mask)
        self.log(f"eval_loss_{batch_idx}: ", loss)
        self.log("eval_accuracy", self.eval_accuracy_metrics, on_epoch=True)

        # return validation loss
        return loss

    def configure_optimizers(self):
        # initialize optimizer
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        # return optimizer
        return optimizer
