import os
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from argparse import ArgumentParser
import torch
import sklearn.metrics as sm
import matplotlib
matplotlib.use('Agg')
import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

try:
    from torchsummary import summary
except ImportError:
    summary = False

from process_data import get_data_whole_image
from unet2d import Unet2d
from loss import Dice_Loss
import torch.distributed as dist
import time

#from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

class STSDelineator(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        
        self.loss_function = Dice_Loss()
        
        self.model = Unet2d(hparams.num_channels, hparams.depth, n_classes=hparams.num_classes, n_base_filters=16, final_activation=False)
        #self.model =  FSDP(self.model)

        ## DataLoader
        (self.train_dataset, self.train_loader, self.valid_dataset, self.valid_loader) = \
            get_data_whole_image(hparams.dataset, "train_data", "train_data_mask", hparams.batch_size, \
                valid_per = hparams.val_per)


    # Delegate forward to underlying model
    def forward(self, x):
        y = self.model(x)
        return y

    def class_weights_of_batch(self, batch):
        x, y = batch
        b_weights = []
        labels = {0: 1., 1: 1.}
        for y_true in y:
            unique_idx, count_idx = np.unique(y_true.view(-1).cpu().numpy().astype(int), return_counts = True)
            for lbl in range(len(unique_idx)):
                labels[unique_idx[lbl]] += count_idx[lbl]

        #weights = [1./ labels[x] for x in labels]

        #total = sum(labels.values())
        #weights = [total/ labels[x] for x in labels]

        value_max = max(labels.values())
        weights = [value_max/ labels[x] for x in labels]
        weights = torch.FloatTensor(weights)
        return weights

    # Train on one batch
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_function(y_pred, y)
        
        tensorboard_logs = {'train_loss':loss}
        self.log('train_loss', loss)
        return {'loss': loss, 'log': tensorboard_logs}

    # Compute advanced metrics
    
    def compute_metrics(self, x, y, y_pred, batch_idx):
        y_sigmoid_pred = torch.sigmoid(y_pred.float())
        preds = torch.round(y_sigmoid_pred).cpu().numpy().reshape(-1)
        #preds = preds.cpu().numpy().reshape(-1)
        targets = y.cpu().numpy().reshape(-1)
        

        score=sm.accuracy_score(targets, preds)        
        f1_score = sm.f1_score(targets, preds)
        # compute the similarity coefficient = size of intersection / size of the union of two label sets.
        iou = sm.jaccard_score(targets, preds)
        mcc_score = sm.matthews_corrcoef(targets, preds)

        # ROC AUC
        fpr, tpr, _ = sm.roc_curve(targets, y_sigmoid_pred.cpu().numpy().reshape(-1))
        auroc = sm.auc(fpr, tpr)

        if batch_idx==0:
            # CM from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
            cm = sm.confusion_matrix(targets, preds)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap = plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set_axis_off()
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], '.2f'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            
            self.logger.experiment.add_figure("confusion_matrix", fig, self.current_epoch)
            
            # Precision Recall Curve
            self.logger.experiment.add_pr_curve('pr_curve', targets, y_sigmoid_pred.cpu().numpy().reshape(-1), self.current_epoch)
            
            # ROC Curve
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auroc)
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonale
            self.logger.experiment.add_figure("ROC_curve", fig, self.current_epoch)
            
        
        return torch.as_tensor(score), torch.as_tensor(f1_score), torch.as_tensor(iou), torch.as_tensor(mcc_score)
    
    # Validate on one batch
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        
        val_loss = self.loss_function(y_pred, y) 

        accuracy, f1_score, iou_coeff, mcc_score  = self.compute_metrics(x, y.cpu(), y_pred.cpu(), batch_idx)

        return {'val_loss': val_loss, 'val_accuracy': accuracy, 'val_f1':f1_score, \
            'iou_coeff':iou_coeff, 'mcc_score':mcc_score}
    

    # Final validation step before next epoch
    
    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        avg_iou_coeff = torch.stack([x['iou_coeff'] for x in outputs]).mean()
        avg_mcc_score = torch.stack([x['mcc_score'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'val_accuracy': avg_accuracy, \
         'val_f1':avg_f1, 'iou_coeff':avg_iou_coeff, 'mcc_score': avg_mcc_score, }

        self.log('val_loss',avg_loss)
        self.log('val_accuracy',avg_accuracy)
        self.log('val_f1',avg_f1)
        self.log('iou_coeff',avg_iou_coeff)
        self.log('mcc_score',avg_mcc_score)

        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    # Setup optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.learning_rate)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr = self.hparams.learning_rate, momentum = 0.9)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7, verbose=True)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        #return [optimizer], [scheduler]
        return [optimizer]
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.0001, type=float)
        parser.add_argument('--batch_size', default=96, type=int)
        parser.add_argument('--img_size', default=224, type=int) # Image size, resized before training (default: 512)
        parser.add_argument('--num_channels', default=3, type=int)
        parser.add_argument('--num_classes', default=1, type=int) # Number of classes (default: 2)
        parser.add_argument('--depth', default=5, type=int)
        parser.add_argument('--loss', default='dice', type=str)# entropy, wdice_ce, gdice_ce, focal_gdice, focal_sdice
        parser.add_argument('--up_mode', default='upconv', type=str) # Method for up pathway (default: upsample, other option: upconv)
        parser.add_argument('--val_per', default=0.2, type=float) # % data for validation
        #parser.add_argument('--add_distance', default=True, type=bool) # Compute the distance channel and add to input
        #parser.add_argument('--class_weight', default=True, type=bool) # Compute the weights for each class (imbalance dataset)
        #parser.add_argument('--dice_weight', default=1., type=float) # Compute the weights for each class (imbalance dataset)
        return parser
    


