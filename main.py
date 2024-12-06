import sys
import os

import math
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import torch.distributed as dist
import time
import logging
from model_pl import STSDelineator
from pytorch_lightning import loggers as pl_loggers

SEED = 2452
torch.manual_seed(SEED)
np.random.seed(SEED)


def main(hparams):
    print(hparams)
    torch.cuda.empty_cache()
    model = STSDelineator(hparams)
    estop_callback = EarlyStopping(monitor = 'val_loss', mode ='min', min_delta = 0.0000, patience = 20, verbose = True)
    chp_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min",)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir = 'lightning_logs_Unet')

    print("New training on {} epochs".format(hparams.epochs))
    trainer = pl.Trainer(max_epochs=hparams.epochs, \
                            weights_summary='top', \
                            num_nodes = hparams.n_nodes, \
                            gpus = hparams.n_gpus, \
                            #accelerator= "ddp", \
                            #amp_level = "O2", \
                            precision = 16, \
                            #accumulate_grad_batches = 3, \
                            callbacks = [chp_callback, estop_callback],
                            logger = tb_logger) 

    # print("Training from checkpoint to ", hparams.epochs)
    # ck_point_path = '/beegfs/vle/IHC_Fanny/Unet/lightning_logs_Unet/default/version_0/checkpoints/epoch=797-step=31121.ckpt'
    # trainer = pl.Trainer(resume_from_checkpoint = ck_point_path, \
    #                         max_epochs=hparams.epochs, \
    #                         weights_summary='top', \
    #                         num_nodes = hparams.n_nodes, \
    #                         gpus = hparams.n_gpus, \
    #                         #accelerator= "ddp", \
    #                         #amp_level = "O2", \
    #                         precision = 16, \
    #                         #plugins='deepspeed_stage_3_offload',
    #                         #accumulate_grad_batches = 3,
    #                         callbacks = [chp_callback, estop_callback],
    #                         logger = tb_logger)
        
    trainer.fit(model)


if __name__ == '__main__':
    
    #os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    main_arg_parser = argparse.ArgumentParser(description="parser for observation generator", add_help=False)
    main_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    main_arg_parser.add_argument("--checkpoint-interval", type=int, default=2000,
                                  help="number of batches after which a checkpoint of the trained model will be created")
    main_arg_parser.add_argument('--epochs', default = 1500, type=int) 
    main_arg_parser.add_argument('--n_nodes', default = 5, type=int)
    main_arg_parser.add_argument('--n_gpus', default = 2, type=int) 
    main_arg_parser.add_argument('--dataset', default='/beegfs/vle/IHC_Fanny/data/LYSTO', type=str) # This folder contains images and masks folders

    
    # add model specific args i
    parser = STSDelineator.add_model_specific_args(main_arg_parser, os.getcwd())
    hyperparams = parser.parse_args()

    main(hyperparams)

