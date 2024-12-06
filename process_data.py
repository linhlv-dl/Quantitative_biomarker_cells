import os
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision
import numpy as np
from tqdm import tqdm
import random
import re
import glob
import itertools
from torchvision import transforms
from PIL import Image

'''
This part contains the program to load whole image and create dataset
'''
class IHC_Dataset(Dataset):
    def __init__(self, patients, masks, transform=None):
        self.patient_images = patients
        self.patient_masks = masks
        self.transforms = transform

    def __len__(self):
        return len(self.patient_images)

    def read_img(self, fname):
        return sitk.ReadImage(fname)

    def __getitem__(self, idx):
        patient = self.patient_images[idx]
        mask = self.patient_masks[idx]
        patient_image = Image.open(patient)
        patient_mask = Image.open(mask)
        #sample = {'image': patient_image, 'mask': patient_mask}
            
        if self.transforms is not None:
            patient_image = self.transforms(patient_image)

        mask_transf = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),])
        patient_mask = mask_transf(patient_mask)
        #patient_mask =  torch.from_numpy(np.asarray(patient_mask).astype("bool"))
        return patient_image.float(), patient_mask.float()

def get_data_whole_image(dataset_path,image_folder, mask_folder, batch_size, valid_per = 0.2):
    root_dir = os.path.expanduser(dataset_path)
    img_dir = os.path.join(root_dir, image_folder)
    mask_dir = os.path.join(root_dir, mask_folder)

    full_list = sorted(list(os.listdir(img_dir)))
    print(img_dir)
    print(mask_dir)
    #full_masks = sorted(list(os.listdir(mask_dir)))

    num_samples = len(full_list)
    indices = list(range(num_samples))
    split = max(int(np.floor(valid_per * num_samples)), 1)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    base_train_patients = [full_list[i] for i in train_idx]
    train_patients = [os.path.join(img_dir, patienti) for patienti in base_train_patients]
    train_masks = [os.path.join(mask_dir, fimage.replace(".png","_binary.png")) for fimage in base_train_patients]

    base_valid_patients = [full_list[i] for i in valid_idx]
    valid_patients = [os.path.join(img_dir, patienti) for patienti in base_valid_patients]
    valid_masks = [os.path.join(mask_dir, fvimage.replace(".png","_binary.png")) for fvimage in base_valid_patients]

    train_transf = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    valid_transf = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

    # To test the scripts
    # train_patients = train_patients[:200]
    # train_masks = train_masks[:200]
    # valid_patients = train_patients[:100]
    # valid_masks = train_masks[:100]

    train_dataset = IHC_Dataset(train_patients, train_masks, train_transf)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
    
    valid_dataset = IHC_Dataset(valid_patients, valid_masks, valid_transf)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=1)
    
    print('Total training images {}'.format(train_dataset.__len__()))
    print('Total validation images {}'.format(valid_dataset.__len__()))
    
    return (train_dataset, train_loader, valid_dataset, valid_loader)
