import os
import sys
import numpy as np
from PIL import Image
import shutil
from skimage.color import rgb2gray
from skimage import morphology

def local_threshold(image):
    grayscale = rgb2gray(image)
    mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        grayscale < 0.6, 30), 30)
    # Use 0.6 for FOXP3, 0.7 for the others
    mask = morphology.closing(mask, morphology.disk(3))
    return mask

def check_tiles_export_png_and_masks(folder_path, save_png, mask_png, begin_at = 0, stop_at = 1):
    list_npz = os.listdir(folder_path)
    all_png = 0
    for idx in range(len(list_npz)):
        if idx >= begin_at and idx <= stop_at:
            npz = list_npz[idx]
            np_patient = np.load(os.path.join(folder_path, npz))['arr_0']
            print(idx, npz, np_patient.shape)
            all_png += np_patient.shape[0]
            
            n_images = np_patient.shape[0]
            for png_idx in range(n_images):
                img_idx = np_patient[png_idx]
                # Export image to png
                img_pil = Image.fromarray(img_idx.astype(np.uint8)).convert('RGB')
                img_pil.save(os.path.join(save_png, npz.replace(' ','_').replace('-','_').replace('.npz','_image_{}.png'.format(png_idx))))
                # Create binary mask and save
                img_pil = local_threshold(img_idx)
                img_pil = Image.fromarray(img_pil.astype(np.uint8)*255)
                img_pil.save(os.path.join(mask_png, npz.replace(' ','_').replace('-','_').replace('.npz','_image_{}_binary.png'.format(png_idx))))
            
    print("Done!")
    return all_png

if __name__ == '__main__':

    # Check the method to count for each region
    #
    MARKER = 'FOXP3'
    tile_folder = '/beegfs/vle/IHC_Fanny/data/{}/npz_folder'.format(MARKER)
    save_png = '/beegfs/vle/IHC_Fanny/data/{}/png_10patients'.format(MARKER)
    save_mask = '/beegfs/vle/IHC_Fanny/data/{}/png_10patients_masks_CD3'.format(MARKER)
    all_png = check_tiles_export_png_and_masks(tile_folder, 
												save_png, 
												save_mask,
												begin_at = 0,
												stop_at = 15)
    print("Total number of png: ", all_png)
    print('Finish!')