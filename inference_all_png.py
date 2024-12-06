import sys
import os
import process_data as prd
import torch
from unet2d import Unet2d
import numpy as np
from torchvision import transforms
from PIL import Image
from cells_count import *
import shutil

def segment(model, patient_dataset, batch_size=1):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    #exam = patient.exam if exam_num == 0 else patient.exam1

    num_images = len(patient_dataset)
    num_batches = (num_images // batch_size) + 1
    list_of_predicts = []
    list_of_probs = []
    for i in range(num_batches):
        start_ind = i*batch_size
        end_ind = min((i+1)*batch_size, num_images)
        
        data = [ patient_dataset[idx] for idx in range(start_ind, end_ind) ]
        imgs = [ t for t, _ in data]
        if len(imgs) > 0:
            # If we would like transform image before prediction
            imgs = torch.stack(imgs, 0)
            
            imgs = imgs.to(device)
            out = model(imgs)
            #print("Predict by model")
            if not model.final_activation:
                out=torch.sigmoid(out)

            bin_preds=torch.round(out).byte()
            for i in range(0, bin_preds.size()[0]):
                bin_pred = bin_preds[i].squeeze()
                bin_pred_np = bin_pred.cpu().detach().numpy()
                list_of_predicts.append(bin_pred_np)
                list_of_probs.append(out.cpu().squeeze().detach().numpy())
            #print("Finish a batch")
        
    return list_of_predicts, list_of_probs

def load_model(chk_path):
    model = Unet2d(3, 5, n_classes=1, n_base_filters=16, final_activation=False)
    print("Loading model from {}".format(chk_path))
    checkpoint = torch.load(chk_path)
    for key in list(checkpoint['state_dict'].keys()):
        new_key = key[6:]
        checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)
        
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def predict_images_in_folder(in_folder, model, save_folder):
    difiles=os.listdir(in_folder)
    difiles.sort()
    test_patients = [os.path.join(in_folder, ftest) for ftest in difiles]
    test_transf = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])
    patient_dataset = prd.IHC_Dataset(test_patients, test_patients, test_transf)
    patient_bin_predictions, list_of_probs = segment(model, patient_dataset, batch_size = 16)
    for idx in range(len(difiles)):
    	# save binary to image
        save_path = os.path.join(save_folder, difiles[idx])
        image = Image.fromarray(patient_bin_predictions[idx].astype(np.uint8)*255)
        image.save(save_path)

        # save the probability maps to npz file
        #save_npz = os.path.join(SAVE_PROB, difiles[idx].replace('.png', '.npz'))
        #np.savez_compressed(save_npz, list_of_probs[idx])
        
def npz_to_png(npz_folder, save_png):
    list_files = sorted(list(os.listdir(npz_folder)))
    list_files = [f for f in list_files if '.npz' in f]
    list_files = list_files[:1]
    for fip in list_files:
        print(fip)
        np_patient = np.load(os.path.join(npz_folder, fip))['arr_0']
        n_images = np_patient.shape[0]
        for idx in range(n_images):
            img_idx = np_patient[idx]
            # Export image to png
            img_pil = Image.fromarray(img_idx.astype(np.uint8)).convert('RGB')
            img_pil.save(os.path.join(save_png, fip.replace(' ','_').replace('-','_').replace('.npz','_image_{}.png'.format(idx))))
    print("Done!")        

def count_cells(bin_map_folder, output_file = 'CD8'):
    pick_folder_bin(bin_map_folder, txt_output_name = output_file, pick = "disc")
    pick_folder_bin(bin_map_folder, txt_output_name = output_file, pick = "dist")
    area_pixels_count(bin_map_folder, save_file=output_file)
    connected_region(bin_map_folder, remove_size = 30, save_file = output_file)


def inference_all(npz_folder, save_png_folder, unet_model, save_bin_folder, save_txt_prefix):
    list_files = sorted(list(os.listdir(npz_folder)))
    list_files = [f for f in list_files if '.npz' in f]
    #list_files = list_files[:2]
    for fip in list_files:
        print(fip)
        np_patient = np.load(os.path.join(npz_folder, fip))['arr_0']
        n_images = np_patient.shape[0]
        for idx in range(n_images):
            img_idx = np_patient[idx]
            # Export image to png
            img_pil = Image.fromarray(img_idx.astype(np.uint8)).convert('RGB')
            img_pil.save(os.path.join(save_png_folder, fip.replace(' ','_').replace('-','_').replace('.npz','_image_{}.png'.format(idx))))
        print("Predicting....")
        # predict_images_in_folder(save_png_folder, model, save_bin_folder)
        # save_txt = save_txt_prefix + fip.replace(' ','_').replace('-','_').replace('.npz','')
        # count_cells(save_bin_folder, output_file = save_txt)

        # print('Removing the files ')
        # shutil.rmtree(save_png_folder)
        # shutil.rmtree(save_bin_folder)
        # os.makedirs(save_png_folder)
        # os.makedirs(save_bin_folder)
        inference_png_all(save_png_folder, unet_model, save_bin_folder, save_txt_prefix)

    print("Done!")

def inference_png_all(save_png_folder, unet_model, save_bin_folder, save_txt_prefix):
    predict_images_in_folder(save_png_folder, unet_model, save_bin_folder)
    save_txt = save_txt_prefix
    count_cells(save_bin_folder, output_file = save_txt)

    #print('Removing the files ')
    #shutil.rmtree(save_png_folder)
    #shutil.rmtree(save_bin_folder)
    #os.makedirs(save_png_folder)
    #os.makedirs(save_bin_folder)

if __name__ == '__main__':
    import sys
    marker = 'LYSTO'
    input_npz_folder = "/beegfs/vle/IHC_Fanny/data/{}/npz_folder2".format(marker)
    input_folder = "/beegfs/vle/IHC_Fanny/data/{}/test_png_patient".format(marker)
    root = "/beegfs/vle/IHC_Fanny/Unet/lightning_logs_Unet/default"

    # CD3 training 1
    #version = 'version_2'
    #ckpt_version = version + '/checkpoints/epoch=394-step=31204.ckpt'

    # CD3 training 2
    #version = 'version_5'
    #ckpt_version = version + '/checkpoints/epoch=621-step=18037.ckpt'

    # FOXP3 training 1
    #version = 'version_6'
    #ckpt_version = version + '/checkpoints/epoch=665-step=15983.ckpt'

    # Train on LYSTO
    version = 'version_9'
    ckpt_version = version + '/checkpoints/epoch=486-step=6330.ckpt'

    chk_path = os.path.join(root, ckpt_version)
    
    save_folder = "/beegfs/vle/IHC_Fanny/data/{}/pred_test_png".format(marker)
    save_txt = "/beegfs/vle/IHC_Fanny/data/{}/txt_prediction_train_lysto/{}_LYSTO_15K_test_".format(marker, marker)
    try:
        #os.makedirs(SAVE_PROB)
        os.makedirs(save_folder)
    except OSError:
        pass
    # To check each step in the inference pipeline
    # # Extract npz to png
    # npz_to_png(input_npz_folder, input_folder)
    # # Predict the stained areas
    # model =load_model(chk_path)
    # predict_images_in_folder(input_folder, model, save_folder)
    # # Count cells
    # count_cells(save_folder, output_file = save_txt)

    # Combine all steps inside a method
    model =load_model(chk_path)
    inference_png_all(input_folder, 
            model, 
            save_folder, 
            save_txt)

    print('Finish!')
    
    