import sys
import os
import process_data as prd
import torch
from unet2d import Unet2d
import numpy as np
from torchvision import transforms
from PIL import Image
import shutil

def segment(model, patient_dataset, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
'''
def post_process_operation(out_image):
    connectFilter = sitk.ConnectedComponentImageFilter()
    connectedImg = connectFilter.Execute(out_image)
    relabel_filter1 = sitk.RelabelComponentImageFilter()
    relabel_image = relabel_filter1.Execute(connectedImg)
    print("Before processing:", relabel_filter1.GetNumberOfObjects())
    largest_size = -1
    if relabel_filter1.GetNumberOfObjects() > 0:
        largest_size = relabel_filter1.GetSizeOfObjectsInPixels()[0]

    if largest_size != -1:
        relabel_filter2 = sitk.RelabelComponentImageFilter()
        relabel_filter2.SetMinimumObjectSize(largest_size - 1)
        relabel_image = relabel_filter2.Execute(connectedImg)
        print("Before processing:", relabel_filter2.GetNumberOfObjects())
    return relabel_image

def to_image_and_save(prediction_list, patient, input_image, save_path, post_process = False):
    img_array = patient.construct_image_by_slices(prediction_list)
    img_array = np.transpose(img_array,(2,1,0))
    out_image = sitk.GetImageFromArray(img_array)
    out_image.SetOrigin(input_image.GetOrigin())
    out_image.SetSpacing(input_image.GetSpacing())
    out_image.SetDirection(input_image.GetDirection())
    out_image = sitk.Cast(out_image, sitk.sitkUInt8)
    if post_process:
        out_image = post_process_operation(out_image)
    sitk.WriteImage(out_image, save_path, True)
    return
'''

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
        
        

if __name__ == '__main__':
    import sys
    marker = 'FOXP3'
    input_folder = "/beegfs/vle/IHC_Fanny/data/{}/png_10patients".format(marker)
    root = "/beegfs/vle/IHC_Fanny/Unet/lightning_logs_Unet/default"
    version = 'version_2'
    ckpt_version = version + '/checkpoints/epoch=394-step=31204.ckpt'
    # FOXP3
    # version = 'version_6'
    # ckpt_version = version + '/checkpoints/epoch=665-step=15983.ckpt'
    chk_path = os.path.join(root, ckpt_version)
    
    #save_folder = "/beegfs/vle/IHC_Fanny/data/prediction_test/bin_maps/" + version
    #SAVE_PROB = "/beegfs/vle/IHC_Fanny/data/prediction_test/prob_maps/" + version
    save_folder = "/beegfs/vle/IHC_Fanny/data/{}/png_10patients_predictions_CD3".format(marker)
    #SAVE_PROB = "/beegfs/vle/IHC_Fanny/data/pred_fanny_1000tiles/prob_maps/" + version
    try:
        #os.makedirs(SAVE_PROB)
        os.makedirs(save_folder)
    except OSError:
        pass

    model =load_model(chk_path)
    predict_images_in_folder(input_folder, model, save_folder)
    shutil.rmtree(input_folder)
    os.makedirs(input_folder)
    print('Finish!')
    
    