import os
import sys
import numpy as np
from PIL import Image
import shutil
import sklearn.metrics as sm

def compute_metrics(mask_folder, pred_folder):
	list_png_masks = sorted(list(os.listdir(mask_folder)))

	nb_images = len(list_png_masks)
	acc_scores = []
	f1_scores = []
	iou_scores = []
	mcc_scores = []
	auc_scores =[]
	batch_size = 512
	num_batches = (nb_images // batch_size) + 1
	for idx in range(num_batches):
		start_ind = idx*batch_size
		end_ind = min((idx+1)*batch_size, nb_images)
		targets = []
		preds = []
		for pidx in range(start_ind, end_ind):
			gt_mask = list_png_masks[pidx]
			pred_mask = gt_mask.replace('_binary.png','.png')

			patient_mask = Image.open(os.path.join(mask_folder, gt_mask))
			patient_mask = np.asarray(patient_mask).astype("int")//255
			patient_pred = Image.open(os.path.join(pred_folder, pred_mask))
			patient_pred = np.asarray(patient_pred).astype("int") //255

			target = patient_mask.reshape(-1)
			pred = patient_pred.reshape(-1)
			targets.append(target)
			preds.append(pred)
		preds = np.asarray(preds).reshape(-1)
		targets = np.asarray(targets).reshape(-1)
		print(np.unique(preds, return_counts = True))
		print(np.unique(targets, return_counts = True))
		#print(preds)
		#print(targets)
		acc_score=sm.accuracy_score(targets, preds)
		f1_score = sm.f1_score(targets, preds)
		iou_score = sm.jaccard_score(targets, preds)
		mcc_score = sm.matthews_corrcoef(targets, preds)
		fpr, tpr, _ = sm.roc_curve(targets, preds, pos_label=1)
		auc_score = sm.auc(fpr, tpr)

		# Save the metrics
		acc_scores.append(acc_score)
		f1_scores.append(f1_score)
		iou_scores.append(iou_score)
		mcc_scores.append(mcc_score)
		auc_scores.append(auc_score)

	avg_acc = sum(acc_scores)/len(acc_scores)
	avg_f1 = sum(f1_scores)/len(f1_scores)
	avg_iou = sum(iou_scores)/len(iou_scores)
	avg_mcc = sum(mcc_scores)/len(mcc_scores)
	avg_auc = sum(auc_scores)/len(auc_scores)

	return avg_acc, avg_f1, avg_iou, avg_mcc, avg_auc

if __name__ == '__main__':
    marker = 'FOXP3'
    mask_folder = "/beegfs/vle/IHC_Fanny/data/{}/png_10patients_masks_CD3".format(marker)
    pred_folder = "/beegfs/vle/IHC_Fanny/data/{}/png_10patients_predictions_CD3".format(marker)
    
    avg_acc, avg_f1, avg_iou, avg_mcc, avg_auc = compute_metrics(mask_folder, pred_folder)
    print("Avg ACC = {}, avg_F1 = {}, avg_IOU = {}, avg_MCC = {}, avg_AUC = {}".format(avg_acc, avg_f1, avg_iou, avg_mcc, avg_auc))
    print('Finish!')
