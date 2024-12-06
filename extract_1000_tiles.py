import os
import sys
import numpy as np
from PIL import Image
import shutil

def process_a_region(txt_file_path, nb_images, tiles_folder, save_folder):
	file_out = open(txt_file_path, 'r')
	all_lines = file_out.readlines()
	all_indexes = []
	all_cells_counts = []
	for line in  all_lines:
		name, nb_cell = line.split('\t')
		index = (name.split('_')[-1]).replace('.png','')
		index = int(index)
		nb_cell = int(nb_cell)
		if nb_cell > 0:
			all_indexes.append(index)
			all_cells_counts.append(nb_cell)
	np_indexes = np.array(all_indexes)
	np_cells = np.array(all_cells_counts)

	# random choose
	if len(np_indexes) > nb_images:
		rd_indices = np.random.choice(len(np_indexes), size = nb_images, replace = False)
		selected_images = np_indexes[rd_indices]
		selected_images_cells = np_cells[rd_indices]
		selected_cells = sum(selected_images_cells)
		remain = 0
		#print(nb_images, len(selected_images), len(selected_images_cells), sum(selected_images_cells))
		# extract and save
		txt_name = (txt_file_path.split('/')[-1]).split('\\')[-1]
		txt_name = txt_name.replace('_disc=3.9.txt','.npz')
		npz_file = txt_name[len(MARKER) + 1:]
		npz_path = os.path.join(tiles_folder, npz_file)
		np_array = np.load(npz_path)['arr_0']
		for index in selected_images:
			arr = np_array[index]
			img_pil = Image.fromarray(arr.astype(np.uint8)).convert('RGB')
			img_pil.save(os.path.join(save_folder, npz_file.replace(' ','_').replace('-','_').replace('.npz','_image_{}.png'.format(index))))
	else:
		selected_images = []
		selected_cells = 0
		remain = nb_images

	return selected_images, selected_cells, remain


def process_all_regions(txt_folder, 
							tiles_folder, 
							total_images = 1000, 
							ffilter = 'disc', 
							save_folder = '/tmp'):
	list_files = sorted(list(os.listdir(txt_folder)))
	list_files = [f for f in list_files if '.txt' in f]
	list_files = [f for f in list_files if ffilter in f]
	nb_images_per_region_ref = round(total_images / len(list_files))

	shutil.rmtree(save_folder)
	os.makedirs(save_folder)
	
	file_all_out = open(LOGS, 'w')
	total_files = 0
	total_cells = 0
	nb_images_per_region = nb_images_per_region_ref
	for fip in list_files:
		list_indexes, total, remain = process_a_region(os.path.join(txt_folder, fip), 
							nb_images_per_region,
							tiles_folder, 
							save_folder)
		total_files += len(list_indexes)
		total_cells += total
		file_all_out.write("{}\t{}\t{}\n".format(fip, list_indexes, total))
		if remain == 0:
			nb_images_per_region = nb_images_per_region_ref
		else:
			nb_images_per_region = nb_images_per_region_ref + remain
		print(fip, total, remain)

	file_all_out.close()
	print("Total extracted png images: ", total_files)
	print("Total stained cells: ", total_cells)
		

if __name__ == '__main__':

    # Check the method to count for each region
    #
    MARKER = 'CD20'

    tiles_folder = "G:/Fanny/WSI_tiles/{}/tiles".format(MARKER)
    txt_folder = 'G:/Fanny/WSI_tiles/{}/txt_prediction_2'.format(MARKER)
    save_folder = 'G:/Fanny/WSI_tiles/{}/png_1K'.format(MARKER)
    LOGS = "G:/Fanny/WSI_tiles/{}/{}_png_1K_log_disc.txt".format(MARKER, MARKER)

    process_all_regions(txt_folder, tiles_folder, total_images = 1000, ffilter = 'disc', save_folder = save_folder)
    print('Finish!')