import os
import sys


def count_cell_for_file(txt_file_path):
	file_out = open(txt_file_path, 'r')
	all_lines = file_out.readlines()
	total  = 0
	for line in  all_lines:
		name, nb_cell = line.split('\t')
		total += int(nb_cell)
	return total

def count_area_for_file(txt_file_path):
	file_out = open(txt_file_path, 'r')
	all_lines = file_out.readlines()
	total  = 0
	for line in  all_lines:
		name, nb_pixels = line.split('\t')
		total += int(nb_pixels)
	return total

def sum_for_each_region(txt_folder, counting_case = 'cell', save_txt = 'CD8.txt'):
	list_files = sorted(list(os.listdir(txt_folder)))
	list_files = [f for f in list_files if '.txt' in f]
	file_all_out = open(save_txt, 'w')
	if counting_case == 'cell_disc':
		for fip in list_files:
			if 'disc=5.9' in fip:
				print(fip)
				name = fip.replace('_disc=5.9.txt','')
				nb_cells = count_cell_for_file(os.path.join(txt_folder, fip))
				file_all_out.write("{}\t{}\n".format(name, nb_cells))
	elif counting_case == 'cell_dist':
		for fip in list_files:
			if 'dist=25' in fip:
				print(fip)
				name = fip.replace('_dist=25.txt','')
				nb_cells = count_cell_for_file(os.path.join(txt_folder, fip))
				file_all_out.write("{}\t{}\n".format(name, nb_cells))
	elif counting_case == 'area':
		for fip in list_files:
			if 'area' in fip:
				print(fip)
				name = fip.replace('_area.txt','')
				st_area = count_area_for_file(os.path.join(txt_folder, fip))
				file_all_out.write("{}\t{}\n".format(name, st_area))
	elif counting_case == 'connected_region':
		for fip in list_files:
			if 'connected_region' in fip:
				print(fip)
				name = fip.replace('_connected_region.txt','')
				st_area = count_area_for_file(os.path.join(txt_folder, fip))
				file_all_out.write("{}\t{}\n".format(name, st_area))
	file_all_out.close()

if __name__ == '__main__':

    # Check the method to count for each region
    #
    marker = 'FOXP3'
    txt_folder = 'G:/Fanny/WSI_tiles/{}/plafrim/txt_prediction_2'.format(marker)
    # For cells counting
    ct_cell = 'cell_disc' # counting_case = 'cell_disc', 'cell_dist' or 'area'
    all_region_txt = 'G:/Fanny/WSI_tiles/{}/{}_all_regions_{}_prediction_2024_2nd.txt'.format(marker, marker,ct_cell)
    sum_for_each_region(txt_folder, counting_case = ct_cell, save_txt = all_region_txt)

    ct_cell = 'cell_dist' # counting_case = 'cell_disc', 'cell_dist' or 'area'
    all_region_txt = 'G:/Fanny/WSI_tiles/{}/{}_all_regions_{}_prediction_2024_2nd.txt'.format(marker, marker,ct_cell)
    sum_for_each_region(txt_folder, counting_case = ct_cell, save_txt = all_region_txt)

    # For area
    ct_area = 'area' # counting_case = 'cell_disc', 'cell_dist' or 'area'
    all_region_txt_area = 'G:/Fanny/WSI_tiles/{}/{}_all_regions_{}_prediction_2024_2nd.txt'.format(marker, marker,ct_area)
    sum_for_each_region(txt_folder, counting_case = ct_area, save_txt = all_region_txt_area)

    # For connected region
    ct_region = 'connected_region' # counting_case = 'cell_disc', 'cell_dist' or 'area'
    all_region_txt_region = 'G:/Fanny/WSI_tiles/{}/{}_connected_regions_{}_prediction_2024_2nd.txt'.format(marker, marker,ct_region)
    sum_for_each_region(txt_folder, counting_case = ct_region, save_txt = all_region_txt_region)
    print('Finish!')
    
    
