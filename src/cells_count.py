import sys
import os
import numpy as np
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

def pick1(bin_map, min_distance = 7):
    distance = ndi.distance_transform_edt(bin_map)
    local_max_coords = feature.peak_local_max(distance, min_distance=min_distance)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)
    segmented_cells = segmentation.watershed(-distance, markers, mask=bin_map)
    return segmented_cells.max()

def pick2(bin_map, diskc = 4):
    smoother = filters.rank.mean(util.img_as_ubyte(bin_map), morphology.disk(diskc))
    cells = measure.label(smoother)
    return cells.max()

def pick_folder_bin(prob_folder, txt_output_name = '21F226_227', pick = "disc"):
    difiles=os.listdir(prob_folder)
    difiles.sort()
    txt_name = txt_output_name
    if pick == 'disc':
        txt_name = txt_name + '_disc=3.9.txt'
    elif pick == 'dist':
        txt_name = txt_name + '_dist=17.txt'
    file_out = open(txt_name, 'w')
    for idx in range(len(difiles)):
        npz_file = difiles[idx]
        # load segmentation image
        img_path = os.path.join(prob_folder, npz_file)
        prob_map = Image.open(img_path)
        prob_map = (np.asarray(prob_map)//255).astype('uint8')
        #print(np.unique(prob_map, return_counts = True))
        if pick == 'disc':
            nb_cells = pick2(prob_map, diskc = 3.9) # disc = 3.9 gives a good result
        elif pick == 'dist':
            nb_cells = pick1(prob_map, min_distance = 17) # distance = 17 gives a good result
        print('{} {} has {} cells'.format(idx, npz_file, nb_cells))
        file_out.write("{}\t{}\n".format(npz_file, nb_cells))
        #if idx > 1:
        #    return
    file_out.close()

def area_pixels_count(bin_folder, save_file = 'save_file'):
	difiles=os.listdir(bin_folder)
	difiles.sort()
	file_out = open(save_file + '_area.txt', 'w')
	for idx in range(len(difiles)):
		npz_file = difiles[idx]
		# load prob_map
		img_path = os.path.join(bin_folder, npz_file)
		prob_map = Image.open(img_path)
		prob_map = np.asarray(prob_map)
		ids, lens = np.unique(prob_map, return_counts = True)
		if len(ids) == 2:
			nb_cells = lens[1]
		else:
			nb_cells = 0
		print('{} {} has {} stained pixels'.format(idx, npz_file, nb_cells))
		file_out.write("{}\t{}\n".format(npz_file, nb_cells))

	file_out.close()

def connected_region(bin_folder, remove_size = 30, save_file = 'save_file'):
    difiles=os.listdir(bin_folder)
    difiles.sort()
    file_out = open(save_file + '_connected_region.txt', 'w')
    for idx in range(len(difiles)):
        npz_file = difiles[idx]
        # load prob_map
        img_path = os.path.join(bin_folder, npz_file)
        prob_map = Image.open(img_path)
        prob_map = np.asarray(prob_map)
        prob_map_bin = prob_map > 0
        # remove small object
        prob_cleaned = morphology.remove_small_objects(prob_map_bin, remove_size)
        labeled_image, nb_cells = measure.label(prob_cleaned, return_num = True)
        print('{} {} has {} stained pixels'.format(idx, npz_file, nb_cells))
        file_out.write("{}\t{}\n".format(npz_file, nb_cells))

    file_out.close()


'''
def pick_centroids(prob_map, min_distance = 20):
    coordinates = peak_local_max(prob_map, min_distance)
    return coordinates

def draw_circle(draw, xy_coord, radius = 1):
    draw.ellipse((xy_coord[0] - radius, xy_coord[1] - radius, xy_coord[0] + radius, xy_coord[1] + radius), fill=(255,0,0), outline = (0,0, 255))

def pick_one_case(input_image, prob_map, min_distance = 20, save_path = None):
    coordinates = pick_centroids(prob_map)
    #print("Number of cells: ", len(coordinates))
    for coord in coordinates:
        draw = ImageDraw.Draw(input_image)
        draw_circle(draw, coord, radius = 2)
        input_image.save(save_path)
    return len(coordinates)


def pick_folder_prob(prob_folder, images_folder, save_folder):
    difiles=os.listdir(prob_folder)
    difiles.sort()
    for idx in range(len(difiles)):
        npz_file = difiles[idx]
        # load prob_map
        npz_path = os.path.join(prob_folder, npz_file)
        prob_map = np.load(npz_path)['arr_0']
        print(prob_map)
        # load image
        img_path = os.path.join(images_folder, npz_file.replace('.npz', '.png'))
        img = Image.open(img_path)

        # pick the max and count cells
        save_path = os.path.join(save_folder, npz_file.replace('.npz', '.png'))
        nb_cells = pick_one_case(img, prob_map, min_distance = 20, save_path = save_path)
        print('{} has {} cells'.format(npz_file, nb_cells))
        if idx > 0:
            break
'''
if __name__ == '__main__':
    import sys

    # input_folder = "/beegfs/vle/IHC_Fanny/data/images_test"
    # prob_map_folder = "/beegfs/vle/IHC_Fanny/Unet/lightning_logs_Unet/default"
    # save_folder = "/beegfs/vle/IHC_Fanny/data/count_cells"
    #input_folder = "G:/Fanny/png_tiles"
    #prob_map_folder = "G:/Fanny/prediction_test/prob_map/version_0"
    #bin_map_folder = "G:/Fanny/WSI_tiles/pred_test_png/version_2/21F245_246"
    bin_map_folder = "G:/Fanny/WSI_tiles/CD20/png_1K_prediction"
    bin_map_folder = "F:/IHC_Fanny/LYSTO_Hackathon/h5files/predictions/pred_test_png_local_green_channel"
    # save_folder = "G:/Fanny/prediction_test/cells_prediction" "/beegfs/vle/IHC_Fanny/data/{}/png_1K_prediction".format(marker)
    # try:
    #     os.makedirs(save_folder)
    # except OSError:
    #     pass
    
    pick_folder_bin(bin_map_folder, txt_output_name = 'txt_counting/LYSTO_green_channel_200_png_1K_prediction_13092024', pick = "disc") # pick = 'disc' or 'dist'
    pick_folder_bin(bin_map_folder, txt_output_name = 'txt_counting/LYSTO_green_channel_200_png_1K_prediction_13092024', pick = "dist")
    #area_pixels_count(bin_map_folder, save_file='21F245_246')
    print('Finish!')
    
    
