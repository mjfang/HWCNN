# Data preprocessing to ensure that all line images have the same dimensions

from scipy import misc
import numpy as np
import glob
import os
import pickle
from math import floor, ceil

def create_form_to_author_map(schema_file):
	form_to_author = {}
	f = open(schema_file, 'r')
	while True:
	  	line = f.readline()
	  	if line == "":
	  		break
	  	line = line.split()
	  	form_to_author[line[0]] = line[1]
	return form_to_author

def create_folder_for_preprocessed_images(preprocessed_pngs_folder_name):
	if not os.path.exists(preprocessed_pngs_folder_name):
	    os.makedirs(preprocessed_pngs_folder_name)

def find_largest_image_dimension(png_root_dir):
	if os.path.exists("largest_raw_image_dimensions.txt"):
		f = open("largest_raw_image_dimensions.txt", 'r')
		largest_image_height = int(f.readline())
		largest_image_width = int(f.readline())
	else:
		image_widths = []
		image_heights = []
		largest_image_height = 0
		largest_image_width = 0
		for folder in glob.glob(png_root_dir):
			sub_folder = folder + "/*"
			for author_folder in glob.glob(sub_folder):
				for line_png_file in glob.glob(author_folder + "/*.png"):
					try:
						line_png_matrix = misc.imread(line_png_file)
					except:
						continue
					height, width = line_png_matrix.shape
					image_heights.append(height)
					image_widths.append(width)
					if height > largest_image_height:
						largest_image_height = height
					if width > largest_image_width:
						largest_image_width = width
		widths = open('widths.pkl', 'wb')
		pickle.dump(image_widths, widths)
		heights = open('heights.pkl', 'wb')
		pickle.dump(image_heights, heights)
	return largest_image_height, largest_image_width

def pad_original_png_images(png_root_dir, preprocessed_pngs_folder, largest_height, largest_width, form_to_author_map):
	for folder in glob.glob(png_root_dir):
		sub_folder = folder + "/*"
		for form_folder_path in glob.glob(sub_folder):
			for line_png_file in glob.glob(form_folder_path + "/*.png"):
				try:
					line_png_matrix = misc.imread(line_png_file)
				except:
					continue
				binary = line_png_matrix > 200
				line_png_matrix[binary] = 255
				height, width = line_png_matrix.shape
				padded_image = line_png_matrix
				if height < 4 or width < 4:
					continue						
				if height <= largest_height:
					height_padding_needed = largest_height - height
					pad_width_top_bottom = (int(floor(height_padding_needed/2)), int(ceil(height_padding_needed - (height_padding_needed/2))))
					padded_image = np.pad(line_png_matrix, (pad_width_top_bottom, (0,0)), 'constant', constant_values=255)
				else:
					padded_image = padded_image[0:largest_height,]
				if width <= largest_width:	
					width_padding_needed = int(largest_width - width)
					pad_width_left_right = (0, width_padding_needed)
					padded_image = np.pad(line_png_matrix, ((0,0), pad_width_left_right), 'constant', constant_values=255)
				else:
					padded_image = padded_image[:, 0:largest_width]
				padded_image = misc.imresize(padded_image,0.25)
				save_preprocessed_image(padded_image, line_png_file, form_to_author_map, preprocessed_pngs_folder)

def save_preprocessed_image(image_array, line_png_file, form_to_author_map, preprocessed_pngs_folder):
	form_name = line_png_file.split("/")[-2]
	file_name = line_png_file.split("/")[-1]
	author_id = form_to_author_map[form_name]
	destination_folder = preprocessed_pngs_folder + "/" + author_id
	if not os.path.exists(destination_folder):
	    os.makedirs(destination_folder)
	destination = destination_folder + "/" + file_name
	misc.imsave(destination, image_array)

preprocessed_pngs_folder = "preprocessed_words"
schema_file = "forms.txt"
raw_images_root_dir = "words/*"
form_to_author_map = create_form_to_author_map(schema_file)
create_folder_for_preprocessed_images(preprocessed_pngs_folder)
largest_height, largest_width = find_largest_image_dimension(raw_images_root_dir)
pad_original_png_images(raw_images_root_dir, preprocessed_pngs_folder, largest_height, largest_width, form_to_author_map)

