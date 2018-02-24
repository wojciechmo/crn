import os
import cv2
import numpy as np

train_images_dir = './data/images'
train_labels_dir = './data/labels'
cityscapes_images_dir = './cityscapes/leftImg8bit/train'
cityscapes_labels_dir = './cityscapes/gtFine/train'

modules_num = 8
start_shape = (8, 4)
final_shape = (start_shape[0]*2**(modules_num-1), start_shape[1]*2**(modules_num-1))
palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def process_dataset(in_dir, out_dir, choice):

	for city_name in os.listdir(in_dir):

		city_path = os.path.join(in_dir, city_name)
		
		for img_name in os.listdir(city_path):
			
			in_img_path = os.path.join(city_path, img_name)
			
			if choice == 'image':
				
				out_img_path = os.path.join(out_dir, img_name)
							
				img = cv2.imread(in_img_path)
				img = cv2.resize(img, final_shape, interpolation=cv2.INTER_LINEAR)
				cv2.imwrite(out_img_path, img)

			elif choice == 'label':
				
				if 'color' not in in_img_path: continue
				
				label_name = os.path.splitext(img_name)[0]
				out_label_path = os.path.join(out_dir, label_name)

				semantic_img = cv2.imread(in_img_path)	
				modules_masks = []
				
				for i in range(0, modules_num):
	
					scale = 2**i			
					semantic_img_resized = cv2.resize(semantic_img, (start_shape[0]*scale, start_shape[1]*scale), interpolation=cv2.INTER_NEAREST)
	
					class_masks = []
					for p in palette:
						class_mask = ((semantic_img_resized[:,:,0] == p[2]) & (semantic_img_resized[:,:,1] == p[1]) & (semantic_img_resized[:,:,2] == p[0]))
						class_masks.append(class_mask)
						
					mask = np.stack(class_masks, 0)
					modules_masks.append(mask)
					
				np.savez(out_label_path, *modules_masks)
				
if __name__ == "__main__":
				
	process_dataset(cityscapes_images_dir, train_images_dir, 'image')
	process_dataset(cityscapes_labels_dir, train_labels_dir, 'label')
