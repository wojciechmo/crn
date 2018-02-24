import os
import cv2
import torch
from train import *

result_dir = './data/results'
model_path = './model/crn-final.pt'
cityscapes_test_labels_dir = './cityscapes/gtFine/val'

k_diversed = 2
modules_num = 8
start_shape = (8, 4)
final_shape = (start_shape[0]*2**(modules_num-1), start_shape[1]*2**(modules_num-1))
semantic_palette = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]

def get_semantic_paths(root_dir, first_n = 3):
	
	img_paths = []
	for city_name in os.listdir(root_dir):

		city_path = os.path.join(root_dir, city_name)
		
		city_num = 0
		for img_name in os.listdir(city_path):
			
			img_path = os.path.join(city_path, img_name)
			
			if 'color' not in img_path: continue
			if city_num >= first_n: break
			
			img_paths.append(img_path)
			city_num = city_num + 1
			
	img_paths = sorted(img_paths)
			
	return img_paths

def preprocess_semantic(semantic):
	
	modules_masks = []
				
	for i in range(0, modules_num):
	
		scale = 2**i			
		semantic_resized = cv2.resize(semantic, (start_shape[0]*scale, start_shape[1]*scale), interpolation=cv2.INTER_NEAREST)
	
		class_masks = []
		for p in semantic_palette:
			class_mask = ((semantic_resized[:,:,0] == p[2]) & (semantic_resized[:,:,1] == p[1]) & (semantic_resized[:,:,2] == p[0]))
			class_masks.append(class_mask.astype(np.float32))
		mask = np.stack(class_masks, 0)
			
		modules_masks.append(mask)
		
	return modules_masks

#----------------------------------------------------------------------------------------------
#--------------------------------------- evaluating -------------------------------------------	
#----------------------------------------------------------------------------------------------

if __name__ == "__main__":

	generator = torch.load(model_path)

	semantic_paths = get_semantic_paths(cityscapes_test_labels_dir)
	
	for semantic_path in semantic_paths:
		
		scene_name = os.path.splitext(os.path.basename(semantic_path))[0]
		
		semantic = cv2.imread(semantic_path)	
		pyramide_masks = preprocess_semantic(semantic)
		semantic = cv2.resize(semantic, final_shape, interpolation=cv2.INTER_NEAREST)

		for i in range(len(pyramide_masks)):
			pyramide_masks[i] = torch.autograd.Variable(torch.from_numpy(pyramide_masks[i]).cuda().unsqueeze(0))
		
		cv2.imwrite(os.path.join(result_dir, scene_name + '-semantic.png'), semantic)
			
		raw_output = generator(pyramide_masks)
		
		raw_output = raw_output.data.cpu().numpy()[0].transpose(1, 2, 0)
		
		synthesized_imgs = []

		for i in range(k_diversed):
			
			synthesized_imgs.append(raw_output[:,:,i*3:(i+1)*3])
			
		for i, synthesized_img in enumerate(synthesized_imgs):

			synthesized_img = np.minimum(np.maximum(synthesized_img, 0.0), 255.0)
			
			print 'generated scene saved:', scene_name
			cv2.imwrite(os.path.join(result_dir, scene_name + '-scene-' + str(i+1) + '.png'), synthesized_img.astype(np.uint8))
	
'''
cityscapes_test_labels_dir = './cityscapes/gtFine/val'
result_dir = './data/result_mask'

semantic_paths = get_semantic_paths(cityscapes_test_labels_dir)
for semantic_path in semantic_paths:
	
	scene_name = os.path.splitext(os.path.basename(semantic_path))[0]
	
	semantic = cv2.imread(semantic_path)
	
	pyramide_masks = preprocess_semantic(semantic)
	
	semantic = cv2.resize(semantic,(0,0),fx=0.5,fy=0.5)

	cv2.imshow('semantic',semantic)
	
	largest = pyramide_masks[-1]
	
	print largest.shape
	
	for i,l in enumerate(largest):
		cv2.imshow('l',l)
		
		#cvt to color
		#cv2.imwrite(os.path.join('result_dir, scene_name + str(i) + '.png'),l)
	
		key = cv2.waitKey()
		if key==27: sys.exit()

'''


