import cv2
import os, sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch import optim

model_path = './model'
images_dir = './data/images'
labels_dir = './data/labels'
vgg19_path = './imagenet-vgg-verydeep-19.mat'

start_shape = (8, 4) # first level of image pyramid
modules_num = 8 # number of refinement modules
semantic_num = 19 # number of classes in input semantic labels
k_diversed = 2 # number of generated diversed images
crn_depths = [1024, 1024, 512, 512, 256, 128, 32, 8] # depths of refinement modules

learning_rate = 1.0e-4
learning_rate_decay = 0.9
learning_rate_epoch_step = 20
batch_size = 2
epoch_num = 500
epoch_save_interval = 100

vgg_layers = \
['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2']

loss_layers = ['conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']

#-----------------------------------------------------------------------------------------------
#--------------------------------- cityscape dataset loader ------------------------------------
#-----------------------------------------------------------------------------------------------

class CityScapeDataset(Dataset):
	
	def __init__(self, images_dir, labels_dir):
		
		self.images = sorted(os.listdir(images_dir))
		self.labels = sorted(os.listdir(labels_dir))
	
		self.images_paths = map(lambda x: os.path.join(images_dir, x), self.images)
		self.labels_paths = map(lambda x: os.path.join(labels_dir, x), self.labels)
		
		if len(self.images_paths) != len(self.labels_paths):
			print 'Different number of images and labels in training dataset.'
			sys.exit()
		
	def __len__(self):
		
		return len(self.images_paths)
		
	def __getitem__(self, idx):
		
		image = cv2.imread(self.images_paths[idx])
		image = image.transpose(2, 0, 1).astype(np.float32)
		image = torch.from_numpy(image)
		
		raw_labels = np.load(self.labels_paths[idx])
		labels_files = sorted(raw_labels.files)
		labels = []
		for label_file in labels_files:
			label=raw_labels[label_file].astype(np.float32)

			labels.append(torch.from_numpy(label))

		return (image, labels)
		
#-----------------------------------------------------------------------------------------------
#--------------------------------- cascaded refinement network ---------------------------------
#-----------------------------------------------------------------------------------------------

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, features,1 ,1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, features,1 ,1), requires_grad=True)
        self.eps = eps

    def forward(self, x):
    
		batch_size = x.size()[0]
		x_view = x.view(batch_size, -1)	
		mean = x_view.mean(1).view(batch_size, 1, 1, 1)
		std = x_view.std(1).view(batch_size, 1, 1, 1)
		
		return self.gamma*(x - mean)/(std + self.eps) + self.beta

class ConvModule(nn.Module):
	
	def __init__(self, channels_in, channels_out, final_module = False):
        
		super(ConvModule, self).__init__()
			
		self.conv1 = nn.Conv2d(channels_in, channels_out, (3, 3), (1, 1), (1, 1), bias=False)
		self.ln1 = LayerNorm(channels_out)
		self.lrelu1 = nn.LeakyReLU(0.1)
		self.conv2 = nn.Conv2d(channels_out, channels_out, (3, 3), (1, 1), (1, 1), bias=False)
		
		self.final_module = final_module
		
		if self.final_module == False:
			self.ln2 = LayerNorm(channels_out)	
			self.lrelu2 = nn.LeakyReLU(0.1)
		else:
			self.conv3 = nn.Conv2d(channels_out, 3*k_diversed, (1, 1), (1, 1))
	
	def forward(self, x):
			
		x = self.conv1(x)
		x = self.ln1(x)
		x = self.lrelu1(x)
		x = self.conv2(x)
		
		if self.final_module == False:
			x = self.ln2(x)
			x = self.lrelu2(x)
		else:
			x = self.conv3(x)

		return x
			
class RefinementModule(nn.Module):
	
	def __init__(self, channels_in, channels_out, first_module=False, final_module=False):
		
		super(RefinementModule, self).__init__()
		
		self.first_module = first_module
		
		if self.first_module == False: 
			self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
			self.convmodule = ConvModule(channels_in + semantic_num, channels_out, final_module)
		else:
			self.convmodule = ConvModule(channels_in, channels_out, final_module)
			
	def forward(self, x, semantic):
		
		if self.first_module == False: 

			x = self.upsample(x)
			x = torch.cat([semantic, x], 1)
		
		x = self.convmodule(x)
		
		return x
				
class ListModule(nn.Module):
	
    def __init__(self, *args):
		
        super(ListModule, self).__init__()
        
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
		
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
            
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
            
        return next(it)

    def __iter__(self):
		
        return iter(self._modules.values())

    def __len__(self):
		
        return len(self._modules)

class Net(nn.Module):
	
	def __init__(self):
		
		super(Net, self).__init__()
		
		refinements_modules = []
		for i in range(modules_num):
					
			if i == 0: in_depth = semantic_num
			else: in_depth = crn_depths[i-1]
			
			out_depth = crn_depths[i]
			first_module = True if i == 0 else False
			final_module = True if i == modules_num-1 else False
				
			rm = RefinementModule(in_depth, out_depth, first_module, final_module)
			refinements_modules.append(rm)
		
		self.refinements_modules = ListModule(*refinements_modules)

	def forward(self, semantics):
		
		x = semantics[0]
		semantic = None
		
		for i in range(modules_num):

			if i>0: 
				semantic = semantics[i]
				
			x = self.refinements_modules[i](x, semantic)

		return x
		
#-----------------------------------------------------------------------------------------------
#------------------------------------- vgg19 network -------------------------------------------
#-----------------------------------------------------------------------------------------------

class VGG19(nn.Module):
	
	def __init__ (self, path):
		
		super(VGG19, self).__init__()
		
		self.bgr_mean = Variable(torch.Tensor([103.939, 116.779, 123.68]).cuda()).view(1, 3, 1, 1)
		
		self.weights = []
		self.biases = []
		
		import scipy.io
		vgg_data = scipy.io.loadmat(path)['layers'][0]
		
		for i, vgg_layer in enumerate(vgg_layers):
			
			if 'conv' in vgg_layer:
				
				weight = vgg_data[i][0][0][0][0][0]
				bias = vgg_data[i][0][0][0][0][1][0]
								
				weight = weight.transpose(3, 2, 0, 1)

				weight = Variable(torch.from_numpy(weight).cuda(), requires_grad=False)
				bias = Variable(torch.from_numpy(bias).cuda(), requires_grad=False)

				self.weights.append(weight)
				self.biases.append(bias)
				
			else:
				
				self.weights.append(None)
				self.biases.append(None)

	def forward(self, x):
		
		x = x - self.bgr_mean
		
		out_layers = []

		for i, vgg_layer in enumerate(vgg_layers):
			
			if 'conv' in vgg_layer:
				
				w = self.weights[i]
				b = self.biases[i]
				x = torch.nn.functional.conv2d(x, weight=w, bias=b, stride=(1, 1), padding=(w.size()[2]/2, w.size()[3]/2))

			if 'relu' in vgg_layer:
				
				x = torch.nn.functional.relu(x)
				
			if 'pool' in vgg_layer:
				
				x = torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2), stride=(2, 2))
			
			if vgg_layer in loss_layers:
				out_layers.append(x)
			
		return out_layers

#-----------------------------------------------------------------------------------------------
#-------------------------------- perceptual diverse loss --------------------------------------
#-----------------------------------------------------------------------------------------------

def diverse_loss(gens_layers, img_layers, mask_layers):
	
	L = len(mask_layers) # number of vgg19 loss layers
	K = len(gens_layers) # number of generated diversed images
	
	lambdas = [] # scaling parameters for vgg layers
	num_elements = [] # number of elements in each vgg layer
	for img_layer in img_layers:
		num_elem = np.prod(img_layer.size()[1:])
		num_elements.append(num_elem)
		lambdas.append(1.0/num_elem) 
	
	norms_gens_img = [] # L1 norms for each diversed image and each vgg19 layer
	for k in range(K):
		norms_gen_img = [] # L1 norms for vgg19 layers
		for l in range(L): 
			norm_gen_img = (gens_layers[k][l] - img_layers[l]).abs()
			norms_gen_img.append(norm_gen_img)
		norms_gens_img.append(norms_gen_img)
	
	losses_gens = [] # vgg19 layers losses calculated separately for each diversed image, losses for each batch kept separated
	for k in range(K):
		losses_gens_for_gen_img = []
		for l in range(L):
			norm_gen_img = norms_gens_img[k][l].view(-1, num_elements[l])
			losses_gens_for_gen_img.append(norm_gen_img.sum(1)*lambdas[l])
		losses_gens.append(losses_gens_for_gen_img)

	loss_gens = [] # list of losses for each diversed image calculated as sum over vgg19 layers, losses for each batch kept separated
	for k in range(K):
		loss_gens.append(sum(losses_gens[k]).view(1, -1))
		
	loss_gens = torch.cat(loss_gens) # k_diversed rows, batch_size cols
	
	batch_losses = torch.min(loss_gens, dim=0)[0] # losses for batches, minimum for diversed images calculated independently for each sample
	
	loss = batch_losses.sum() # final loss

	return loss
	
#----------------------------------------------------------------------------------------------
#----------------------------------------- training -------------------------------------------	
#----------------------------------------------------------------------------------------------

def make_dir(directory):

	if not os.path.exists(directory):
		os.makedirs(directory)

if __name__ == "__main__":
	
	make_dir(model_path)
	
	net = Net().cuda()
	vgg19 = VGG19(vgg19_path).cuda()

	optimizer = optim.Adam(net.parameters(), lr=learning_rate)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=learning_rate_epoch_step, gamma=learning_rate_decay)

	dataset = CityScapeDataset(images_dir, labels_dir)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	
	print 'Start of training.'
	
	for epoch_iteration in range(epoch_num):
		
		print 'epoch iteration:', epoch_iteration, 'learning_rate:', ("%01.03e" % lr_scheduler.get_lr()[0])

		for iteration, data in enumerate(dataloader):

			image, label = data

			for i in range(len(label)):
				label[i] = Variable(label[i].cuda())
			image = Variable(image.cuda())

			gen_images_concat = net(label)

			gen_images = []
			for i in range(k_diversed):
				gen_images.append(gen_images_concat[:,i*3:(i+1)*3,:,:])
				
			image_vgg19_layers = vgg19(image)

			gen_images_vgg19_layers = []
			for gen_image in gen_images:
				gen_images_vgg19_layers.append(vgg19(gen_image))

			mask_vgg19_layers = []
			for i in range(-len(image_vgg19_layers), 0): # take as many pyramid layers from mask as needed for loss
				mask_vgg19_layers.append(label[i])

			loss = diverse_loss(gen_images_vgg19_layers, image_vgg19_layers, mask_vgg19_layers)

			print 'iteration:', iteration, 'batch loss:', loss.data.cpu().numpy()[0]

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		
		lr_scheduler.step()
		
		if (epoch_iteration+1) % epoch_save_interval == 0:	
			torch.save(net, model_path + '/crn-epoch-' + str(epoch_iteration) + '.pt')
		
	torch.save(net, model_path + '/crn-final.pt')

	print 'End of training.'
