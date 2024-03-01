from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
#==========================dataset load==========================

class Rescale(object):
	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label, name = sample['image'],sample['label'],sample['name']

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'image':img,'label':lbl, 'name':name}

class RandomCrop(object):
	def __init__(self,output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size
	def __call__(self,sample):
		image, label, name = sample['image'], sample['label'], sample['name']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		label = label[top: top + new_h, left: left + new_w]

		return {'image': image, 'label': label, 'name': name}

class ToTensor(object):
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):
		image, label, name = sample['image'], sample['label'], sample['name']

		tmpLbl = np.zeros(label.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)

		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		image = image/np.max(image)
		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.4669)/0.2437
			tmpImg[:,:,1] = (image[:,:,0]-0.4669)/0.2437
			tmpImg[:,:,2] = (image[:,:,0]-0.4669)/0.2437
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.4669)/0.2437
			tmpImg[:,:,1] = (image[:,:,1]-0.4669)/0.2437
			tmpImg[:,:,2] = (image[:,:,2]-0.4669)/0.2437

		tmpLbl[:,:,0] = label[:,:,0]

		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg),
			'label': torch.from_numpy(tmpLbl),
			'name': name}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = io.imread(self.image_name_list[idx])
		name = self.image_name_list[idx].split('/')[-1][:-4]

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]

		sample = {'image':image, 'label':label, 'name':name}

		if self.transform:
			sample = self.transform(sample)

		return sample