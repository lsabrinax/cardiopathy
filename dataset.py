from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class CMRDataset(Dataset):
	"""docstring for CMRDataset"""
	def __init__(self, labelfile='label.txt', transforms=None, dtype='train'):
		# super(CMRDataset, self).__init__()
		with open(labelfile, 'r') as f:
			gt = f.readlines()
		self.gt = gt
		self.label2idx = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
		self.transforms = transforms
		self.dtype = dtype
	
	def __getitem__(self, index):
		line = self.gt[index].strip()
		if self.dtype == 'train':
			imgpath, label = tuple(line.split())
			label = self.label2idx[label]
		imgpath = line
		
		try:
			img = Image.open(imgpath)
		except:
			print('%s has cropted!' % imgpath)
			return self[index+1]
		if self.transforms:
			img = self.transforms(img)
		if self.dtype == 'test':
			label = imgpath
		return img, label

	def __len__(self):
		return len(self.gt)


class FeatureDataset(Dataset):

	def __init__(self, root='training', train=True, test=False):
		self.test = test
		self.root = root
		self.label2idx = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
		path = os.listdir(root)
		path_len = len(path)
		if train:
			self.path = path[:int(path_len*0.8)]
		elif test == False:
			self.path = path[int(path_len*0.8):]
		else:
			self.path = path

	def __getitem__(self, index):
		file = os.path.join(self.root, self.path[index], 'Info.cfg')
		with open(file, 'r') as f:
			lines = f.readlines()
		data = []
		data += float(lines[0].strip().split()[-1])
		data += float(lines[1].strip().split()[-1])

		if self.test:
			data += float(lines[2].strip().split()[-1])
			data += float(lines[4].strip().split()[-1])
			return torch.Tensor(data)
		else:
			data += float(lines[3].strip().split()[-1])
			data += float(lines[5].strip().split()[-1])
			label = self.label2idx[lines[2].strip().split()[-1]]
			return (torch.Tensor(data), label)

	
	def __len__(self):
		return len(self.path)