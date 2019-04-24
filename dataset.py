from torch.utils.data import Dataset
from PIL import Image


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

