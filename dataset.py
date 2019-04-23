from torch.utils.data import Dataset
from PIL import Image


class CMRDataset(Dataset):
	"""docstring for CMRDataset"""
	def __init__(self, labelfile='label.txt', transforms=None):
		# super(CMRDataset, self).__init__()
		with open(labelfile, 'r') as f:
			gt = f.readlines()
		self.gt = gt
		self.label2idx = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}
		self.transforms = transforms
	
	def __getitem__(self, index):
		line = self.gt[index]
		imgpath, label = tuple(line.strip().split())
		try:
			img = Image.open(imgpath)
		except:
			print('%s has cropted!' % imgpath)
			return self[index+1]
		if self.transforms:
			img = self.transforms(img)
		label = self.label2idx[label]
		return img, label

	def __len__(self):
		return len(self.gt)

