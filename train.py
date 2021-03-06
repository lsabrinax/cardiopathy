from torchvision.models import resnet34
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torchnet.meter as meter
import torch.nn as nn
import torch.optim as optim
import dataset
import visdom
import utils
import torch
import torch.backends.cudnn as cudnn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--labelfile', required=True)
parser.add_argument('--type', required=True, help='train or test')
parser.add_argument('--pretrained', default='')

opt = parser.parse_args()

model = resnet34(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
# nn.init.constant_(model.features[0].bias, 0)
model.fc = nn.Linear(in_features=512, out_features=5, bias=True)

if opt.type == 'test':
	print('load pretrained model from %s' % opt.pretrained)
	model.load_state_dict(torch.load(opt.pretrained))
transform = T.Compose([
	T.Resize(224),
	T.CenterCrop(224),
	T.ToTensor(),
	T.Normalize(mean=[0.5], std=[0.5])
])


idx2label = {0: 'NOR', 1: 'MINF', 2: 'DCM', 3: 'HCM', 4: 'RV'}
data_set = dataset.CMRDataset(labelfile=opt.labelfile, transforms=transform, dtype=opt.type)
dataloader = DataLoader(data_set, batch_size=32, shuffle=True, num_workers=2)
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
loss_meter = utils.averager()

use_gpu = torch.cuda.is_available()
if use_gpu:
	model.cuda()
	cudnn.benchmark = True
	criterion.cuda()


def train():
	model.train()
	vis = visdom.Visdom(env='CMR', port=8893)
	mes = ''
	
	for epoch in range(20):
		loss_meter.reset()

		for i, (img, label) in enumerate(dataloader):
			if use_gpu:
				img = img.cuda()
				label = label.cuda()
			pred = model(img)
			loss = criterion(pred, label)
			model.zero_grad()
			loss.backward()
			optimizer.step()

			loss_meter.add(loss)
			if (i+1) % 10 == 0:
				vis.line(X=torch.Tensor([i+1]), Y=loss.data.view(-1), win='train_loss', update='append' if i > 10 else  None, opts={'title': 'train_loss'})

			if (i+1) % 100 == 0:
				mes += '[%d/%d][%d/%d] loss: %f<br>' % (epoch+1, 20, i+1, len(dataloader), loss_meter.val())
				loss_meter.reset()
				vis.text(mes, win='text', opts={'title': 'display_message'})
		torch.save(model.state_dict(), 'expr/restnet34_%d.pth' % epoch)

def test():
	model.eval()
	# nsample = 0
	for img, imgpath in dataloader:
		# nsample += img.size(0)
		if use_gpu:
			img = img.cuda()
			pred = model(img)
			prob = nn.functional.softmax(pred, dim=1)
			res, index = torch.topk(prob, 1)
			with open('pred.txt', 'a') as f:
				for path, idx in zip(imgpath, index):
					label = idx2label[idx.item()]
					f.write(path+' '+label+'\n')

if __name__ == '__main__':
	if opt.type == 'train':
		train()
	elif opt.type == 'test':
		test()

