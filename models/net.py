import torch.nn as nn

class FeatureNet(nn.Module):

	def __init__(self, inchannel, outchannel):
		super(FeatureNet, self).__init__()
		self.features = nn.Sequential(
			nn.Linear(inchannel, 10),
			nn.ReLU(inplace=True),
			nn.Linear(10, outchannel),
			nn.Softmax(dim=1)
		)

	def forward(self, x):
		x = self.features(x)
		return x