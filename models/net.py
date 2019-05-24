import torch.nn as nn

class FeatureNet(nn.Module):

	def __init__(self, inchannel, outchannel):
		super(FeatureNet, self).__init__()
		self.features = nn.Sequential(
			nn.Linear(inchannel, 5),
			nn.BatchNorm1d(5),
			nn.Sigmoid(),
			# nn.Linear(10, 5),
			# nn.ReLU(inplace=True),
			nn.Linear(5, outchannel),
		)

	def forward(self, x):
		x = self.features(x)
		return x