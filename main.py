from models.net import FeatureNet
import dataset
import fire
import torch as t
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from vis import Visualizer
import os

class Config(object):
    env = 'CMR_linear'
    port = 8893
    batch_size = 5
    lr = 0.01
    model_path = None #模型路径
    max_epoch = 10
    data_root = '../training'
    num_workers = 2
    plot_every = 2
    use_gpu = True
    debug_file = 'tmp/debug'
opt = Config()

def train(**kwargs):
    # opt = Config()
    for k, v in kwargs.items():
        setattr(opt, k, v)
    vis = Visualizer(opt.env, opt.port)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    lr = opt.lr

    #网络配置
    featurenet = FeatureNet(4, 5)
    if opt.model_path:
        featurenet.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    featurenet.to(device)

    #加载数据
    data_set = dataset.FeatureDataset(root=opt.data_root, train=True, test=False)
    dataloader = DataLoader(data_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataset = dataset.FeatureDataset(root=opt.data_root, train=False, test=False)
    val_dataloader = DataLoader(val_dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    #定义优化器和随时函数
    optimizer = t.optim.Adam(featurenet.parameters(), lr)
    criterion = t.nn.CrossEntropyLoss().to(device)

    #计算重要指标
    loss_meter = AverageValueMeter()

    #开始训练
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        for ii, (data, label) in enumerate(dataloader):
            feature = data.to(device)
            target = label.to(device)

            optimizer.zero_grad()
            prob = featurenet(feature)
            # print(prob)
            # print(target)
            loss = criterion(prob, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            if (ii + 1) % opt.plot_every:
                vis.plot('train_loss', loss_meter.value()[0])
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()
        t.save(featurenet.state_dict(), 'checkpoints/%d.pth' % epoch)

        #验证和可视化
        accu, loss = val(featurenet, val_dataloader, criterion)
        featurenet.train()
        vis.plot('val_loss', loss)
        vis.log('epoch: {epoch}, loss: {loss}, accu: {accu}'.format(
            epoch=epoch, loss=loss, accu=accu
        ))

        lr = lr * 0.9
        for param_group in optimizer.param_groups:
            optimizer[param_group] = lr



@t.no_grad()
def val(model, dataloader, criterion):

    model.eval()
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    ncorrect = 0
    nsample = 0
    loss_meter = AverageValueMeter()
    loss_meter.reset()
    for ii, (data, label) in enumerate(dataloader):
        nsample += data.size()[0]
        feature = data.to(device)
        target = label.to(device)
        prob = model(feature)
        loss = criterion(prob, target)
        score = t.nn.functional.softmax(prob, dim=1)
        index = score.topk(1)[1].view(-1)
        loss_meter.add(loss.item())
        ncorrect += (index == target).cpu().sum().item()

    accu = float(ncorrect) / nsample * 100
    loss = loss_meter.value()[0]
    return accu, loss

if __name__ == '__main__':
    fire.Fire()
