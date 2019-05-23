import visdom
import numpy as np
import time

class Visualizer(object):

    def __init__(self, env='CMR_linear', port=8893, **kwargs):

        self.vis = visdom.Visdom(env=env, port=port, **kwargs)
        self.index = {}
        self.log_text = ''

    def plot(self, name, value):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([value]), X=np.array([x]), win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append')

        self.index[name] = x + 1

    def log(self, info, win='log_text'):

        self.log_text += '[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H_%M_%S'),
            info=info
        )

        self.vis.text(self.log_text, win=win)