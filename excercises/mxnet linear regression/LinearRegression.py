from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet.gluon import loss as gloss
from mxnet import gluon
from mxnet import init

class LinearRegression(object):

    def __init__(self, params):
        self.num_inputs = params["num_inputs"]
        self.num_examples = params["num_examples"]
        self.features = nd.random.normal(scale=1, shape=(self.num_examples, self.num_inputs))
        self.labels = nd.dot(self.features, params["true_w"]) + params["true_b"]
        self.labels += nd.random.normal(scale=0.01, shape=self.labels.shape)
        self.num_epochs = params["num_epochs"]
        self.batch_size = params["batch_size"]

    def setData(self):
        self.dataset = gdata.ArrayDataset(self.features, self.labels)
        self.data_iter = gdata.DataLoader(self.dataset, self.batch_size, shuffle=True)

    def setModel(self):
        self.net = nn.Sequential()
        self.net.add(nn.Dense(1))
        self.net.initialize(init.Normal(sigma=0.01))

    def setLoss(self):
        self.loss = gloss.L2Loss()

    def setOptimizer(self):
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': 0.03})

    def run(self):
        self.setData()
        self.setModel()
        self.setLoss()
        self.setOptimizer()
        self.train()

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            for X, y in self.data_iter:
                with autograd.record():
                    l = self.loss(self.net(X), y)
                    l.backward()
                self.trainer.step(self.batch_size)
            l = self.loss(self.net(self.features), self.labels)
            print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

if __name__ == "__main__":

    params = {}
    params["num_inputs"] = 5
    params["num_examples"] = 1000
    params["true_w"] = nd.array([2, -3.4, 5, 1, 2])
    params["true_b"] = 4.2
    params["batch_size"] = 10
    params["num_epochs"] = 4

    lrg = LinearRegression(params)
    lrg.run()
