from mxnet import autograd, gluon, init
from mxnet.gluon import nn
from mxnet.gluon.data import DataLoader
import mxnet as mx
import time
import datetime
import os
import logging
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import copy

# ensure dictionary is exist or not, create one if it's not exist
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# define block for Resnet
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def forward(self, X):
        Y = mx.nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return mx.nd.relu(Y + X)

def resnet_block(num_channels, num_residuals, first_block=False):
    block = nn.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            block.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            block.add(Residual(num_channels))
    return block


def acc(output,label):
    # accuracy (scalar): average number of times the prediction is correct
    return (output.argmax(axis=1) == label.astype('float32')).mean().asscalar()

# create pairs of triple images for each images
def make_pairs(data_iter, length, pos_img, batch_size):
    dataset = []
    for epoch in range(length):
        for batch in data_iter:
            for i in range(batch_size):
                j = random.randint(0, batch_size-1)
                while int(batch.label[0][j].asscalar()) == int(batch.label[0][i].asscalar()):
                    j = random.randint(0, batch_size-1)
                # dataset: [[anc1, pos1, neg1], [anc2, pos2, neg2], ...]
                dataset += [[batch.data[0][i].asnumpy(), pos_img[int(batch.label[0][i].asscalar())].asnumpy(), batch.data[0][j].asnumpy()]]
    print(len(dataset))
    #return np.asarray(dataset)
    return mx.nd.array(dataset)

# create a dataset of positive image for each identities
def define_pos(data_iter, length, batch_size):
    pos_img = {}
    for epoch in range(length):
        for batch in data_iter:
            for i in range(batch_size):
                if int(batch.label[0][i].asscalar()) not in pos_img:
                    pos_img[int(batch.label[0][i].asscalar())] = copy.copy(batch.data[0][i])
    # pos_img: {1: img1, 2: img2, ...}
    return pos_img


training_lst_path = os.path.join(sys.argv[1], 'train.lst')
training_rec_path = os.path.join(sys.argv[1], 'train.rec')
training_idx_path = os.path.join(sys.argv[1], 'train.idx')
testing_lst_path = os.path.join(sys.argv[1], 'test.lst')
testing_rec_path = os.path.join(sys.argv[1], 'test.rec')
testing_idx_path = os.path.join(sys.argv[1], 'test.idx')

# To count the number of training and validation set
with open(training_lst_path, "r") as file:
    data = file.readlines()
    training_img_number = len(data)

with open(testing_lst_path, "r") as file:
        data = file.readlines()
        testing_img_number = len(data)

print("Totoal number of training samples = ", training_img_number)
print("Totoal number of testing samples = ", testing_img_number)


training_img_size = 128
training_img_channel = 3               # channel: gray = 1, RGB = 3
training_img_classes = 260             # number of identities
batch_size = 32
dshape = (batch_size, training_img_channel, training_img_size, training_img_size)

epoch_size = training_img_number / batch_size
print("epoch_size = %d" %(epoch_size))

train_dataiter = mx.io.ImageRecordIter(path_imgrec=training_rec_path, path_imgidx=training_idx_path, batch_size=batch_size, data_shape=(training_img_channel, training_img_size, training_img_size), rand_crop=True, rand_mirror=True, shuffle=True, label_width=1)
test_dataiter = mx.io.ImageRecordIter(path_imgrec=testing_rec_path, path_imgidx=testing_idx_path, batch_size=batch_size, data_shape=(training_img_channel, training_img_size, training_img_size), rand_crop=True, rand_mirror=True, label_width=1)

print('defining positive image...')
# Store a positive image for each identities
pos_img_train = define_pos(train_dataiter, int(epoch_size), batch_size)
pos_img_test = define_pos(test_dataiter, int(testing_img_number/batch_size), batch_size)


train_dataiter.reset()
test_dataiter.reset()

print('making training pairs...')
# create pairs of triple images for each entries (images)
data_train = DataLoader(make_pairs(train_dataiter, int(epoch_size), pos_img_train, batch_size), batch_size, shuffle=True)
print('making testing pairs...')
data_test = DataLoader(make_pairs(test_dataiter, int(testing_img_number/batch_size), pos_img_test, batch_size), batch_size)


# use logging: to save the information of model (with date and time)
log_save_dir = '~/lab307/asia_celeb/log/'
ensure_dir(log_save_dir)
model_save_dir = '~/lab307/asia_celeb/model/'
ensure_dir(model_save_dir)
model_save_name = 'resnet'

logging.basicConfig(filename=log_save_dir + model_save_name + datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".log", level=logging.INFO)
root_logger = logging.getLogger()
stdout_handler = logging.StreamHandler(sys.stdout)
root_logger.addHandler(stdout_handler)
root_logger.setLevel(logging.DEBUG)

devs = mx.gpu()

# network
net = nn.Sequential()
net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
        nn.BatchNorm(), 
        nn.Activation('relu'),
        nn.MaxPool2D(pool_size=3, strides=2, padding=1))

net.add(resnet_block(64, 6, first_block=True),
        nn.Dropout(.5),
        resnet_block(128, 8),
        #nn.Dropout(.5),
        resnet_block(256, 12),
        nn.Dropout(.5),
        resnet_block(512, 6))

net.add(nn.GlobalAvgPool2D(), 
        nn.Dense(training_img_classes))

# initialized by Xavier
net.initialize(ctx=devs, init=init.Xavier())
# define triplet loss to be our loss function
triplet_loss = gluon.loss.TripletLoss(margin=0.2)
# create a SGD trainer with lr=0.01
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate':0.01, 'wd':0.001})

print('start training...')
for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time.time()
    # training loop
    for i, batch in enumerate(data_train):
        # get anchor, positive and negative instance
        anc_ins = batch[:, 0].copyto(devs)
        pos_ins = batch[:, 1].copyto(devs)
        neg_ins = batch[:, 2].copyto(devs)
        with mx.autograd.record():                              # record gradient of error
            output = net(anc_ins)                               # forward pass
            pos = net(pos_ins)
            neg = net(neg_ins)
            loss = triplet_loss(output, pos, neg)   # get the loss
        loss.backward()                                         # backpropagation
        trainer.step(batch_size)
        train_loss += mx.nd.mean(loss).asscalar()
        #train_loss += loss.mean().asscalar()
        train_acc += np.sum(np.where(loss.asnumpy() == 0, 1, 0))

    # validation loop
    for i, batch in enumerate(data_test):
        anc_ins = batch[:, 0].copyto(devs)
        pos_ins = batch[:, 1].copyto(devs)
        neg_ins = batch[:, 2].copyto(devs)
        output = net(anc_ins)
        pos = net(pos_ins)
        neg = net(neg_ins)
        loss = triplet_loss(output, pos, neg)
        valid_acc += np.sum(np.where(loss.asnumpy() == 0, 1, 0))

    # save parameters for each epoch
    paramfile = "Resnet-%04d.params" %(epoch)
    net.save_parameters(paramfile)

    print("Epoch {}: loss {:g}, train acc {:g}, test acc {:g}, in {:.1f} sec".format(
            epoch, train_loss/(training_img_number/batch_size), train_acc/training_img_number, valid_acc/testing_img_number, time.time()-tic))
