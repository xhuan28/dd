#!/usr/bin/env python

import numpy as np
import sys
import random
import os
caffe_root = '/home/xiaojun/xinyao/objdet/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import tempfile

#training images for fine-tuning
TRAIN_POS_NUM = 109215
TRAIN_NEG_NUM = 1888002

#validation images
VAL_POS_NUM = 36405
VAL_NEG_NUM = 629334

data_path = '/home/xiaojun/xinyao/objdet/dataset/'

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2

weights = os.path.join(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
assert os.path.exists(weights)

niter = 10000  # number of iterations to train

"""
Prapare dataset for training and testing
"""
def prepare_dataset(data_path, train_pos_num, train_neg_num, val_pos_num, val_neg_num):
    os.chdir(data_path)   
    neg_list = os.listdir(data_path + 'neg')
    pos_list = os.listdir(data_path + 'pos')
    neg_sample = random.sample(neg_list, train_neg_num+val_neg_num)
    pos_sample = random.sample(pos_list, train_pos_num+val_pos_num)
        
    with open('val.txt', 'w') as f:
        for i in range(val_pos_num):
            f.write(data_path + 'pos/' + pos_sample[i] + ' ' + '1' + '\n')
        for i in range(val_neg_num):
            f.write(data_path + 'neg/' + neg_sample[i] + ' ' + '0' + '\n')
        
    with open('train.txt', 'w') as f:
        for i in range(val_pos_num,train_pos_num+val_pos_num,1):
            f.write(data_path + 'pos/' + pos_sample[i] + ' ' + '1' + '\n')
        for i in range(val_neg_num,train_neg_num+val_neg_num,1):
            f.write(data_path + 'neg/' + neg_sample[i] + ' ' + '0' + '\n')

    print "train.txt and val.txt are saved on: %s" %data_path

"""
Define layer proto
"""
def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)

def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name

def detect_net(train=True, learn_all=False, subset=None,test_file = None):
    if subset is None:
        if train:
        	source = data_path+'train.txt'
        elif test_file is not None:
        	source = test_file
        else:
        	source = data_path+'val.txt'
    transform_param = dict(mirror=train, crop_size=227,
        mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    detect_data, detect_label = L.ImageData(
        transform_param=transform_param, source=source,
        batch_size=50, new_height=256, new_width=256, shuffle=True, ntop=2)
    return caffenet(data=detect_data, label=detect_label, train=train,
                    num_classes=2,
                    classifier_name='fc8_detect',
                    learn_all=learn_all)

def solver(train_net_path, test_net_path=None, base_lr=0.001):
    s = caffe_pb2.SolverParameter()
    
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.
        
    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 100000  # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000
    
    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 1000 
    s.snapshot_prefix = caffe_root + 'models/objdet_caffe/objdet_caffe'
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        print f.name
        return f.name
 
def run_solvers(niter, solvers, disp_interval=10):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2d%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)     
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights

def eval_detect_net(weights, test_iters=1000,test_file = None):
    test_net = caffe.Net(detect_net(train=False,test_file = test_file), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        acc = test_net.forward()['acc']
	accuracy += acc
	print str(it)+' iter:'+' accuracy = '+str(acc)
    accuracy /= test_iters
    return test_net, accuracy


if __name__ == '__main__':
   
    caffe.set_mode_gpu()

    #1.Prepare Training Data and Testing Data
    prepare_dataset(data_path, TRAIN_POS_NUM, TRAIN_NEG_NUM, VAL_POS_NUM, VAL_NEG_NUM)
    
    #2.Fine tune detection network
    # 1st round training
    # Retrain the last layer and use the trainted weights as the initial weights for fine tunning the whole CNN
    detect_solver_filename = solver(detect_net(train=True))
    detect_solver = caffe.get_solver(detect_solver_filename)
    # Initialize with CaffeNet weights
    detect_solver.net.copy_from(weights)

    print '#1: Running solvers for %d iterations...' % niter
    solvers = [('pretrained', detect_solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print '1st round training is Done.'

    train_loss = loss['pretrained']
    train_acc = acc['pretrained']
    detect_weights = weights['pretrained']

    # 2nd round training
    end_to_end_net = detect_net(train=True, learn_all=True)

    # Set base_lr to 1e-3, the same as last time when learning only the classifier.
    # You may want to play around with different values of this or other
    # optimization parameters when fine-tuning.  For example, if learning diverges
    # (e.g., the loss gets very large or goes to infinity/NaN), you should try
    # decreasing base_lr (e.g., to 1e-4, then 1e-5, etc., until you find a value
    # for which learning does not diverge).
    base_lr = 0.001

    detect_solver_filename = solver(end_to_end_net, base_lr=base_lr)
    detect_solver = caffe.get_solver(detect_solver_filename)
    detect_solver.net.copy_from(detect_weights)

    print '#2: Running solvers for %d iterations...' % niter
    solvers = [('pretrained, end-to-end', detect_solver)]
    _, _, finetuned_weights = run_solvers(niter, solvers)
    print '2nd round training is Done.'

    detect_weights_ft = finetuned_weights['pretrained, end-to-end']

    # Delete solvers to save memory.
    del detect_solver, solvers

    #3.Evaluate the trained network
    test_net, accuracy = eval_detect_net(detect_weights_ft)
    print 'Accuracy, finetuned from ImageNet initialization: %3.1f%%' % (100*accuracy, )
