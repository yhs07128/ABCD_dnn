import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
import pickle
import pandas as pd
from tensorflow.python.keras.backend import dtype
tfk = tf.keras
tfd = tfp.distributions

def normal_sp(params): 
    #return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.1 * params[:,1:2]))# both parameters are learnable
    return tfd.Normal(loc=params[:,0:1], scale=tf.nn.sigmoid(params[:,1:2]))# both parameters are learnable

def NLL(y, distr): 
    return -distr.log_prob(y)

class ExtendedABCD(tfk.Model):
    def __init__(self, ncrvars, batchsize, mode=1):
        super(ExtendedABCD, self).__init__()
        assert ncrvars>0, f'Number of control variables is currently {ncrvars}. It should be greater than 0'
        self.ncrvars = ncrvars
        assert batchsize>0, f'batch size is currently {batchsize}. It should be > 0'
        self.batchsize = batchsize
        assert mode>=1 and mode<=3, "wrong mode. mode should be between 1 and 3. Stopping!"
        self.model = self.createmodel(mode)

    def createmodel(self, mode):
        inputshape = (self.ncrvars,)
        netin = tfk.layers.Input(shape=inputshape)
        nrows = tf.shape(netin)[0]
        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / self.batchsize
        bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / self.batchsize
        outerproduct = tf.einsum('bi,bj->bij', netin, netin)
        outerproduct = tf.reshape(outerproduct, (nrows,4))
        mergedinput = tf.concat([netin, outerproduct], axis=1)
        if mode==1:
            #net= tfk.layers.Dense(32, activation=tf.nn.relu )(mergedinput)
            net= mergedinput
            netout = tfk.layers.Dense(1)(net)
            name = 'Extended ABCD'
        else:
            if mode==2:
                #meannet = mergedinput
                meannet = tfk.layers.Dense(32, activation=tf.nn.relu)(mergedinput)
                widthnet = tfk.layers.Dense(32, activation=tf.nn.relu)(mergedinput)
                param0 = tfp.layers.DenseFlipout(1, kernel_divergence_fn=kernel_divergence_fn, bias_divergence_fn=bias_divergence_fn)(meannet)
                param1 = tfp.layers.DenseFlipout(1, kernel_divergence_fn=kernel_divergence_fn, bias_divergence_fn=bias_divergence_fn)(widthnet)
                name = 'Extended ABCD Dense Flipout'
            elif mode==3:
                meannet = mergedinput
                #meannet = tfk.layers.Dense(32, activation=tf.nn.relu)(mergedinput)
                widthnet = tfk.layers.Dense(32, activation=tf.nn.relu)(mergedinput)
                param0 = tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1./self.batchsize)(meannet)
                param1 = tfp.layers.DenseVariational(1, posterior_mean_field, prior_trainable, kl_weight=1./self.batchsize)(widthnet)
                name = 'Extended ABCD Dense Variational'
            params = tf.concat([param0, param1], axis=1)
            netout = tfp.layers.DistributionLambda(normal_sp)(params)       
        return tfk.Model(netin, netout)

    def call(self, x):
        return self.model(x)