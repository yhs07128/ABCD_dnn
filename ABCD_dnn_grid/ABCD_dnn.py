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
from onehotencoder import OneHotEncoder_int
from NAF import NAF2, invNAF_bisect
tfk = tf.keras
tfd = tfp.distributions

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class SawtoothSchedule(LearningRateSchedule):
    def __init__(self, start_learning_rate=0.0001, end_learning_rate=0.000001, cycle_steps=100, random_fluctuation = 0.0, name=None):
        super(SawtoothSchedule, self).__init__()
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.cycle_steps = cycle_steps
        self.random_fluctuation = random_fluctuation
        self.name = name
    pass

    def __call__(self, step):
        phase = step % self.cycle_steps
        lr = self.start_learning_rate + (self.end_learning_rate-self.start_learning_rate)* (phase/self.cycle_steps)
        if (self.random_fluctuation>0):
            lr *= np.random.normal(1.0, self.random_fluctuation)
        return lr

    def get_config(self):
        return {
            "start_learning_rate": self.start_learning_rate,
            "end_learning_rate": self.end_learning_rate,
            "cycle_steps": self.cycle_steps,
            "random_fluctuation": self.random_fluctuation,
            "name": self.name
        }



class ABCDnn(object):
    def __init__(self, inputdim_categorical_list, inputdim, minibatch=128, netarch=[1024, 256, 64], nafdim=16, depth=2, LRrange=[0.0001, 0.0001, 1, 0.0], \
        conddim=0, beta1=0.5, beta2=0.9, lamb=1.0, retrain=False, seed=100, permute=False, savedir='./abcdnn/', savefile='abcdnn.pkl'):
        self.inputdim_categorical_list = inputdim_categorical_list
        self.inputdim = inputdim
        self.inputdimcat = int(np.sum(inputdim_categorical_list))
        self.inputdimreal = inputdim - self.inputdimcat
        self.minibatch = minibatch
        self.netarch = netarch
        self.nafdim = nafdim
        self.depth = depth
        self.LRrange = LRrange
        self.conddim = conddim
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb = lamb
        self.retrain = retrain
        self.savedir = savedir
        self.savefile = savefile
        self.global_step = tf.Variable(0, name='global_step')
        self.monitor_record = []
        self.monitorevery = 50
        self.seed = seed
        self.permute = permute
        self.setup()

    def setup(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.createmodel()
        self.checkpoint = tf.train.Checkpoint(global_step=self.global_step, model = self.model, optimizer=self.optimizer,)
        self.checkpointmgr = tf.train.CheckpointManager(self.checkpoint, directory=self.savedir+'/model1', checkpoint_name='ckpt1', max_to_keep=5)

        self.checkpoint_out = tf.train.Checkpoint(global_step=self.global_step, model = self.model_out)
        self.checkpointmgr_out = tf.train.CheckpointManager(self.checkpoint_out, directory=self.savedir+'/model2', checkpoint_name='ckpt2', max_to_keep=5)

        if (not self.retrain) and os.path.exists(self.savedir):
            status = self.checkpoint.restore(self.checkpointmgr.latest_checkpoint)
            status.assert_existing_objects_matched()
            status = self.checkpoint_out.restore(self.checkpointmgr_out.latest_checkpoint)
            status.assert_existing_objects_matched()
            print('loaded model from checkpoint')
            if os.path.exists(os.path.join(self.savedir, self.savefile)):
                print("Reading monitor file")
                self.load_training_monitor()
            print("Resuming from step", self.global_step)
        elif not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        pass

    def createmodel(self):
        
        # input distribution transformed to Gaussian
        self.model = NAF2(self.inputdimreal, self.conddim, nafdim=self.nafdim, depth=self.depth, permute=self.permute)

        # target  distribution transformed to Gaussian
        self.model_out = NAF2(self.inputdimreal, self.conddim, nafdim=self.nafdim, depth=self.depth, permute=self.permute)
        
        self.trainvars = self.model.trainable_variables + self.model_out.trainable_variables

        #tf.keras.utils.plot_model(self.model, to_file=self.savedir+'ABCD_dnn.png')
        lr_fn = SawtoothSchedule(self.LRrange[0], self.LRrange[1], self.LRrange[2], self.LRrange[3])
        self.optimizer = tfk.optimizers.Adam(learning_rate = lr_fn,  beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-5, name='nafopt')
        pass

    def setrealdata(self, numpydata, eventweight=None):
        self.numpydata = numpydata
        self.ntotalevents = numpydata.shape[0]
        self.datacounter = 0
        self.randorder = np.random.permutation(self.numpydata.shape[0])
        if eventweight is not None:
            self.eventweight = eventweight
        else:
            self.eventweight = np.ones((self.ntotalevents, 1), np.float32)
        pass


    def savehyperparameters(self):
        """Write hyper parameters into file
        """
        params = [self.inputdim, self.conddim, self.LRrange, self.beta1, self.beta2, self.minibatch, self.nafdim, self.depth]
        pickle.dump(params, open(os.path.join(self.savedir, 'hyperparams.pkl'), 'wb'))
        pass

    def monitor(self):
        self.monitor_record.append([self.checkpoint.global_step.numpy(), self.glossv.numpy()])

    def save_training_monitor(self):
        pickle.dump(self.monitor_record, open(os.path.join(self.savedir, self.savefile), 'wb'))
        pass

    def load_training_monitor(self):
        fullfile = os.path.join(self.savedir, self.savefile)
        if os.path.exists(fullfile):
            self.monitor_record = pickle.load(open(fullfile, 'rb'))
            self.epoch = self.monitor_record[-1][0] + 1
        pass

    def get_next_batch(self, size=None, cond=False):
        """Return minibatch from random ordered numpy data
        """
        if size is None:
            size = self.minibatch
        if self.datacounter + size >= self.ntotalevents:
            self.datacounter = 0
            self.randorder = np.random.permutation(self.numpydata.shape[0])

        batchbegin = self.datacounter
        if not cond:
            batchend = batchbegin + size
            self.datacounter += size
            nextbatch = self.numpydata[self.randorder[batchbegin:batchend], 0::]
        else:
            nextconditional = self.numpydata[self.randorder[batchbegin], self.inputdimreal:]
            # find all data 
            bigenough = False
            while(not bigenough):
                match = (self.numpydata[:, self.inputdimreal:] == nextconditional).all(axis=1)
                nsamples_incategory = np.count_nonzero(match)
                if nsamples_incategory>=self.minibatch:
                    bigenough = True
                batchbegin += 1
                if batchbegin >= self.ntotalevents:
                    batchbegin = 0
                
            matchingarr = self.numpydata[match, :]
            matchingwgt = self.eventweight[match, :]
            randorder = np.random.permutation(matchingarr.shape[0])
            nextbatch = matchingarr[randorder[0:self.minibatch], :]
            nextbatchwgt = matchingwgt[randorder[0:self.minibatch], :]

        return nextbatch

    @tf.function
    def train_step(self, sourcebatch, targetbatch):
        # update discriminator ncritics times
        if self.conddim>0:
            #conditionals = tf.constant(targetbatch[:, -self.conddim:]) # use this when tf.function decorator is not used
            conditionals = targetbatch[:, -self.conddim:]
        jacfx = 0
        jacfx_t = 0
        fx_i = []
        fx_t_i = []

        with tf.GradientTape() as losstape:
            #featurex=tf.constant(sourcebatch[:, :self.inputdim]) # use this when tf.function decorator is not used
            featurex=sourcebatch[:, :self.inputdimreal]
            # source distribution to be transformed to a Gaussian
            with tf.GradientTape(persistent=True) as jactape:
                jactape.watch(featurex)
                fx = self.model(tf.concat([featurex, conditionals], axis=-1), training=True)
                for i in range(self.inputdimreal):
                    fx_i.append(fx[:, i])
            for i in range(self.inputdimreal):
                jacfx_i = jactape.gradient(fx_i[i], featurex) # gradient per column, equal to a column of jacobian
                tmp = tf.math.log(tf.math.abs(jacfx_i[:,i])+1.0e-7)
                jacfx += tmp
                #tf.debugging.check_numerics(jacfx, 'bad jacfx!!')
            lnPz_source = -0.5*tf.reduce_mean(tf.reduce_sum(tf.math.square(fx), axis=1))
            d_kl_source = -tf.reduce_mean(jacfx) - lnPz_source

            # target distribution also transformed to a Gaussian
            #featurex_t=tf.constant(targetbatch[:, :self.inputdim])
            featurex_t=targetbatch[:, :self.inputdimreal]
            with tf.GradientTape(persistent=True) as jactape_t:
                jactape_t.watch(featurex_t)
                fx_t = self.model_out(tf.concat([featurex_t, conditionals], axis=-1), training=True)
                for i in range(self.inputdimreal):
                    fx_t_i.append(fx_t[:, i])
            for i in range(self.inputdimreal):
                jacfx_t_i = jactape_t.gradient(fx_t_i[i], featurex_t) # gradient per column, equal to a column of jacobian
                tmp = tf.math.log(tf.math.abs(jacfx_t_i[:,i])+1.0e-7)
                jacfx_t += tmp
                #tf.debugging.check_numerics(jacfx, 'bad jacfx_t!!')
            lnPz_target = -0.5*tf.reduce_mean(tf.reduce_sum(tf.math.square(fx_t), axis=1))
            d_kl_target = -tf.reduce_mean(jacfx_t) - lnPz_target

            d_kl = d_kl_source + d_kl_target 

        #tf.debugging.check_numerics(d_kl, 'bad!!')
        if tf.math.is_finite(d_kl):
            grad = losstape.gradient(d_kl, self.trainvars)
            self.optimizer.apply_gradients(zip(grad, self.trainvars))

        meangloss = d_kl

        return meangloss

    def train(self, steps=1000):
        for istep in range(steps):
            source = self.get_next_batch()
            target = self.get_next_batch(cond=True)
            self.glossv = self.train_step(source, target)
            # generator update
            if istep % self.monitorevery == 0:
                print(f'{self.checkpoint.global_step.numpy()} {self.glossv.numpy():.3e} ')
                self.monitor()
                self.checkpointmgr.save()
                self.checkpointmgr_out.save()
            self.checkpoint.global_step.assign_add(1) # increment counter
        self.checkpointmgr.save()
        self.checkpointmgr_out.save()
        self.save_training_monitor()

    def display_training(self):
        # Following section is for creating movie files from trainings

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        monarray = np.array(self.monitor_record)
        x = monarray[0::, 0]
        ax.plot(x, monarray[0::, 1], color='r', label='gloss')
        ax.set_yscale('linear')
        ax.legend()

        plt.draw()

        fig.savefig(os.path.join(self.savedir, 'trainingperf.pdf'))
        pass

    def generate_sample(self, condition, repeatfirst=False):
        xin = self.get_next_batch()

        if not repeatfirst:
            xin_nocond = xin[:, :self.inputdimreal]
        else:
            xin_nocond = np.repeat(xin[0, :self.inputdimreal], self.minibatch, axis=0).reshape((self.minibatch, self.inputdimreal))

        yin = np.repeat(np.array(condition, dtype=np.float32), self.minibatch, axis=0) # copy the same
        netin = np.hstack((xin_nocond, yin))
        youthat = self.predict(netin)
        return youthat

    def predict(self, datain):
        z = self.model.predict(datain)
        y = invNAF_bisect(self.model_out, z, self.inputdimreal, datain[:, self.inputdimreal:])
        return y

    def predict_getlatent(self, datain):
        z = self.model.predict(datain)
        y = invNAF_bisect(self.model_out, z, self.inputdimreal, datain[:, self.inputdimreal:])
        return y, z

    def getlatent(self, datain):
        z = self.model.predict(datain)
        return z



class Morph_to_Gauss(object):
    def __init__(self, inputdim, minibatch=128, netarch=[1024, 256, 64], nafdim=16, depth=2, LRrange=[0.0001, 0.0001, 1, 0.0], \
        beta1=0.5, beta2=0.9, lamb=1.0, retrain=False, seed=100, permute=False, savedir='./morphgauss/', savefile='morphgaus.pkl'):
        self.inputdim = inputdim
        self.minibatch = minibatch
        self.netarch = netarch
        self.nafdim = nafdim
        self.depth = depth
        self.LRrange = LRrange
        self.beta1 = beta1
        self.beta2 = beta2
        self.lamb = lamb
        self.retrain = retrain
        self.savedir = savedir
        self.savefile = savefile
        self.global_step = tf.Variable(0, name='global_step')
        self.monitor_record = []
        self.monitorevery = 50
        self.seed = seed
        self.permute = permute
        self.setup()

    def setup(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.createmodel()
        self.checkpoint = tf.train.Checkpoint(global_step=self.global_step, model = self.model, optimizer=self.optimizer)
        self.checkpointmgr = tf.train.CheckpointManager(self.checkpoint, directory=self.savedir, max_to_keep=5)
        if (not self.retrain) and os.path.exists(self.savedir):
            status = self.checkpoint.restore(self.checkpointmgr.latest_checkpoint)
            status.assert_existing_objects_matched()
            print('loaded model from checkpoint')
            if os.path.exists(os.path.join(self.savedir, self.savefile)):
                print("Reading monitor file")
                self.load_training_monitor()
            print("Resuming from step", self.global_step)
        elif not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        pass

    def createmodel(self):
        
        # input distribution transformed to Gaussian
        self.model = NAF2(self.inputdim, 0, nafdim=self.nafdim, depth=self.depth, permute=self.permute)

        self.trainvars = self.model.trainable_variables 

        #tf.keras.utils.plot_model(self.model, to_file=self.savedir+'ABCD_dnn.png')
        lr_fn = SawtoothSchedule(self.LRrange[0], self.LRrange[1], self.LRrange[2], self.LRrange[3])
        self.optimizer = tfk.optimizers.Adam(learning_rate = lr_fn,  beta_1=self.beta1, beta_2=self.beta2, epsilon=1e-5, name='nafopt')
        pass

    def setrealdata(self, numpydata, eventweight=None):
        self.numpydata = numpydata
        self.ntotalevents = numpydata.shape[0]
        self.datacounter = 0
        self.randorder = np.random.permutation(self.numpydata.shape[0])
        if eventweight is not None:
            self.eventweight = eventweight
        else:
            self.eventweight = np.ones((self.ntotalevents, 1), np.float32)
        pass


    def savehyperparameters(self):
        """Write hyper parameters into file
        """
        params = [self.inputdim, self.conddim, self.LRrange, self.beta1, self.beta2, self.minibatch, self.nafdim, self.depth]
        pickle.dump(params, open(os.path.join(self.savedir, 'hyperparams.pkl'), 'wb'))
        pass

    def monitor(self):
        self.monitor_record.append([self.checkpoint.global_step.numpy(), self.glossv.numpy()])

    def save_training_monitor(self):
        pickle.dump(self.monitor_record, open(os.path.join(self.savedir, self.savefile), 'wb'))
        pass

    def load_training_monitor(self):
        fullfile = os.path.join(self.savedir, self.savefile)
        if os.path.exists(fullfile):
            self.monitor_record = pickle.load(open(fullfile, 'rb'))
            self.epoch = self.monitor_record[-1][0] + 1
        pass

    def get_next_batch(self, size=None, cond=False):
        """Return minibatch from random ordered numpy data
        """
        if size is None:
            size = self.minibatch
        if self.datacounter + size >= self.ntotalevents:
            self.datacounter = 0
            self.randorder = np.random.permutation(self.numpydata.shape[0])

        batchbegin = self.datacounter
        batchend = batchbegin + size
        self.datacounter += size
        nextbatch = self.numpydata[self.randorder[batchbegin:batchend], 0::]

        return nextbatch

    @tf.function
    def train_step(self, sourcebatch):
        jacfx = 0
        fx_i = []
        with tf.GradientTape() as losstape:
            #featurex=tf.constant(sourcebatch[:, :self.inputdim]) # use this when tf.function decorator is not used
            featurex=sourcebatch[:, :self.inputdim]
            # source distribution to be transformed to a Gaussian
            with tf.GradientTape(persistent=True) as jactape:
                jactape.watch(featurex)
                fx = self.model(featurex, training=True)
                for i in range(self.inputdim):
                    fx_i.append(fx[:, i])
            for i in range(self.inputdim):
                jacfx_i = jactape.gradient(fx_i[i], featurex) # gradient per column, equal to a column of jacobian
                tmp = tf.math.log(tf.math.abs(jacfx_i[:,i]) + 1.0e-7)
                tmp = tf.where(tf.math.is_finite(tmp), tmp, -10.0)
                tmp = tf.clip_by_value(tmp, -10.0, 5.0)
                jacfx += tmp

            lnPz = -0.5*tf.reduce_mean(tf.reduce_sum(tf.math.square(fx), axis=1))
            d_kl_source = -tf.reduce_mean(jacfx) - lnPz

            d_kl = d_kl_source 

        #tf.debugging.check_numerics(d_kl, 'bad!!')
        if tf.math.is_finite(d_kl):
            grad = losstape.gradient(d_kl, self.trainvars)
            self.optimizer.apply_gradients(zip(grad, self.trainvars))

        meangloss = d_kl

        return meangloss

    def train(self, steps=1000):
        for istep in range(steps):
            source = self.get_next_batch()
            self.glossv = self.train_step(source)
            # generator update
            if istep % self.monitorevery == 0:
                print(f'{self.checkpoint.global_step.numpy()} {self.glossv.numpy():.3e} ')
                self.monitor()
                self.checkpointmgr.save()
            self.checkpoint.global_step.assign_add(1) # increment counter
        self.checkpointmgr.save()
        self.save_training_monitor()

    def display_training(self):
        # Following section is for creating movie files from trainings

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        monarray = np.array(self.monitor_record)
        x = monarray[0::, 0]
        ax.plot(x, monarray[0::, 1], color='r', label='loss')
        ax.set_yscale('linear')
        ax.legend()

        plt.draw()

        fig.savefig(os.path.join(self.savedir, 'trainingperf.pdf'))
        pass

    def generate_sample(self):
        zin = np.random.normal(size=(self.minibatch, self.inputdim)).astype(dtype=np.float32)

        xout = self.predict(zin)
        return xout

    def predict(self, zin):
        y = invNAF_bisect(self.model, zin, self.inputdim, None)
        return y

    def getlatent(self, datain):
        z = self.model.predict(datain)
        return z

def secderiv(h, prediction, prediction_minush, prediction_plush):
    tmp = tf.math.square((prediction_plush + prediction_minush - 2*prediction)/h**2)
    return tmp

# try extended ABCD with NN.. possible ? Not possible
# try something simpler
def extendedABCD(ncrvars):
    inputshape = (ncrvars,)
    netin = tfk.layers.Input(shape=inputshape)
    nrows = tf.shape(netin)[0]
    mode = 1
    if mode==1:
        a=tf.expand_dims(netin, axis=-1)
        b=tf.expand_dims(netin, axis=1)
        outerproductmat = tf.linalg.matmul(a, b)
        oprod_lower=tf.linalg.LinearOperatorLowerTriangular(outerproductmat)
        elements = int(ncrvars * (ncrvars+1) /2)
        outerproduct = tf.reshape(oprod_lower.to_dense(), shape=(nrows, ncrvars**2))
    elif mode==2:
        outerproduct = tf.einsum('bi,bj->bij', netin, netin)
        outerproduct = tf.reshape(outerproduct, (nrows,ncrvars**2))
    collect = tf.concat([netin, outerproduct], axis=1)
    netout = tfk.layers.Dense(1)(collect)
    return tfk.Model(netin, netout)

# Specify the surrogate posterior over `keras.layers.Dense` `kernel` and `bias`.
def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                        scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])

# Specify the prior over `keras.layers.Dense` `kernel` and `bias`.
def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t, scale=1),
            reinterpreted_batch_ndims=1)),
    ])

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