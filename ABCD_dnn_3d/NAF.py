import tensorflow as tf
import numpy as np

from vectorutils import *

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfk = tf.keras


def invsigmoid(x):
    #return tf.math.log((xclip+1e-10)/(1.0-xclip+1e-10))
    return tf.math.log(x/(1.0-x))


# construct NAF model
def NAF2(inputdim, conddim, nafdim, depth=1, permute=True):

    xin = tfk.layers.Input(shape=(inputdim+conddim, ))

    if conddim>0:
        xcondin = xin[:, inputdim:]
    else:
        xcondin = tf.zeros(shape=(tf.shape(xin)[0], 1), dtype=tf.float32)

    xfeatures = xin[:, :inputdim]
    netout = None
    nextfeature = xfeatures
    for idepth in range(depth):
        if permute:
            randperm = np.random.permutation(inputdim).astype('int32')
            permutation = tf.constant(randperm, name=f'permutation{idepth}')
            #permutation = tf.Variable(randperm, name=f'permutation{idepth}', trainable=False)
        else:
            permutation = tf.range(inputdim, dtype='int32',  name=f'permutation{idepth}')
        permuter = tfb.Permute(permutation=permutation, name=f'permute{idepth}')
        xfeatures_permuted = permuter.forward(nextfeature)
        outlist = []
        for iv in range(inputdim):
            xiv = tf.reshape(xfeatures_permuted[:, iv], [-1, 1])
            net = xiv
            condnet = xcondin
            condnet = tfk.layers.Dense(256, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dense(256, activation=tf.nn.leaky_relu)(condnet)
            w1 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            b1 = tfk.layers.Dense(nafdim, activation=None)(condnet)

            net1 = tf.nn.sigmoid(w1 * net + b1)
            condnet = xcondin
            condnet = tfk.layers.Dense(256, activation=tf.nn.leaky_relu)(condnet)
            condnet = tfk.layers.Dense(256, activation=tf.nn.leaky_relu)(condnet)
            w2 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            w2 = w2/ (tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize

            net = invsigmoid(tf.reduce_sum(net1 * w2, axis=1, keepdims=True))
            outlist.append(net)
            xcondin = tf.concat([xcondin, xiv], axis=1)
        outputlayer_permuted = tf.concat(outlist, axis=1)
        outputlayer = permuter.inverse(outputlayer_permuted)
        nextfeature = outputlayer

    return tfk.Model(xin, outputlayer)

# bisection method
# this is the sure way to find the inverse
def invNAF_bisect(nafmodel, output, inputdim, cond):
    ndim = inputdim
    npts = output.shape[0]
    epssq = 1.0e-5 
    left = -1.0
    right = 1.0

    maxtrial1 = 7
    maxtrial2 = 20

    if cond is None:
        hascondition = False
    else:
        hascondition = True

    trialfeatinput_left = np.zeros(shape=(npts, ndim), dtype=np.float32)
    trialfeatinput_right = np.zeros(shape=(npts, ndim), dtype=np.float32)

    # append conditions
    if cond is not None:
        trialfeatinput_left = np.concatenate([trialfeatinput_left, cond], axis=-1)
        trialfeatinput_right = np.concatenate([trialfeatinput_right, cond], axis=-1)

    for idim in range(ndim):
        trialfeatinput_left[:, idim] = left
        trialfeatinput_right[:, idim] = right

        trialoutput_left = nafmodel(trialfeatinput_left)[:,idim].numpy()
        trialoutput_right = nafmodel(trialfeatinput_right)[:,idim].numpy()

        outputi = output[:, idim]
        residual_left = trialoutput_left - outputi
        residual_right = trialoutput_right - outputi

        # first check whether the signs are opposite
        # increase boundary by factor 2 if not OK
        boundarycheckok = False
        itrial = 0
        while not boundarycheckok and itrial<maxtrial1:
            itrial += 1
            signs = residual_left * residual_right
            resizeboundary = (signs > 0.0)
            counttrue = np.count_nonzero(resizeboundary)
            if counttrue == 0:
                boundarycheckok = True
            else:
                trialfeatinput_left[resizeboundary,idim] = 1.5 * trialfeatinput_left[resizeboundary,idim]
                trialfeatinput_right[resizeboundary,idim] = 1.5 * trialfeatinput_right[resizeboundary,idim]
                trialoutput_left = nafmodel(trialfeatinput_left)[:,idim].numpy()
                trialoutput_right = nafmodel(trialfeatinput_right)[:,idim].numpy()
                residual_left = trialoutput_left - outputi
                residual_right = trialoutput_right - outputi
        
        if counttrue>0: # could not resolve then eliminate the data point
            selectrows = np.logical_not(resizeboundary)
            output = output[selectrows]
            trialfeatinput_left = trialfeatinput_left[selectrows]
            trialfeatinput_right = trialfeatinput_right[selectrows]
            outputi = outputi[selectrows]
            residual_left = residual_left[selectrows]
            residual_right = residual_right[selectrows]
        # apply bisection method now
        converged = False
        itrial = 0
        while not converged and itrial<maxtrial2:
            itrial += 1
            midpoint = (trialfeatinput_left + trialfeatinput_right)/2.0
            trialoutput_midpoint = nafmodel(midpoint)[:,idim].numpy()
            residual_midpoint = trialoutput_midpoint - outputi
            
            if np.count_nonzero(np.square(residual_midpoint) < epssq) == npts:
                converged = True

            leftboundaryOK = (residual_left * residual_midpoint <= 0.0)
            rightboundaryOK = (residual_right * residual_midpoint <= 0.0)
            
            trialfeatinput_right[leftboundaryOK, :] = midpoint[leftboundaryOK,:]
            residual_right[leftboundaryOK] = residual_midpoint[leftboundaryOK]
            trialfeatinput_left[rightboundaryOK, :] = midpoint[rightboundaryOK, :]
            residual_left[rightboundaryOK] = residual_midpoint[rightboundaryOK]
        trialfeatinput_left = midpoint
        trialfeatinput_right = midpoint.copy()
    if midpoint.shape[0] < npts:
        print(f'{npts - midpoint.shape[0]} out of {npts} failed')
    return midpoint[:, :inputdim]
   


def test_NAF():
    nafmodel = NAF(3, 3, 10)
    nafmodel.summary()
    print(nafmodel(np.array([[0.5, -0.5, 0.0,  1, 1, 0], [0.5, -0.5, 0.0,  1, 1, 0]], dtype=np.float32)))
    pass

def test_invNAF():
    nafmodel = NAF2(2,0,14)
    input = np.array([[0.1, 0.5]], np.float32)
    output = nafmodel(input)
    print(output)

    inputestimate = invNAF3(nafmodel, output, 2, None)
    print(inputestimate)
    pass


if __name__ == "__main__":
    test_invNAF()
    #tfk.utils.plot_model(nafmodel, to_file='NAF.png')
    pass
