import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from ABCD_dnn import ExtendedABCD, NLL

def prepdata(crlist, nrepeat=100):
    sampleddata = []
    for acr in crlist:
        nmean = acr[-1]*1.0
        for itoy in range(nrepeat):
            sampledentry = np.random.normal(nmean, np.sqrt(nmean))
            if sampledentry>0:
                asample = acr.copy()
                asample[-1] = sampledentry
                sampleddata.append(asample)
            else: continue
                
    return sampleddata

def sample3d(model, shape, nrepeat=1000):
    xbin = np.linspace(0, 1, shape[0])
    ybin = np.linspace(0, 1, shape[1])
    zbin = np.linspace(0, 1, shape[2])
    
    xx, yy, zz = np.meshgrid(xbin, ybin, zbin, indexing='ij')
    
    cond = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
            
    cond_multiple = np.repeat(cond, nrepeat, axis=0)
    srout = (np.power(10.0, model.predict(cond_multiple))*10000.0).reshape((-1, nrepeat))
    
    srout_mean = np.average(srout, axis=1).reshape(shape)
    srout_std = np.std(srout, axis=1).reshape(shape)
    
    return srout_mean, srout_std

def testABCDrate(crlist):
  
    datalist = prepdata(crlist, nrepeat=100)

    datanp = np.array(datalist, dtype=np.float32)

    featin = datanp[:, 0:3]
    target = datanp[:, 3]
    targetout = np.log10(target/10000.0)

    ncontrolregions = len(crlist)

    # build simple model
    netin = keras.layers.Input(shape=(3,))
    net = netin
    nrows = tf.shape(net)[0]

    mode = 2 # 1=Ext. 2=DenseFlipout 3=DenseVariational 
    model = ExtendedABCD(3, 128, mode=mode)
    if mode==1:
        lossfn = keras.losses.MSE
    else:
        lossfn = NLL
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=lossfn)

    # early stopping
    callback = keras.callbacks.EarlyStopping(min_delta=1e-2, patience=5, monitor='loss')

    model.fit(featin, targetout, batch_size=128, epochs=200, verbose=0, callbacks=[callback])
    #model.summary()
    #print(model.trainable_variables)
    
    return model