import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from ABCD_dnn import ExtendedABCD, NLL

def prepinput(hist, row, column):
    xbin = np.linspace(0, 1, hist.shape[0])
    ybin = np.linspace(0, 1, hist.shape[1])
    
    crlist = []
    
    for i in range(hist.shape[0]):
        if i<row:
            for j in range(hist.shape[1]):
                crlist.append([xbin[i], ybin[j], hist[i,j]])
        else:
            for j in range(column):
                crlist.append([xbin[i], ybin[j], hist[i,j]])
    
    return crlist

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

def sample(model, cond, nrepeat=1000):
    cond_multiple = np.repeat(cond, nrepeat, axis=0)
    srout = np.power(10.0, model.predict(cond_multiple))*10000.0
    #srout = model.predict(cond_multiple)
    av = np.average(srout)
    std = np.std(srout)    
    return av, std

def samplegrid(model, hist):
    xbin = np.linspace(0, 1, hist.shape[0])
    ybin = np.linspace(0, 1, hist.shape[1])
    
    grid = np.zeros(hist.shape)
    grid_std = np.zeros(hist.shape)
    
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            grid[i,j], grid_std[i,j] = sample(model, [[xbin[i], ybin[j]]])
            
    return grid, grid_std

def testABCDrate(crlist):
  
    datalist = prepdata(crlist, nrepeat=100)

    datanp = np.array(datalist, dtype=np.float32)

    featin = datanp[:, 0:2]
    target = datanp[:, 2:]
    targetout = np.log10(target/10000.0)

    ncontrolregions = len(crlist)

    # build simple model
    netin = keras.layers.Input(shape=(2,))
    net = netin
    nrows = tf.shape(net)[0]

    mode = 2 # 1=Ext. 2=DenseFlipout 3=DenseVariational 
    model = ExtendedABCD(2, 128, mode=mode)
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