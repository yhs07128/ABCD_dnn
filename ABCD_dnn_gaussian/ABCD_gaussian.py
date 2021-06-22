import numpy as np
import matplotlib.pyplot as plt


def gen_sample3d(mean_bknd, cov_bknd, mean_sig, cov_sig, n_bknd, n_sig):
    
    bknd = np.random.multivariate_normal(mean_bknd, cov_bknd, n_bknd)
    sig = np.random.multivariate_normal(mean_sig, cov_sig, n_sig)
    
    return bknd, sig

def plot(bknd, sig, s=2, figsize=(15,4)):
    x_bknd, y_bknd, z_bknd = bknd.T
    x_sig, y_sig, z_sig = sig.T
    
    fig, ax = plt.subplots(1,3, figsize=figsize)

    ax[0].scatter(x_sig,y_sig, s=s, color='red')
    ax[1].scatter(x_sig,z_sig, s=s, color='red')
    ax[2].scatter(y_sig,z_sig, s=s, color='red')
    ax[0].scatter(x_bknd,y_bknd, s=s)
    ax[1].scatter(x_bknd,z_bknd, s=s)
    ax[2].scatter(y_bknd,z_bknd, s=s)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('z')
    ax[2].set_xlabel('y')
    ax[2].set_ylabel('z')


def plot3d(bknd, sig, s=2, figsize=(7,7)):
    x_bknd, y_bknd, z_bknd = bknd.T
    x_sig, y_sig, z_sig = sig.T
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_bknd,y_bknd,z_bknd, s=s)
    ax.scatter(x_sig,y_sig,z_sig, s=s, color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    
def abcd2d(x, y, bins):
    
    xy_a = (x>bins[0])&(x<bins[1])&(y>bins[0])&(y<bins[1])
    xy_b = (x>bins[1])&(y>bins[0])&(y<bins[1])
    xy_c = (y>bins[1])&(x>bins[0])&(x<bins[1])
    xy_d = (x>bins[1])&(y>bins[1])
    
    return xy_a.sum()+xy_b.sum()+xy_c.sum(), xy_d.sum(), xy_c.sum()*(xy_b.sum()/xy_a.sum())


def abcd3d(bknd, sig, bins):
    
    x, y, z = bknd.T
    xy_bknd, xy_true, xy_bknd_abcd = abcd2d(x, y, bins)
    xz_bknd, xz_true, xz_bknd_abcd = abcd2d(x, z, bins)
    yz_bknd, yz_true, yz_bknd_abcd = abcd2d(y, z, bins)
    
    x, y, z = np.concatenate((bknd, sig)).T
    xy_total, xy_total_sig, xy_total_abcd = abcd2d(x, y, bins)
    xz_total, xz_total_sig, xz_total_abcd = abcd2d(x, z, bins)
    yz_total, yz_total_sig, yz_total_abcd = abcd2d(y, z, bins)
    
    bknd_true = [xy_true, xz_true, yz_true]
    total_true = [xy_total_sig, xz_total_sig, yz_total_sig]
    sig_contam = [1-xy_bknd/xy_total, 1-xz_bknd/xz_total, 1-yz_bknd/yz_total]
    abcd_bknd = [xy_bknd_abcd, xz_bknd_abcd, yz_bknd_abcd]
    abcd_total = [xy_total_abcd, xz_total_abcd, yz_total_abcd]
    
    return bknd_true, total_true, sig_contam, abcd_bknd, abcd_total