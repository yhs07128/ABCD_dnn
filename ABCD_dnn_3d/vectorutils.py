import numpy as np

# invariant mass
# a,b are massless vectors (px, py, pz)
def invmass_4v(a):
    """Return invariant mass of a four vector
    
    Arguments:
        a {numpy array} -- [description]
    
    Returns:
        [type] -- [description]
    """

    return np.sqrt(a[3]**2 - a[0]**2 - a[1]**2 - a[2]**2)

def invmass_3varr(a,b):
    """Calculate invariant mass of two three-vectors assuming they're massless
    
    Arguments:
        a {numpy array} -- three-vector px, py, pz
        b {numpy array} -- three-vector px, py, pz
    
    Returns:
        numpy array -- array of invariant mass
    """

    # energy
    Ea = np.linalg.norm(a, axis=1)
    Eb = np.linalg.norm(b, axis=1)

    Esum = np.reshape(Ea+Eb, newshape=(a.shape[0], 1))
    abvecsum = a + b

    hsarray = np.hstack((abvecsum, Esum))    

    return invmass_4varr(hsarray)

def invmass_3vlist(a):
    """Calculate invariant mass of a three-vector list, assuming they're massless
    
    Arguments:
        a {list of numpy matrix} -- each matrix contains rows of px, py, pz
    
    Returns:
        numpy array -- containing invariant masses
    """

    nrows = a[0].shape[0]
    # energy sum
    Esum = np.zeros(nrows)
    # 3 vector sum
    vecsum = np.zeros((nrows, 3))
    for acolumn in a:
        Esum = Esum + np.linalg.norm(acolumn, axis=1)
        vecsum = vecsum + acolumn
    Esum = np.reshape(Esum, newshape=(nrows, 1))

    hsarray = np.hstack((vecsum, Esum))

    return invmass_4varr(hsarray)


def invmass_4varr(a):
    """Calculate invariant mass
    
    Arguments:
        a {numpy matrix} -- each row containts a four-vector px, py, pz, E
    
    Returns:
        numpy array -- invariant mass
    """

    return np.apply_along_axis(invmass_4v, axis=1, arr=a)

def pt(a):
    """Transverse momenta
    
    Arguments:
        a {numpy matrix} -- Each row contains either a three-vector or a four-vector
    
    Returns:
        [type] -- [description]
    """

    return np.sqrt(a[0]**2 + a[1]**2)

def pt_varr(a):
    return np.apply_along_axis(pt, axis=1, arr=a)

def eta(a):
    """Pseudorapidty
    
    Arguments:
        a {numpy array} -- It should be a three-vector
    
    Returns:
        [type] -- [description]
    """
    a3 = np.sqrt(a[0]**2+ a[1]**2 + a[2]**2)
    return -0.5*np.log( (a3+a[2]) / (a3-a[2]) )

def eta_varr(a):
    return np.apply_along_axis(eta, axis=1, arr=a)

def twodot(a,b):
    return a[0::, 0]*b[0::, 0] + a[0::, 1]*b[0::, 1]


def deta(a,b):
    return np.apply_along_axis(eta, axis=1, arr=a) - np.apply_along_axis(eta, axis=1, arr=b)

def dphi(a,b):
    return np.arccos ( twodot(a,b) / np.sqrt(twodot(a,a)) / np.sqrt(twodot(b,b)) )

def dR(a,b):
    return np.hypot(deta(a,b), dphi(a,b))