import kernels
import numpy as np

def get_k(pnt, lscale, size):
    Kov = np.zeros((size, size))
    for idx in range(size):
        for idy in range(size):
            Kov[idx, idy] = kernels.matern3(pnt[idx], pnt[idy], lscale)
    return Kov

def get_ks(pnt1, pnt2, lscale, sz1, refratio):
    kovs = np.zeros((sz1, refratio**2))
    for idy in range(refratio**2):
        for idx in range(sz1):
            kovs[idx, idy] = kernels.matern3(pnt1[idy], pnt2[idx], lscale)
    return kovs

def get_pnt9(dx, dy):
    return np.array([[-dx, -dy], [0, -dy], [dx, -dy], 
                   [-dx,   0], [0,   0], [dx,   0], 
                   [-dx,  dy], [0,  dy], [dx,  dy]])

def get_MSpnts(dx, dy):
    pnts = np.zeros((9, 9, 2))
    for idx in range(9):
        pnts[idx] = get_pnt9(dx, dy)
    pnts[0:3,:,1] -= dy
    pnts[0,:,0] -= dx
    pnts[2,:,0] += dx 

    pnts[3,:,0] -= dx
    pnts[5,:,0] += dx
    pnts[6:9,:,1] += dy 
    pnts[6,:,0] -= dx
    pnts[8,:,0] += dx
    return pnts 

def get_MSweights(dx, dy, lscale, refratio):
    kovs = np.zeros((9, 9, refratio**2))
    pnts = get_MSpnts(dx,dy)
    Kov = get_k(pnts[4], lscale, 9)
    Kinv = np.linalg.inv(Kov)
    kpnt = get_kpnt(dx, dy, refratio)
    for idx in range(9):
        kovs[idx] = get_ks(kpnt, pnts[idx], lscale, 9, refratio)
        kovs[idx] = np.dot(Kinv, kovs[idx])
    return kovs


def get_pntS(dx, dy): 
    grdy, grdx = np.mgrid[-2:3,-2:3]
    grdx = grdx*dx
    grdy = grdy*dy
    pnts = np.array([grdx.flatten(), grdy.flatten()]).transpose()
    return pnts

def get_kpnt(delx, dely, refratio):
    kpnt = np.zeros((refratio**2, 2))
    if refratio == 4:
        start = -0.375
    else:
        start = -0.25
    for idy in range(refratio):
        for idx in range(refratio):
            idk = idx + idy*refratio
            kpnt[idk, 0] = (start + 1.0/refratio*idx)*delx
            kpnt[idk, 1] = (start + 1.0/refratio*idy)*dely
    return kpnt

def get_SuperK(pnts, dx, dy, lscale):
    SuperK = np.zeros((25,25))
    for idx in range(25):
        for idy in range(25):
            SuperK[idx, idy] = kernels.matern3(pnts[idx], pnts[idy], lscale)
    return SuperK

def get_Superkstar(pnts, dx, dy, lscale, refratio):
    kovs = np.zeros((25, refratio**2))
    kpnts = get_kpnt(dx, dy, refratio)
    kovs = get_ks(kpnts, pnts, lscale, 25, refratio)
    return kovs

def get_SuperWeights(dx, dy, lscale, refratio):
    pnts = get_pntS(dx,dy)
    Kinv = np.linalg.inv(get_SuperK(pnts, dx, dy, lscale))
    kovs = get_Superkstar(pnts, dx, dy, lscale, refratio)
    for idx in range(refratio**2):
        kovs[:,idx] = np.dot(kovs[:,idx], Kinv)
    return kovs

def const_A(kovs):
    A = np.array([[kovs[0, 0], 0, 0, 0, 0, 0, 0, 0, 0],
                  [kovs[0, 1], kovs[1, 0], 0, 0, 0, 0, 0, 0, 0], 
                  [kovs[0, 2], kovs[1, 1], kovs[2, 0], 0, 0, 0, 0, 0, 0], 
                  [0, kovs[1, 2], kovs[2, 1], 0, 0, 0, 0, 0, 0], 
                  [0, 0, kovs[2, 2], 0, 0, 0, 0, 0, 0], 
                  [kovs[0, 3], 0, 0, kovs[3, 0], 0, 0, 0, 0, 0], 
                  [kovs[0, 4], kovs[1, 3], 0, kovs[3, 1], kovs[4, 0], 0, 0, 0, 0], 
                  [kovs[0, 5], kovs[1, 4], kovs[2, 3], kovs[3, 2], kovs[4, 1], kovs[5, 0], 0, 0, 0], 
                  [0, kovs[1, 5], kovs[2, 4], 0, kovs[4, 2], kovs[5, 1], 0, 0, 0],
                  [0, 0, kovs[2, 5], 0, 0, kovs[5, 2], 0, 0, 0], 
                  [kovs[0, 6], 0, 0, kovs[3, 3], 0, 0, kovs[6, 0], 0, 0], 
                  [kovs[0, 7], kovs[1, 6], 0, kovs[3, 4], kovs[4, 3], 0, kovs[6, 1], kovs[7, 0], 0], 
                  [kovs[0, 8], kovs[1, 7], kovs[2, 6], kovs[3, 5], kovs[4, 4], kovs[5, 3], kovs[6, 2], kovs[7, 1], kovs[8, 0]],
                  [0, kovs[1, 8], kovs[2, 7], 0, kovs[4, 5], kovs[5, 4], 0, kovs[7,2], kovs[8,1]], 
                  [0, 0, kovs[2, 8], 0, 0, kovs[5,5], 0, 0, kovs[8, 2]], 
                  [0, 0, 0, kovs[3, 6], 0, 0, kovs[6, 3], 0, 0], 
                  [0, 0, 0, kovs[3, 7], kovs[4, 6], 0, kovs[6, 4], kovs[7, 3], 0], 
                  [0, 0, 0, kovs[3, 8], kovs[4, 7], kovs[5, 6], kovs[6, 5], kovs[7, 4], kovs[8, 3]],
                  [0, 0, 0, 0, kovs[4, 8], kovs[5, 7], 0, kovs[7, 5], kovs[8, 4]], 
                  [0, 0, 0, 0, 0, kovs[5, 8], 0, 0, kovs[8, 5]], 
                  [0, 0, 0, 0, 0, 0, kovs[6, 6], 0, 0], 
                  [0, 0, 0, 0, 0, 0, kovs[6, 7], kovs[7, 6], 0], 
                  [0, 0, 0, 0, 0, 0, kovs[6, 8], kovs[7, 7], kovs[8, 6]], 
                  [0, 0, 0, 0, 0, 0, 0, kovs[7, 8], kovs[8, 7]], 
                  [0, 0, 0, 0, 0, 0, 0, 0, kovs[8, 8]]])
    return A

def get_gammas(kovs, total, refr): 
    gammas = np.zeros((refr**2, 9))
    for idx in range(refr**2):
        A = const_A(kovs[:,:,idx])
        gammas[idx] = np.linalg.lstsq(A, total[:,idx], rcond=None)[0]
    return gammas

