import numpy as np

def loadsten(dat):
    return np.array([dat[0, 0], dat[1, 0], dat[2, 0],
                     dat[0, 1], dat[1, 1], dat[2, 1],
                     dat[0, 2], dat[1, 2], dat[2, 2]])

def gp9(width, height, lx, ly, ks, dat_in, dat_out):
    #3 Channels
    for k in range(3):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                sten = dat_in[i-1:i+2, j-1:j+2,k].flatten('F') # loadsten(dat_in[i - 1:i + 2, j - 1:j + 2, k])
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        ksid = iref + jref * lx
                        dat_out[ii, jj, k] = np.dot(ks[:,ksid], sten)
#Just copy border for now
        for i in range(0, height, height - 1):
            for j in range(width):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]
        for i in range(height):
            for j in range(0, width, width - 1):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]

def GPMS(sten_imjm, sten_ijm, sten_ipjm, sten_imj, 
         sten_ij, sten_ipj, sten_imjp, sten_ijp, sten_ipjp, 
         wsm, weights, ksid):
    print('interp')
    print(wsm[0]*(np.dot(weights[0,:,ksid], sten_imjm)),
            wsm[1]*(np.dot(weights[1,:,ksid], sten_ijm)),
            wsm[2]*(np.dot(weights[2,:,ksid], sten_ipjm)),
            wsm[3]*(np.dot(weights[3,:,ksid], sten_imj)),
            wsm[4]*(np.dot(weights[4,:,ksid], sten_ij)),
            wsm[5]*(np.dot(weights[5,:,ksid], sten_ipj)),
            wsm[6]*(np.dot(weights[6,:,ksid], sten_imjp)),
            wsm[7]*(np.dot(weights[7,:,ksid], sten_ijp)),
            wsm[8]*(np.dot(weights[8,:,ksid], sten_ipjp)))
    model = wsm[0]*(np.dot(weights[0,:,ksid], sten_imjm)) +\
            wsm[1]*(np.dot(weights[1,:,ksid], sten_ijm))  +\
            wsm[2]*(np.dot(weights[2,:,ksid], sten_ipjm)) +\
            wsm[3]*(np.dot(weights[3,:,ksid], sten_imj))  +\
            wsm[4]*(np.dot(weights[4,:,ksid], sten_ij))   +\
            wsm[5]*(np.dot(weights[5,:,ksid], sten_ipj))  +\
            wsm[6]*(np.dot(weights[6,:,ksid], sten_imjp)) +\
            wsm[7]*(np.dot(weights[7,:,ksid], sten_ijp))  +\
            wsm[8]*(np.dot(weights[8,:,ksid], sten_ipjp))
    print(model)
    input()
    return model

def multi_sampled_gp(width, height, lx, ly, weights, dat_in, dat_out, vec, eig, gamma, channels):
    beta = np.zeros(9)
    wsm = np.zeros(9)
    #3 Channels
    for k in range(channels):
        for i in range(2, height - 2):
            for j in range(2, width - 2):
                sten_imjm = dat_in[i-2:i+1, j-2:j+1, k].flatten('F')
                sten_ijm = dat_in[i-1:i+2, j-2:j+1, k].flatten('F')
                sten_ipjm = dat_in[i:i+3, j-2:j+1, k].flatten('F')
                sten_imj = dat_in[i-2:i+1, j-1:j+2, k].flatten('F')
                sten_ij = dat_in[i-1:i+2, j-1:j+2, k].flatten('F')
                sten_ipj = dat_in[i:i+3, j-1:j+2, k].flatten('F')
                sten_imjp = dat_in[i-2:i+1, j:j+3, k].flatten('F')
                sten_ijp = dat_in[i-1:i+2, j:j+3, k].flatten('F')
                sten_ipjp = dat_in[i:i+3, j:j+3, k].flatten('F')
                
                beta *= 0 
                for idx in range(9):
                    beta[0] += (1.0/eig[idx])*(np.dot(vec[idx], sten_imjm)**2)
                    beta[1] += (1.0/eig[idx])*(np.dot(vec[idx], sten_ijm)**2)
                    beta[2] += (1.0/eig[idx])*(np.dot(vec[idx], sten_ipjm)**2)
                    beta[3] += (1.0/eig[idx])*(np.dot(vec[idx], sten_imj)**2)
                    beta[4] += (1.0/eig[idx])*(np.dot(vec[idx], sten_ij)**2)
                    beta[5] += (1.0/eig[idx])*(np.dot(vec[idx], sten_ipj)**2)
                    beta[6] += (1.0/eig[idx])*(np.dot(vec[idx], sten_imjp)**2)
                    beta[7] += (1.0/eig[idx])*(np.dot(vec[idx], sten_ijp)**2)
                    beta[8] += (1.0/eig[idx])*(np.dot(vec[idx], sten_ipjp)**2)
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        ksid = iref + jref*ly
                        wsm *= 0
                        for idx in range(9):
                            wsm[idx] = gamma[ksid,idx]/(1.0e-36 + beta[idx])
                        wsm = wsm/sum(wsm)
                        dat_out[ii, jj, k] = GPMS(sten_imjm, sten_ijm, sten_ipjm, sten_imj, 
                                                  sten_ij, sten_ipj, sten_imjp, sten_ijp, sten_ipjp, 
                                                  wsm, weights, ksid)

#Just copy border for now
        for i in range(0,2):
            for j in range(width):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]
        for i in range(height-2,height):
            for j in range(width):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]

        for i in range(height):
            for j in range(0, 2):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]
            for j in range(width-2, width):
                for iref in range(lx):
                    ii = i * lx + iref
                    for jref in range(ly):
                        jj = j * ly + jref
                        dat_out[ii, jj, k] = dat_in[i, j, k]

