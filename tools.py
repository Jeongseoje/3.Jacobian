'''
Code developer : Sumin Kim in Korea Polar Research Institute
These codes are free. You can use and modify these codes for your own works.
However, if you publish your research work with these code, 
please contact me by E-mail : seismin@kopri.re.kr
Enjoy with these codes
'''




import numpy as np
#from .class_first_par import par

def vel_exp(nx,nz,no,npml,rtype,vel):
    nx_e = nx + 2*npml
    nz_e = nz + 2*npml
    
    nx_a = nx_e + 2*no
    nz_a = nz_e + 2*no
    
    vp_e=np.zeros((nz_a,nx_a),dtype=rtype)
    vp_e[no+npml:no+npml+nz,no+npml:no+npml+nx]=vel
    vp_e[:no+npml,no+npml:no+npml+nx]=vel[0,:]
    vp_e[no+npml+nz-1:,no+npml:no+npml+nx]=vel[nz-1,:]
    for ix in np.arange(no+npml):
        vp_e[no+npml:no+npml+nz,ix]=vel[:,0]
        vp_e[no+npml:no+npml+nz,no+npml+nx+ix]=vel[:,nx-1]
        for iz in np.arange(no+npml):
            vp_e[iz,ix]=vel[0,0]
            vp_e[iz,no+npml+nx+ix]=vel[0,nx-1]
            vp_e[no+npml+nz+iz,ix]=vel[nz-1,0]
            vp_e[no+npml+nz+iz,no+npml+nx+ix]=vel[nz-1,nx-1]
    return vp_e

def ass_shot(nshot,fshot_x,dshot_x,fshot_z,dshot_z,npml,no): 
    shot_x=np.zeros((nshot),dtype=np.int32)
    shot_z=np.zeros((nshot),dtype=np.int32)
    for ir in np.arange(nshot):
        shot_x[ir]=npml+fshot_x+dshot_x*ir+no  
    for ir in np.arange(nshot):
        shot_z[ir] = npml+fshot_z+dshot_z+no
    return shot_x,shot_z    


def ass_rcv(ishot,nrcv,frcv_x,drcv_x,frcv_z,drcv_z,npml,no,geom_ID): 
    rcv_x=np.zeros((nrcv),dtype=np.int32)
    rcv_z=np.zeros((nrcv),dtype=np.int32)
    if(geom_ID==1):
        for ir in np.arange(nrcv):
            rcv_x[ir]=npml+frcv_x+drcv_x*ir +no
    elif(geom_ID==2): # streamer type
        for ir in np.arange(nrcv):
            rcv_x[ir]=shot_x[ishot] + noffset + drcv_x*ir
    elif(geom_ID==3): # split spread type
        for ir in np.arange(nrcv):
            rcv_x[ir] = (shot_x[ishot]-int(nrcv/2))+drcv_x*ir
    else:
        print( "Choose the correct option")
    for ir in np.arange(nrcv):
        rcv_z[ir] = npml+frcv_z + drcv_z+no    
    return rcv_x,rcv_z

def ricker(nt,dt,fmax,rtype):
    fdom=fmax/3;
    factor=1; t0=0.2#1.2/fdom; 
    a=(np.pi*fdom)**2
    ricker=np.zeros((nt),rtype)
    for it in np.arange(nt):
        dm = a*((it*dt-t0)**2)
        ricker[it] = factor*(1-(2*dm))*np.exp(-dm)
    source=ricker
    return source 
