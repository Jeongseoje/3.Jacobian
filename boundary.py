'''
Code developer : Sumin Kim in Korea Polar Research Institute
These codes are free. You can use and modify these codes for your own works.
However, if you publish your research work with these code, 
please contact me by E-mail : seismin@kopri.re.kr
Enjoy with these codes
'''


import numpy as np


def gen_cpml(npml, fmax, dx,dt,nx_e,nz_e, rtype):
    freq=fmax
    rpml=(npml)*dx # rmpl=thickness
    Rcoef=0.0001
    K_max_PML = 1. 
    alpha_max_pml = 2*np.pi*(freq/2)
    vmax=5000;
    npower=3

    d_x=np.zeros((nx_e),rtype); 
    d_xh=np.zeros((nx_e),rtype)
    k_x=np.ones((nx_e),rtype); 
    k_xh=np.ones((nx_e),rtype)
    alpha_x=np.zeros((nx_e),rtype);
    alpha_xh=np.zeros((nx_e),rtype)
    a_x=np.zeros((nx_e),rtype); 
    a_xh=np.zeros((nx_e),rtype)
    b_x=np.zeros((nx_e),rtype); 
    b_xh=np.zeros((nx_e),rtype)

    d_z=np.zeros((nz_e),rtype); 
    d_zh=np.zeros((nz_e),rtype)
    k_z=np.ones((nz_e),rtype); 
    k_zh=np.ones((nz_e),rtype)
    alpha_z=np.zeros((nz_e),rtype); 
    alpha_zh=np.zeros((nz_e),rtype)
    a_z=np.zeros((nz_e),rtype); 
    a_zh=np.zeros((nz_e),rtype)
    b_z=np.zeros((nz_e),rtype); 
    b_zh=np.zeros((nz_e),rtype)

    xori_left=rpml; xori_rig=nx_e*dx-rpml;
    for ix in np.arange(nx_e):
        d0_x = -(npower+1)*vmax*np.log(Rcoef)/(2*rpml)
        xval=dx*ix
    # left edge
        xpos = xori_left-xval #xpos=abscicaa
        if(xpos>=0):
            xpos_nor=xpos/rpml
            d_x[ix] = d0_x*xpos_nor**npower
            k_x[ix] = 1+(K_max_PML-1)*xpos_nor**npower
            alpha_x[ix] = alpha_max_pml*(1-xpos_nor)
        xpos = xori_left-(xval+dx/2)
        if(xpos>=0):
            xpos_nor = xpos/rpml
            d_xh[ix] = d0_x*xpos_nor**npower
            k_xh[ix] = 1+(K_max_PML-1)*xpos_nor**npower
            alpha_xh[ix] = alpha_max_pml*(1-xpos_nor)
                        # right edge
        xpos = xval-xori_rig+dx
        if(xpos>=0):
            xpos_nor = xpos/rpml
            d_x[ix]=d0_x*xpos_nor**npower
            k_x[ix]=1+(K_max_PML-1)*xpos_nor**npower
            alpha_x[ix]=alpha_max_pml*(1-xpos_nor)
        xpos=xval+dx/2-xori_rig+dx
        if(xpos>=0):
            xpos_nor=xpos/rpml
            d_xh[ix]=d0_x*xpos_nor**npower
            k_xh[ix]=1+(K_max_PML-1)*xpos_nor**npower
            alpha_xh[ix]=alpha_max_pml*(1-xpos_nor)
        if(alpha_x[ix]<0):
            alpha_x[ix]=0
        if(alpha_xh[ix]<0):
            alpha_xh[ix]=0
        b_x[ix] = np.exp(-(d_x[ix]/k_x[ix]+alpha_x[ix])*dt)
        b_xh[ix] = np.exp(-(d_xh[ix]/k_xh[ix]+alpha_xh[ix])*dt)
        if(abs(d_x[ix])>0.0000001):
            a_x[ix] = d_x[ix]*(b_x[ix]-1)/(k_x[ix]*(d_x[ix]+k_x[ix]*alpha_x[ix]))
        if(abs(d_xh[ix])>0.0000001):
            a_xh[ix] = d_xh[ix]*(b_xh[ix]-1)/(k_xh[ix]*(d_xh[ix]+k_xh[ix]*alpha_xh[ix]))

    # z direction
    for iz in np.arange(nz_e):
        d0_z = -(npower+1)*vmax*np.log(Rcoef)/(2*rpml)
        zval=dx*(iz)
        
        zori_left=rpml; 
        zori_rig=nz_e*dx-rpml;
        # up edge
        zpos = zori_left-zval #xpos=abscicaa
        if(zpos>=0):
            zpos_nor=zpos/rpml
        d_z[iz] = d0_z*zpos_nor**npower
        k_z[iz] = 1+(K_max_PML-1)*zpos_nor**npower
        alpha_z[iz] = alpha_max_pml*(1-zpos_nor)
        zpos = zori_left-(zval+dx/2)
        if(zpos>=0):
            zpos_nor = zpos/rpml
            d_zh[iz] = d0_z*zpos_nor**npower
            k_zh[iz] = 1+(K_max_PML-1)*zpos_nor**npower
            alpha_zh[iz] = alpha_max_pml*(1-zpos_nor)
        # right edge
        zpos = zval-zori_rig+dx
        if(zpos>=0):
            zpos_nor = zpos/rpml
            d_z[iz]=d0_z*zpos_nor**npower
            k_z[iz]=1+(K_max_PML-1)*zpos_nor**npower
            alpha_z[iz]=alpha_max_pml*(1-zpos_nor)
        zpos=zval+dx/2-zori_rig+dx
        if(zpos>=0):
            zpos_nor=zpos/rpml
            d_zh[iz]=d0_z*zpos_nor**npower
            k_zh[iz]=1+(K_max_PML-1)*zpos_nor**npower
            alpha_zh[iz]=alpha_max_pml*(1-zpos_nor)
        if(alpha_z[iz]<0):
            alpha_z[iz]=0
        if(alpha_zh[iz]<0):
            alpha_zh[iz]=0
        b_z[iz] = np.exp(-(d_z[iz]/k_z[iz]+alpha_z[iz])*dt)
        b_zh[iz] = np.exp(-(d_zh[iz]/k_zh[iz]+alpha_zh[iz])*dt)
        if(abs(d_z[iz])>0.0000001):
            a_z[iz] = d_z[iz]*(b_z[iz]-1)/(k_z[iz]*(d_z[iz]+k_z[iz]*alpha_z[iz]))
        if(abs(d_zh[iz])>0.0000001):
            a_zh[iz] = d_zh[iz]*(b_zh[iz]-1)/(k_zh[iz]*(d_zh[iz]+k_zh[iz]*alpha_zh[iz]))
            
    return a_x,a_xh,a_z,a_zh,b_x,b_xh,b_z,b_zh
        