'''
Code developer : Sumin Kim in Korea Polar Research Institute
These codes are free. You can use and modify these codes for your own works.
However, if you publish your research work with these code, 
please contact me by E-mail : seismin@kopri.re.kr
Enjoy with these codes
'''


import numpy as np 
from stencil import *
from scipy.ndimage import gaussian_filter
def forward(ishot,vp,
            nx,nz,nt,npml,no,
            dx,dz,dt,
            source,
            sx,sz,rx,rz,
            nrcv,
            C,
           a_x,a_xh,a_z,a_zh,
           b_x,b_xh,b_z,b_zh,rtype):
    
    nx_e      = nx + 2*npml
    nz_e      = nz + 2*npml
    nx_a      = nx_e + 2*no
    nz_a      = nz_e + 2*no
    
    rho       = np.ones((nz_e,nx_e),rtype)
    pr        = np.zeros((nz_a,nx_a),rtype); 
    vx        = np.zeros_like(pr);vz=np.zeros_like(pr)
    dum_pr_x  = np.zeros((nz_e,nx_e),rtype); 
    dum_pr_z  = np.copy(dum_pr_x); 
    dum_vx_x  = np.copy(dum_pr_z); 
    dum_vz_z  = np.copy(dum_pr_z)
    sg_p      = np.zeros((nt,nrcv),rtype)
    u1        = np.zeros((nz,nx),rtype);
    u2        = np.zeros_like(u1);u3=np.zeros_like(u1)
    rho_hx    = np.zeros((nz_e,nx_e),rtype);
    rho_hz    = np.zeros_like(rho_hx)
    
    for ix in np.arange(nx_e-1):
        rho_hx[:,ix]=0.5*(rho[:,ix]+rho[:,ix+1])
    for iz in np.arange(nz_e-1):
        rho_hz[iz,:]=0.5*(rho[iz,:]+rho[iz+1,:])
        rho_hx[:,nx_e-1]=np.copy(rho_hx[:,nx_e-2])
        rho_hz[nz_e-1,:]=np.copy(rho_hz[nz_e-2,:])   
        
    isx=sx[ishot]; isz=sz[ishot]
    wave = np.zeros((nt,nz_e,nx_e),rtype)
     
    ircv_x_val=rx-npml-no
    for it in np.arange(nt):
        temp1=deriv_pr_x(C,no,nx_a,nz_a,dx,pr)
        dum_pr_x,temp1=apply_cpml_x(a_xh,b_xh,dum_pr_x,temp1)
        vx[no:no+nz_e,no:no+nx_e] = vx[no:no+nz_e,no:no+nx_e]+(dt/rho)*temp1
                # Vz
        temp2=deriv_pr_z(C,no,nx_a,nz_a,dx,pr)
        dum_pr_z,temp2=apply_cpml_z(a_zh,b_zh,dum_pr_z,temp2)     
        vz[no:no+nz_e,no:no+nx_e] = vz[no:no+nz_e,no:no+nx_e] + (dt/rho)*temp2
                    # Presure
        temp3=deriv_vx_x(C,no,nx_a,nz_a,dx,vx)
        dum_vx_x,temp3=apply_cpml_x(a_x,b_x,dum_vx_x,temp3)
        temp4=deriv_vz_z(C,no,nx_a,nz_a,dx,vz)
        dum_vz_z,temp4=apply_cpml_z(a_z,b_z,dum_vz_z,temp4)
        pr[no:no+nz_e,no:no+nx_e] = pr[no:no+nz_e,no:no+nx_e]+(vp**2*rho)*(dt)*(temp3+temp4)
                    # add source

        pr[isz,isx] += source[it]* \
                        (rho[isz-no,isx-no]*vp[isz-no,isx-no]**2)*dt
        
        wave[it,:,:] = pr[no:no+nz_e,no:no+nx_e]
                    # save seismogratm

        for ir in range(nrcv):
            if(ircv_x_val[ir]>=0 and ircv_x_val[ir]<nx ):
                sg_p[it,ir]=pr[rz[ir],rx[ir]]
    return sg_p,wave

def forward_fwi(ishot,vp,
            nx,nz,nt,npml,no,
            dx,dz,dt,
            source,
            sx,sz,rx,rz,
            nrcv,
            C,
           a_x,a_xh,a_z,a_zh,
           b_x,b_xh,b_z,b_zh,rtype):
    
    nx_e      = nx + 2*npml
    nz_e      = nz + 2*npml
    nx_a      = nx_e + 2*no
    nz_a      = nz_e + 2*no
    
    rho       = np.ones((nz_e,nx_e),rtype)
    pr        = np.zeros((nz_a,nx_a),rtype); 
    vx        = np.zeros_like(pr);vz=np.zeros_like(pr)
    dum_pr_x  = np.zeros((nz_e,nx_e),rtype); 
    dum_pr_z  = np.copy(dum_pr_x); 
    dum_vx_x  = np.copy(dum_pr_z); 
    dum_vz_z  = np.copy(dum_pr_z)
    sg_p      = np.zeros((nt,nrcv),rtype)
    u1        = np.zeros((nz,nx),rtype);
    u2        = np.zeros_like(u1);u3=np.zeros_like(u1)
    rho_hx    = np.zeros((nz_e,nx_e),rtype);
    rho_hz    = np.zeros_like(rho_hx)
    
    for ix in np.arange(nx_e-1):
        rho_hx[:,ix]=0.5*(rho[:,ix]+rho[:,ix+1])
    for iz in np.arange(nz_e-1):
        rho_hz[iz,:]=0.5*(rho[iz,:]+rho[iz+1,:])
        rho_hx[:,nx_e-1]=np.copy(rho_hx[:,nx_e-2])
        rho_hz[nz_e-1,:]=np.copy(rho_hz[nz_e-2,:])   
        
    isx=sx[ishot]; isz=sz[ishot]
    wave = np.zeros((nt,nz_e,nx_e),rtype)
     
    ircv_x_val=rx-npml-no
    
    
    u3 = np.zeros_like(pr)
    u2 = np.zeros_like(pr)
    
    for it in np.arange(nt):
        temp1=deriv_pr_x(C,no,nx_a,nz_a,dx,pr)
        dum_pr_x,temp1=apply_cpml_x(a_xh,b_xh,dum_pr_x,temp1)
        vx[no:no+nz_e,no:no+nx_e] = vx[no:no+nz_e,no:no+nx_e]+(dt/rho)*temp1
                # Vz
        temp2=deriv_pr_z(C,no,nx_a,nz_a,dx,pr)
        dum_pr_z,temp2=apply_cpml_z(a_zh,b_zh,dum_pr_z,temp2)     
        vz[no:no+nz_e,no:no+nx_e] = vz[no:no+nz_e,no:no+nx_e] + (dt/rho)*temp2
                    # Presure
        temp3=deriv_vx_x(C,no,nx_a,nz_a,dx,vx)
        dum_vx_x,temp3=apply_cpml_x(a_x,b_x,dum_vx_x,temp3)
        temp4=deriv_vz_z(C,no,nx_a,nz_a,dx,vz)
        dum_vz_z,temp4=apply_cpml_z(a_z,b_z,dum_vz_z,temp4)
        pr[no:no+nz_e,no:no+nx_e] = pr[no:no+nz_e,no:no+nx_e]+(vp**2*rho)*(dt)*(temp3+temp4)
                    # add source

        pr[isz,isx] += source[it]* \
                        (rho[isz-no,isx-no]*vp[isz-no,isx-no]**2)*dt
        u3 = pr.copy()
        
        wave[it,:,:] = 2*(u3-u2 )[no:no+nz_e,no:no+nx_e] / dt /(vp**3*rho)

        for ir in range(nrcv):
            if(ircv_x_val[ir]>=0 and ircv_x_val[ir]<nx ):
                sg_p[it,ir]=pr[rz[ir],rx[ir]]
                
        u2 = u3.copy()
        u3 = np.zeros_like(pr)
    return sg_p,wave


def adjoint_fwi(ishot,vp,
            nx,nz,nt,npml,no,
            dx,dz,dt,
            data,
            swave,
            sx,sz,rx,rz,
            nrcv,
            C,
           a_x,a_xh,a_z,a_zh,
           b_x,b_xh,b_z,b_zh,rtype):
    
    nx_e      = nx + 2*npml
    nz_e      = nz + 2*npml
    nx_a      = nx_e + 2*no
    nz_a      = nz_e + 2*no
    
    rho       = np.ones((nz_e,nx_e),rtype)
    pr        = np.zeros((nz_a,nx_a),rtype); 
    vx        = np.zeros_like(pr);vz=np.zeros_like(pr)
    dum_pr_x  = np.zeros((nz_e,nx_e),rtype); 
    dum_pr_z  = np.copy(dum_pr_x); 
    dum_vx_x  = np.copy(dum_pr_z); 
    dum_vz_z  = np.copy(dum_pr_z)
    sg_p      = np.zeros((nt,nrcv),rtype)
    u1        = np.zeros((nz,nx),rtype);
    u2        = np.zeros_like(u1);u3=np.zeros_like(u1)
    rho_hx    = np.zeros((nz_e,nx_e),rtype);
    rho_hz    = np.zeros_like(rho_hx)
    
    for ix in np.arange(nx_e-1):
        rho_hx[:,ix]=0.5*(rho[:,ix]+rho[:,ix+1])
    for iz in np.arange(nz_e-1):
        rho_hz[iz,:]=0.5*(rho[iz,:]+rho[iz+1,:])
        rho_hx[:,nx_e-1]=np.copy(rho_hx[:,nx_e-2])
        rho_hz[nz_e-1,:]=np.copy(rho_hz[nz_e-2,:])   
        
    isx=sx[ishot]; isz=sz[ishot]
    bwave = np.zeros((nt,nz,nx),rtype)
    grad = np.zeros((nz,nx),rtype)
    ircv_x_val=rx-npml-no
    
    
    u3 = np.zeros_like(pr)
    u2 = np.zeros_like(pr)
    
    for it in np.arange(nt):
        temp1=deriv_pr_x(C,no,nx_a,nz_a,dx,pr)
        dum_pr_x,temp1=apply_cpml_x(a_xh,b_xh,dum_pr_x,temp1)
        vx[no:no+nz_e,no:no+nx_e] = vx[no:no+nz_e,no:no+nx_e]+(dt/rho)*temp1
                # Vz
        temp2=deriv_pr_z(C,no,nx_a,nz_a,dx,pr)
        dum_pr_z,temp2=apply_cpml_z(a_zh,b_zh,dum_pr_z,temp2)     
        vz[no:no+nz_e,no:no+nx_e] = vz[no:no+nz_e,no:no+nx_e] + (dt/rho)*temp2
                    # Presure
        temp3=deriv_vx_x(C,no,nx_a,nz_a,dx,vx)
        dum_vx_x,temp3=apply_cpml_x(a_x,b_x,dum_vx_x,temp3)
        temp4=deriv_vz_z(C,no,nx_a,nz_a,dx,vz)
        dum_vz_z,temp4=apply_cpml_z(a_z,b_z,dum_vz_z,temp4)
        pr[no:no+nz_e,no:no+nx_e] = pr[no:no+nz_e,no:no+nx_e]+(vp**2*rho)*(dt)*(temp3+temp4)
                    # add source
        for ir in range(nrcv):
            if(ircv_x_val[ir]>=0 and ircv_x_val[ir]<nx ):
                pr[rz[ir],rx[ir]]+=data[nt-it-1,ir]
                
        wav_for = swave[nt-it-1,:,:]
        wav_adj = pr[no+npml:no+npml+nz,no+npml:no+npml+nx]
        grad += wav_for*wav_adj
        bwave[nt-it-1,:,:] = wav_adj
    return grad,bwave




def adjoint_rfwi(ishot,vp,
            nx,nz,nt,npml,no,
            dx,dz,dt,
            data,
            swave,
            sx,sz,rx,rz,
            nrcv,
            C,
           a_x,a_xh,a_z,a_zh,
           b_x,b_xh,b_z,b_zh,rtype,option):
    
    nx_e      = nx + 2*npml
    nz_e      = nz + 2*npml
    nx_a      = nx_e + 2*no
    nz_a      = nz_e + 2*no
    
    rho       = np.ones((nz_e,nx_e),rtype)
    pr        = np.zeros((nz_a,nx_a),rtype); 
    vx        = np.zeros_like(pr);vz=np.zeros_like(pr)
    dum_pr_x  = np.zeros((nz_e,nx_e),rtype); 
    dum_pr_z  = np.copy(dum_pr_x); 
    dum_vx_x  = np.copy(dum_pr_z); 
    dum_vz_z  = np.copy(dum_pr_z)
    sg_p      = np.zeros((nt,nrcv),rtype)
    u1        = np.zeros((nz,nx),rtype);
    u2        = np.zeros_like(u1);u3=np.zeros_like(u1)
    rho_hx    = np.zeros((nz_e,nx_e),rtype);
    rho_hz    = np.zeros_like(rho_hx)
    
    for ix in np.arange(nx_e-1):
        rho_hx[:,ix]=0.5*(rho[:,ix]+rho[:,ix+1])
    for iz in np.arange(nz_e-1):
        rho_hz[iz,:]=0.5*(rho[iz,:]+rho[iz+1,:])
        rho_hx[:,nx_e-1]=np.copy(rho_hx[:,nx_e-2])
        rho_hz[nz_e-1,:]=np.copy(rho_hz[nz_e-2,:])   
        
    isx=sx[ishot]; isz=sz[ishot]
    
    grad = np.zeros((nz,nx),rtype)
    ircv_x_val=rx-npml-no
    
    
    u3 = np.zeros_like(pr)
    u2 = np.zeros_like(pr)
    
    for it in np.arange(nt):
        temp1=deriv_pr_x(C,no,nx_a,nz_a,dx,pr)
        dum_pr_x,temp1=apply_cpml_x(a_xh,b_xh,dum_pr_x,temp1)
        vx[no:no+nz_e,no:no+nx_e] = vx[no:no+nz_e,no:no+nx_e]+(dt/rho)*temp1
                # Vz
        temp2=deriv_pr_z(C,no,nx_a,nz_a,dx,pr)
        dum_pr_z,temp2=apply_cpml_z(a_zh,b_zh,dum_pr_z,temp2)     
        vz[no:no+nz_e,no:no+nx_e] = vz[no:no+nz_e,no:no+nx_e] + (dt/rho)*temp2
                    # Presure
        temp3=deriv_vx_x(C,no,nx_a,nz_a,dx,vx)
        dum_vx_x,temp3=apply_cpml_x(a_x,b_x,dum_vx_x,temp3)
        temp4=deriv_vz_z(C,no,nx_a,nz_a,dx,vz)
        dum_vz_z,temp4=apply_cpml_z(a_z,b_z,dum_vz_z,temp4)
        pr[no:no+nz_e,no:no+nx_e] = pr[no:no+nz_e,no:no+nx_e]+(vp**2*rho)*(dt)*(temp3+temp4)
                    # add source
        for ir in range(nrcv):
            if(ircv_x_val[ir]>=0 and ircv_x_val[ir]<nx ):
                pr[rz[ir],rx[ir]]+=data[nt-it-1,ir]
                
        
        wav_for = swave[nt-it-1,:,:]
        wav_adj = pr[no+npml:no+npml+nz,no+npml:no+npml+nx]
        
        wav_for_h = hilb_z(wav_for)
        wav_adj_h = hilb_z(wav_adj)
        
        gg1 = wav_for*wav_adj
        gg2 = wav_for_h * wav_adj_h
        if option == 'low':
            grad += (gg1 + gg2)
        else:
            grad += (gg1 - gg2)
        
        
    if option == 'low':
        grad = gaussian_filter(grad,sigma=10)
    
    return grad




def hilb_z(data):
    fil_hflen=51; fil_len=2*fil_hflen+1;
    fil = np.zeros((fil_len))
    for iz in np.arange(1,fil_hflen):
        taper=0.54+0.46*np.cos(np.pi*iz/fil_hflen);
        fil[fil_hflen+iz]=taper*(-(iz%2)*2.0/(np.pi*iz))
        fil[fil_hflen-iz]=-fil[fil_hflen+iz]
    # do hilbert
    output=np.zeros((data.shape))
    for ix in np.arange(data.shape[1]):
        output[:,ix] = np.convolve(fil,data[:,ix],'same')
    return output