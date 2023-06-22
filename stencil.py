'''
Code developer : Sumin Kim in Korea Polar Research Institute
These codes are free. You can use and modify these codes for your own works.
However, if you publish your research work with these code, 
please contact me by E-mail : seismin@kopri.re.kr
Enjoy with these codes
'''

import numpy as np


def deriv_pr_x(C,n,nx_a,nz_a,dx,inarr):
    outarr  =C[0]*(inarr[n:nz_a-n,(n+1):nx_a-(n-1)]-inarr[n:nz_a-n,(n-0):nx_a-(n+0)]) + \
             C[1]*(inarr[n:nz_a-n,(n+2):nx_a-(n-2)]-inarr[n:nz_a-n,(n-1):nx_a-(n+1)]) 
 
    outarr/=dx
    return outarr

def deriv_pr_z(C,n,nx_a,nz_a,dx,inarr):
    outarr  =C[0]*(inarr[(n+1):nz_a-(n-1),n:nx_a-n]-inarr[(n-0):nz_a-(n+0),n:nx_a-n]) + \
             C[1]*(inarr[(n+2):nz_a-(n-2),n:nx_a-n]-inarr[(n-1):nz_a-(n+1),n:nx_a-n]) 
    outarr/=dx
    return outarr

def deriv_vx_x(C,n,nx_a,nz_a,dx,inarr):
    outarr  =C[0]*(inarr[n:nz_a-n,(n+0):nx_a-(n-0)]-inarr[n:nz_a-n,(n-1):nx_a-(n+1)]) + \
             C[1]*(inarr[n:nz_a-n,(n+1):nx_a-(n-1)]-inarr[n:nz_a-n,(n-2):nx_a-(n+2)]) 

    outarr/=dx
    return outarr

def deriv_vz_z(C,n,nx_a,nz_a,dx,inarr):
    outarr  =C[0]*(inarr[(n+0):nz_a-(n-0),n:nx_a-n]-inarr[(n-1):nz_a-(n+1),n:nx_a-n]) + \
             C[1]*(inarr[(n+1):nz_a-(n-1),n:nx_a-n]-inarr[(n-2):nz_a-(n+2),n:nx_a-n]) 

    outarr/=dx
    return outarr




def apply_cpml_x(ax,bx,dum,inarr):
    dum=dum*bx + ax*inarr
    inarr+=dum
    return dum,inarr
def apply_cpml_z(az,bz,dum,inarr):
    dum=(dum.T*bz).T + (inarr.T*az).T
    inarr+=dum
    return dum,inarr