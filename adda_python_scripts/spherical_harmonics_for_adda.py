# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
#from scipy.special import sph_harm
#from scipy.special import lpmv
#from scipy.misc import factorial
import scipy as sp
import matplotlib.pyplot as plt

import shtns
#==============================================================================
#  Spherical Harmonics 
#==============================================================================


class Spharmt(object):
    """
    wrapper class for commonly used spectral transform operations in
    atmospheric models.  Provides an interface to shtns compatible
    with pyspharm (pyspharm.googlecode.com).
    """
    def __init__(self,nlons,nlats,ntrunc,rsphere,gridtype='gaussian',alpha=0):
        """initialize
        nlons:  number of longitudes
        nlats:  number of latitudes"""
        self._shtns = shtns.sht(ntrunc, ntrunc, 1, \
                                shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)        
        self._shtns.set_grid(nlats,nlons,shtns.sht_quick_init|shtns.SHT_PHI_CONTIGUOUS,1.e-10)        
        self.lats = np.arccos(self._shtns.cos_theta)
        self.lons = (2.*np.pi/nlons)*np.arange(nlons)
        self.nlons = nlons
        self.nlats = nlats
        self.ntrunc = ntrunc
        self.nlm = self._shtns.nlm
        self.degree = self._shtns.l
        self.lap = -self.degree*(self.degree+1.0).astype(np.complex)
        self.invlap = np.zeros(self.lap.shape, self.lap.dtype)
        self.invlap[1:] = 1./self.lap[1:]
        self.rsphere = rsphere
        self.lap = self.lap/rsphere**2
        self.invlap = self.invlap*rsphere**2
    def grdtospec(self,data):
        """compute spectral coefficients from gridded data"""
        return self._shtns.analys(data)
    def spectogrd(self,dataspec):
        """compute gridded data from spectral coefficients"""
        return self._shtns.synth(dataspec)
    def getuv(self,vrtspec,divspec):
        """compute wind vector from spectral coeffs of vorticity and divergence"""
        return self._shtns.synth((self.invlap/self.rsphere)*vrtspec, (self.invlap/self.rsphere)*divspec)
    def getvrtdivspec(self,u,v):
        """compute spectral coeffs of vorticity and divergence from wind vector"""
        vrtspec, divspec = self._shtns.analys(u, v)
        return self.lap*self.rsphere*vrtspec, self.lap*self.rsphere*divspec
    def getgrad(self,divspec):
        """compute gradient vector from spectral coeffs"""
        vrtspec = np.zeros(divspec.shape, dtype=np.complex)
        u,v = self._shtns.synth(vrtspec,divspec)
        return u/self.rsphere, v/self.rsphere
    def zrotate(self,dataspec,alpha):
        """rotate spectral coefficients by angle alpha"""
        RLM=self._shtns.Zrotate(dataspec,alpha)
        return RLM
    def gauss_wts(self):
        """computes gaussian weights for theta integration"""
        return self._shtns.gauss_wts()
    def idx(self,l,m):
        """get the index of the spectral array for coefficient l,m"""
        return self._shtns.idx(l,m)    



#==============================================================================
#   2D integral of regular grid
#==============================================================================
def twoD_int(K,THETA,PHI,wtheta=[]):
    #   simple function do integrate a 2d field using midpoint rule
    #   K: integral kernel
    #   THETA gridded Y component
    #   PHI gridded X component
    #
    #   important! THETA and PHI must 2d numpy array with each the same dimensions
    
    
#    Ktemp=np.sum((K[:,0:-1]+K[:,1:])/2*(PHI[:,1:]-PHI[:,0:-1]),axis=1)
#
#    atemp=np.sum((Ktemp[0:-1]+Ktemp[1:])/2*(THETA[1:,0]-THETA[0:-1,0]))
    
    
    
    Ktemp=sp.integrate.trapz(K, x=PHI, axis=1)#, even='avg')
    
    if len(wtheta)==0:
        atemp=sp.integrate.trapz(Ktemp, x=THETA)#, even='avg')
    else:
        atemp=np.sum(Ktemp*wtheta)        
    
    
#    Ktemp=sp.integrate.simps(K, x=THETA[:,0], axis=0, even='avg')
#    atemp=sp.integrate.simps(Ktemp, x=PHI[0,:], even='avg')

    return atemp


#==============================================================================
#  spherical harmonics expansion 
#==============================================================================


def sph_expansion(f_true,theta,phi,eps=0,diagnostics=False):
    # function to transform a field on a sphere into spherical harmonics
    # with given mean accuracy
    # 
    # Input:
    # f_true        field on sphere
    # theta         polar grid (colatitudes) as a vector
    # phi           azimuth grid (longitudes) as a vector
    # eps           mean accuracy
    # diagnostics   Set True if output about the expansion is wanted
    #
    # Output:
    # sph_coeffs    spherical harmonics expansion coefficients
    # x             spherical harmonics object
    # f_int         input field but interpolate to gaussian grid for theta
    # delta_temp    mean difference between spherical expansion and original field
    
    #sphere size
    rsphere = 1
    
    # grid, time step info
    nlons = len(phi)  # number of longitudes
    
    
    if len(theta) % 2 ==0:
        nlats = len(theta)   # for gaussian grid.
    elif len(theta) % 2 ==1:
        nlats = int(len(theta)-1)   # for gaussian grid.
#    
#    if nlats<nlons:
#        ntrunc = int(min([nlats,nlons])/1.)-1  # spectral truncation (for alias-free computations)
#    elif nlats>=nlons:
#        ntrunc = int(min([nlats,nlons])/2.)-1

    ntrunc=int(np.floor(nlons/2))
    
    if ntrunc>nlats-1:
        ntrunc=nlats-1
        
    
    #setup spherical harmonics object
    x = Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
    
    lons,lats = np.meshgrid(x.lons, x.lats)
    
    #get the weights for gauss legendre quadrature
    w=x.gauss_wts()
    w=np.concatenate((w,np.flip(w,axis=0)))
    
    #interpolate function to grid for spherical harmics toolbox
    f_true=np.float64(f_true)
    fun = sp.interpolate.interp1d(theta, f_true, kind='linear', axis=0)
    f_int = fun(lats[:,0])
    
#    f_int=np.float64(f_int)
    
    #transform field to spherical harmonics
    sph_coeffs=x.grdtospec(f_int)
    
    #transform spherical harmonics to field
    f_test=x.spectogrd(sph_coeffs)
    
    #estimate difference
    dtemp=np.abs(f_test-f_int)**2*np.sin(lats)
    dtemp_int=np.sqrt(twoD_int(dtemp,lats[:,0],lons[0,:],wtheta=w))    
    
    delta=np.copy(dtemp_int)
    
    if delta>=eps:
        print('=============================================')   
        print('delta            = '+str(delta))        
        print('eps              = '+str(eps))
        print('delta < eps = '+str(delta<eps))
        print('sph expansion has bigger error than set accuracy!')
        
        delta_temp=delta
        
    else:
        # reduce spherical resolution to save data space as long as delta < eps
        
        
        #prepare iteration    
        ntrunc_up=ntrunc
        ntrunc_dn=2
        
        ntrunc_new=int(round((ntrunc_up-ntrunc_dn)/2))
        
        
        #iterate by divide and conquer
        while ntrunc_new!=ntrunc_up and ntrunc_new!=ntrunc_dn:
            
            #new resolution to evaluate
            ntrunc_cn=ntrunc_new
            
            #make sph transformation
            x = Spharmt(nlons,nlats,ntrunc_cn,rsphere,gridtype='gaussian')
            
            #get the weights for gauss legendre quadrature
            w=x.gauss_wts()
            w=np.concatenate((w,np.flip(w,axis=0)))
    
            #transform field to spherical harmonics
            sph_coeffs=x.grdtospec(f_int)
            
            #transform spherical harmonics to field
            f_test=x.spectogrd(sph_coeffs)
            
            #estimate difference
            dtemp=np.abs(f_test-f_int)**2*np.sin(lats)
            dtemp_int=np.sqrt(twoD_int(dtemp,lats[:,0],lons[0,:],wtheta=w))

            delta_temp=dtemp_int
            
            
            if diagnostics==True:
                print('=============================================')   
                print('delta            = '+str(delta))
                print('delta_temp       = '+str(delta_temp))
                print('eps              = '+str(eps))
                print('delta_temp < eps = '+str(delta_temp<eps))
                print('len(sph_coeffs)  = '+str(len(sph_coeffs)))
                print('ntrunc_up        = '+str(ntrunc_up))
                print('ntrunc_dn        = '+str(ntrunc_dn))
                print('ntrunc_cn        = '+str(ntrunc_cn))        
           
            
            
            #prepare search intervall for next round
            if delta_temp>eps:
                
                ntrunc_dn=ntrunc_cn
                ntrunc_new=int(np.ceil((ntrunc_up+ntrunc_dn)/2))
            
            else:
                
                ntrunc_up=ntrunc_cn
                ntrunc_new=int(np.floor((ntrunc_up+ntrunc_dn)/2))               
                delta=delta_temp
            
            
            if diagnostics==True:
                print('ntrunc_new      = '+str(ntrunc_new))  
                print('    ')
                
                
        if delta_temp>eps:
            
            #make sph transformation
            x = Spharmt(nlons,nlats,ntrunc_new,rsphere,gridtype='gaussian')
            
            
            #get the weights for gauss legendre quadrature
            w=x.gauss_wts()
            w=np.concatenate((w,np.flip(w,axis=0)))
    
            #transform field to spherical harmonics
            sph_coeffs=x.grdtospec(f_int)
            
            #transform spherical harmonics to field
            f_test=x.spectogrd(sph_coeffs)
            
            #estimate difference
            dtemp=np.abs(f_test-f_int)**2*np.sin(lats)
            dtemp_int=np.sqrt(twoD_int(dtemp,lats[:,0],lons[0,:],wtheta=w))
            
            delta_temp=dtemp_int
            
            
            if diagnostics==True:
                print('=============================================')   
                print('delta            = '+str(delta))
                print('delta_temp       = '+str(delta_temp))
                print('eps              = '+str(eps))
                print('delta_temp < eps = '+str(delta_temp<eps))
                print('len(sph_coeffs)  = '+str(len(sph_coeffs)))
                print('ntrunc_up        = '+str(ntrunc_up))
                print('ntrunc_dn        = '+str(ntrunc_dn))
                print('ntrunc_cn        = '+str(ntrunc_cn))      
            
                
            
    return sph_coeffs, x, f_int,  delta_temp

#==============================================================================
#           start: Expand mueller_matrix in spherical harmonics 
#==============================================================================

def expand_mueller_2_spharmonics(input_data,eps_sphexp,diagnostics=False):
    # function to transform a field on a sphere into spherical harmonics
    # with given mean accuracy
    # 
    # Input:
    # input_data            dict of ADDA output, consisting of mueller matrix 
    #                       ('mueller_array'), scattered directions (azimuth:'phis_grid',
    #                       polar: 'thetas_grid'), and extinction matrix, which is unused 
    #                       here.
    # eps_sphexp            mean accuracy
    # diagnostics           Set True if output about the expansion is wanted
    #
    # Output:
    # mueller_matrix_sph    Expansion coefficients for each matrix element of
    #                       the mueller matrix
    # rel_error             relative error of the expansion for each matrix 
    #                       element of the mueller matrix
    
            
    mueller_matrix_reg=np.float64(input_data['mueller_array'])
    phis_grid=np.float64(input_data['phis_grid']*np.pi/180)
    thetas_grid=np.float64(input_data['thetas_grid']*np.pi/180)
    
    [PHI,THETA]=np.meshgrid(phis_grid,thetas_grid)
    
    #calculate variance of the first element
    M11=mueller_matrix_reg[:,:,0]
    m11_int=np.sqrt(twoD_int(M11**2*np.sin(THETA),thetas_grid,phis_grid))
    
    #accuracy based on the mean variance of the M11 element
    eps_m=m11_int*eps_sphexp
    
    #allocate
    mueller_matrix_sph = [None] * np.size(mueller_matrix_reg,axis=2)
    rel_error = [None] * np.size(mueller_matrix_reg,axis=2)
    
    max_len_coeffs=np.array([0])
    max_ntrunc=np.array([0])
    
    #expand
    for m_idx in range(0,np.size(mueller_matrix_reg,axis=2)):
    
        f_reg=mueller_matrix_reg[:,:,m_idx]
        
        [sph_coeffs, x, _, delta]=sph_expansion(
                f_reg,thetas_grid,phis_grid,eps=eps_m,diagnostics=diagnostics)
        
        mueller_matrix_sph[m_idx]=np.complex64(sph_coeffs)
        rel_error[m_idx]=delta/m11_int
        
        logic=np.array(max_len_coeffs<len(sph_coeffs))
         
        max_len_coeffs[logic]=len(sph_coeffs)
        max_ntrunc[logic]=x.ntrunc
        
    return mueller_matrix_sph,rel_error,max_len_coeffs, max_ntrunc


#==============================================================================
#   visualize mueller matrix sph
#==============================================================================

def visualize_mueller_matrix_sph(sph_coeffs,sph_object, max_len_coeffs, max_ntrunc):
#
#function to visualize sph expansion of mueller matrix    
    
    
    if max_ntrunc % 2 ==0:
        min_grid_size=max_ntrunc+2
    else:
        min_grid_size=max_ntrunc+1
        
    if min_grid_size<32:
        min_grid_size=32    

    x = Spharmt(min_grid_size*2,min_grid_size,max_ntrunc,1,gridtype='gaussian')
        
    xlons,xlats = np.meshgrid(x.lons, x.lats)
    
    
    fig = plt.figure()
    
    
    sph_coeffs_null=np.zeros(max_len_coeffs,dtype=complex)
    
    if len(np.shape(sph_coeffs))==1:
        
        temp=sph_coeffs_null.copy()
        temp[0:len(sph_coeffs)]=np.complex128(sph_coeffs)
    
        mjj=x.spectogrd(temp)
        
        plt.subplot(1, 1, 1)
        h1=plt.pcolormesh(xlons,xlats,mjj)
        
        plt.ylabel('\theta [deg]')
        plt.xlabel('\phi [deg]')        
        fig.colorbar(h1)

    else:
        
        
        for j in range(0,np.size(sph_coeffs,axis=0)):
            
            temp=sph_coeffs_null.copy()
            temp[0:len(sph_coeffs[j,:])]=np.complex128(sph_coeffs[j,:])
            
            mjj=x.spectogrd(temp)
            
            plt.subplot(4, 4, j+1)
            h1=plt.pcolormesh(xlons,xlats,mjj)
            
            if j==0:
                plt.ylabel('\theta [deg]')
                
            if j==15: 
                plt.xlabel('\phi [deg]')
            
            fig.colorbar(h1)
        
    
#==============================================================================
#   map spherical expansion to another spherical expansion    
#==============================================================================

def map_sphexp(sph_coeffs,sph_object):
    #Function to map from one spherical harmonics expansion to another 
    #Important the expansion, which is mapped to another has to be
    # of smaller or same lmax as the expansion, to which it is mapped.
    
    #get truncation or lmax
    ntrunc=int(1/2+np.sqrt(1/4+2*len(sph_coeffs))-2)

    #create local spherical object
#    sph_local=shtns.sht(ntrunc, ntrunc, 1, shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
#    local_m=sph_local.m
#    local_l=sph_local.l

    #build array with the l coefficients   
    local_l=np.zeros(len(sph_coeffs),dtype=int)    
    start=0
    end=0
    for i in range(0,ntrunc+1):
        
        lnew=np.arange(i,ntrunc+1,1)
        
        end=end+len(lnew)
        
        local_l[start:end]=lnew
        
        start=end
        
    

    
    #build array with the m coefficients   
    local_m=np.zeros(len(sph_coeffs),dtype=int)    
    start=0
    end=0
    for i in range(0,ntrunc+1):
        
        mnew=np.ones(ntrunc+1-i)*i
        
        end=end+len(mnew)
        
        local_m[start:end]=mnew
        
        start=end


    #allocate
    sph_coeffs_new=np.zeros(sph_object.nlm,dtype=complex)
    
    
    if sph_object.nlm<len(sph_coeffs):
        
        RuntimeError('target expansion has less elements than source expansion!')
    
    #do the mapping
    for i in range(0,len(local_l)):
        
        idx=sph_object.idx(int(local_l[i]),int(local_m[i]))
        
        sph_coeffs_new[idx]=sph_coeffs[i]
        
        
        
        
    return sph_coeffs_new        
        
        
    


    
    