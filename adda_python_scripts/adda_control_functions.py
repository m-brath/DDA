# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:24:44 2015

@author: u242031
"""


import subprocess as sub
import numpy as np
#import scipy as sp
import datetime
import glob
import os
import shutil
#import pdb
import tempfile

import zipfile
try:
#    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored'}

from netCDF4 import Dataset

try:
    import shtns as _shtns
except ImportError:          # storage helpers degrade gracefully without it
    _shtns = None

from posix_spawn import spawn_command


#==============================================================================
# Extinction matrix
#==============================================================================




def extinction_matrix(s11,s12,s21,s22,lamb):
    #Function for calculation of extinction matrix from scattering amplitude matrix
    #according to Mishenko et al.,2002

    #to be consistent with the Mishchenko definitions the scattering
    #amplitudes from ADDA have to multiplicated with
    # 1/(-ik)=lamb/(-2*pi*i)
    s11=s11*lamb/(-1j*2*np.pi)
    s12=s12*lamb/(-1j*2*np.pi)
    s21=s21*lamb/(-1j*2*np.pi)
    s22=s22*lamb/(-1j*2*np.pi)



    #Diagonal elements
    K=np.identity(4)*np.imag(s11+s22)

    #Off diagonals
    K[0,1]=np.imag(s11-s22)
    K[1,0]=K[0,1]

    K[0,2]=-np.imag(s12+s21)
    K[2,0]=K[0,2]

    K[3,0]=np.real(s21-s12)
    K[0,3]=K[3,0]

    K[1,2]=np.imag(s21-s12)
    K[2,1]=-K[1,2]

    K[1,3]=-np.real(s12+s21)
    K[3,1]=-K[1,3]

    K[2,3]=np.real(s22-s11)
    K[3,2]=-K[2,3]

    #finally,  in Mishenko was written 2*pi/k, which is equal to the wavelength
    K=K*lamb


    return K

#==============================================================================
#  Size-parameter-adaptive scattering grid (Phase 8)
#------------------------------------------------------------------------------
#  The angular structure of any scattered field is band-limited by the number
#  of significant partial waves, ~ x + 4x^(1/3) + 2 (Wiscombe 1979-80 style
#  criterion; x = pi*Dveq/lambda). Orientation averaging only makes fields
#  smoother. Hence the ADDA (-scat_grid) output grid does NOT need a fixed
#  181x361 sampling at small x — it should scale with the size parameter.
#
#  The grid must still provide everything the pipeline extracts from it:
#    - an exact theta=0 row (forward ampl_scatgrid column -> extinction matrix)
#    - an exact theta=180 point + full phi ring (Phase 6 backscatter matrix)
#    - phi range 0..360 with BOTH endpoints (historic production convention;
#      parsing does np.unique + reshape, mirror_phi_index_map works on any
#      uniform ring)
#  spha.sph_expansion caps its initial ntrunc at min(floor(nphi/2),
#  nlats_eff-1), nlats_eff = ntheta (ntheta even) or ntheta-1 (odd). The
#  sizes returned by scat_grid_sizes() realize an effective cap of exactly
#  'ntrunc_cap' with the production convention reproduced at
#  ntrunc_cap=179/180 (-> 181x361).
#
#  NOTE on the error metric: spha.sph_expansion measures its round-trip
#  residual on the *interpolated Gaussian* grid, so a too-coarse theta grid
#  hides linear-interpolation error from the criterion. The safety 'margin'
#  is therefore a physical Nyquist margin, calibrated by the twin gate
#  analysis/phase8_scatgrid_twin.py — not the eps criterion.
#==============================================================================

def sph_ntrunc_physical(x):
    #physical band-limit of the scattering angles in SH degree (partial-wave
    #count); smooth, continuous, monotonically increasing in x
    x=max(float(x), 0.)
    return x + 4.0*np.cbrt(x) + 2.0


def scatgrid_ntrunc_cap(x, margin=6.0, floor_ntrunc=8):
    #initial shtns truncation cap that an adaptive grid should realize;
    #margin is a Nyquist/safety factor on the physical band-limit. The
    #calibrated default 6.0 comes from the Phase-8 twin gate
    #(analysis/phase8_scatgrid_twin.py): the ACTUAL eps_sphexp-limited
    #stored ntrunc is 10 (x=0.02), 26 (x=0.5), 81 (x=5), 48 (x=10) — i.e.
    #3-6x the raw Wiscombe count — and margin 6 keeps the grid cap above
    #that for all sizes (margin 4 truncates the eps target at x~0.5-5).
    return int(max(float(floor_ntrunc),
                   np.ceil(margin*sph_ntrunc_physical(x))))


def scat_grid_sizes(ntrunc_cap, x=0.0, kappa=0.17, theta_mode='uniform'):
    #(ntheta, nphi) for the ADDA scat_params grid.
    #nphi = 2*cap+1 gives floor(nphi/2) = cap exactly (Fourier/phi content
    #follows the partial-wave SH cap).
    #theta resolution is a SEPARATE, scale-invariant requirement: angular
    #features of a scatterer at size parameter x have width ~1/x rad (the
    #diffraction peak plus the propagation-attached polarization-frame
    #structure at theta=0/180), so linear-interpolation-safe sampling needs
    #~2pi/kappa samples per feature: ntheta ~ pi*x/kappa + 1. The calibrated
    #default kappa=0.17 rad reproduces the production 181-point grid at the
    #x=10 size-parameter limit exactly (18.5*10+1 ~ 186, capped at 181), so
    #the adaptive grid is never finer and never coarser-in-x than today's
    #static one; savings come from phi and from small x. Gauss-node theta
    #mode ('gauss') is kept for experiments only — it LACKS near-pole
    #samples and fails the twin gate (Phase 8: 0.48 rel L2 on M34/M44
    #already at x=0.02).
    ntrunc_cap = int(ntrunc_cap)
    # EVEN number of phi points: spacing 360/nphi then divides 180 deg, so
    # the Phase-4/6 mirror phi maps (phi -> -phi, 180-phi, 180+phi) map the
    # periodic ring exactly onto grid nodes (mirror_phi_index_map). Odd
    # counts put 180 deg between nodes -> fill-in raises.
    nphi = 2 * ntrunc_cap + 2
    if theta_mode == 'uniform':
        ntheta = max(ntrunc_cap + 2,
                     int(np.ceil(np.pi * float(x) / kappa)) + 1,
                     5)
        ntheta = min(ntheta, 181)   # production grid is the resolution ceiling
        return ntheta, nphi
    elif theta_mode == 'gauss':
        return ntrunc_cap + 3, nphi
    else:
        raise ValueError("theta_mode must be 'uniform' or 'gauss'")


def theta_values_gauss(ntrunc_cap):
    #theta [deg] ascending: 0, Gauss-Legendre nodes (cos-theta Gauss points),
    #180 — ascending; first entry 0 keeps ampl_data_temp[0] the forward row
    ngl = int(ntrunc_cap) + 1
    nodes, _ = np.polynomial.legendre.leggauss(ngl)
    th = np.degrees(np.arccos(np.sort(nodes)))
    return np.concatenate(([0.0], th, [180.0]))


def write_scat_grid_params(fname, ntrunc_cap, x=0.0, kappa=0.17,
                           theta_mode='uniform'):
    #write an ADDA '-scat_grid_inp' grid file and return the path; phi is
    #always the production-style uniform range grid 0..360 (both endpoints),
    #theta either uniform 0..180 (adaptive ntheta, see scat_grid_sizes) or
    #the experimental Gauss-node 'values' list
    ntheta, nphi = scat_grid_sizes(ntrunc_cap, x=x, kappa=kappa,
                                   theta_mode=theta_mode)
    head = ("# Scattering-angle grid auto-generated by "
            "acf.write_scat_grid_params\n"
            "# adaptive (Phase 8): ntrunc_cap={N} mode={mode} "
            "-> {nt} theta x {np} phi\n"
            "# comments are not allowed inside the lists (pairs,values)\n\n"
            "# grid or pairs\n"
            "global_type=grid\n\n"
            "theta:\n"
            ).format(N=int(ntrunc_cap), mode=theta_mode, nt=ntheta, np=nphi)
    if theta_mode == 'uniform':
        head += ("#range or values\n"
                 "type=range\n"
                 "#number of points (number of intervals + 1)\n"
                 "N={nt}\n"
                 "min=0\n"
                 "max=180\n\n"
                 ).format(nt=ntheta)
    else:
        th = theta_values_gauss(ntrunc_cap)
        head += ("#range or values\n"
                 "type=values\n"
                 "N={nt}\n"
                 "values=\n"
                 ).format(nt=ntheta)
        head += ''.join('%.10f\n' % v for v in th) + '\n'
    # phi: proper PERIODIC ring — endpoint excluded (max = 360 - 360/nphi),
    # unlike the historic static scat_params.dat whose 0..360 181x361 grid
    # duplicates phi=0/360 and thereby injects an aliasing error into the
    # periodic-FFT (shtns) analysis that grows as dphi^2 on coarser grids
    # (Phase 8 twin at x=5: 3.2e-2 rel L2 on M11 for nphi=169 vs 361).
    # phi=0 and phi=180 remain exact grid rows (j=0 and j=(nphi-1)/2).
    phi_max = 360.0 * (nphi - 1) / nphi
    tail = ("phi:\n"
            "#range or values\n"
            "type=range\n"
            "#number of points (number of intervals + 1)\n"
            "N={np}\n"
            "min=0\n"
            "max={pmax:.10f}\n"
            ).format(np=nphi, pmax=phi_max)
    with open(fname, 'w') as fh:
        fh.write(head + tail)
    return fname


def replace_scat_grid_inp(static_clo, gridfile):
    #swap the argument of '-scat_grid_inp' inside a static_clo list (or
    #append the pair if absent); returns a NEW list, input untouched
    clo=list(static_clo)
    if '-scat_grid_inp' in clo:
        i=clo.index('-scat_grid_inp')
        if i+1>=len(clo):
            raise RuntimeError("static_clo: '-scat_grid_inp' without argument")
        clo[i+1]=str(gridfile)
    else:
        clo=clo+['-scat_grid_inp', str(gridfile)]
    return clo


#==============================================================================
# Adda call
#==============================================================================

def adda_call(input_data):

#    pdb.set_trace()


    #wavelength
    lamb=input_data['lambda']

    #Particle Size
    D=input_data['Size']

    #refractive Index
    m=input_data['refrative_index']

    #incidence angle
    theta=input_data['theta']

    #alpha
    alpha=input_data['alpha']

    #beta angle
    beta=input_data['beta']

    #gamma angle
    gamma=input_data['gamma']

    #precision
    eps=input_data['eps']

    #ramdrive
    ramdrive=input_data['ramdrive']

    #simulation basename
#    simbasename=input_data['simbasename']

    #Name of the zipname
    zipname=input_data['zipname']

    #main command for adda
    main_command=input_data['main_command']

    #Adda's static command line options
    static_clo=input_data['static_clo']
    
    zipname_vid=('vertex_X{XV:07.0f}_Y{YV:07.0f}_Z{ZV:07.0f}').format

    with zipfile.ZipFile(zipname, mode='a') as zf:


        #Make arguments for adda
        #-----------------------


        #refractive index
        m_str=['-m',str(m.real),str(m.imag)]

        #Wavelength
        wavelen_str=['-lambda',str(lamb)]

        #Size
        size_str=['-eq_rad',str(D)]

        #precision
        eps_str=['-eps',str(eps)]


        #Particle orientation
        pori_str=['-orient',str(alpha),str(beta),str(gamma)]

        #Incidence direction
        if np.size(theta)==1:
            px=[np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)]
            prop_str=['-prop',str(px[0]),str(px[1]),str(px[2])]
        else:
            prop_str=['-prop',str(theta[0]),str(theta[1]),str(theta[2])]


        #Output directory
        path_name=""
        try:
            path_name = tempfile.mkdtemp(dir=ramdrive)
            dir_str=['-dir',path_name+'/temp']

            #command_str
            commando=main_command+static_clo+\
                      m_str+wavelen_str+size_str+pori_str+\
                      prop_str+dir_str+eps_str


            print('  ')
            print(commando)

    #        pdb.set_trace()

            #run adda==========================================================

            #sub.call(commando)
            ret=spawn_command(' '.join(commando))

            #ADDAs exit status was never checked before (Phase 3) — a crashed
            #or signalled ADDA used to pass silently into the parse step
            if int(ret)!=0:
                raise RuntimeError('ADDA terminated abnormally '
                                   '(wait status {}, signal {}): {}'
                                   .format(int(ret), int(ret)&0x7f,
                                           ' '.join(commando[:4])+' ...'))

            #==================================================================

    #        pdb.set_trace()

            #Load amplitude matrix
            ampl_data_temp=np.loadtxt(path_name+'/temp/ampl_scatgrid', skiprows=1)

            #only forward direction is needed for extinction matrix
            ampl_raw_array=ampl_data_temp[0,2:]

            #The four coefficients of the scattering amplitude matrix
            s11=ampl_raw_array[2]+1j*ampl_raw_array[3]
            s12=ampl_raw_array[4]+1j*ampl_raw_array[5]
            s21=ampl_raw_array[6]+1j*ampl_raw_array[7]
            s22=ampl_raw_array[0]+1j*ampl_raw_array[1]


            #Calculate extinctrion matrix
            ext_mat=extinction_matrix(s11,s12,s21,s22,lamb)



            #Load mueller_scatgrid
            data_temp=np.loadtxt(path_name+'/temp/mueller_scatgrid', skiprows=1)

            phis_grid=data_temp[:,1]
            thetas_grid=data_temp[:,0]

    #        test_phi=data_temp[:,1]
    #        test_theta=data_temp[:,0]

            #make it unique
            phis_grid=np.unique(phis_grid)
            thetas_grid=np.unique(thetas_grid)



            #Mueller matrix
            mueller_raw_array=data_temp[:,2:]
            qmm=mueller_raw_array.shape
            mueller_raw_array=np.reshape(mueller_raw_array,(thetas_grid.size,
                                                phis_grid.size,qmm[1]))

            mueller_array=mueller_raw_array


            # Phase 6: exact backscatter Mueller matrix (4x4).
            # mueller_scatgrid lives on the propagation-ATTACHED (theta,phi)
            # grid; Theta=180 is the exact-backscatter ring (scattering
            # direction = -incidence), Phi=0 is one distinguished row of it.
            # Raw values are k-normalized BH-amplitude products (dimension-
            # less per sr) — NOT physical m^2/sr; the conversion lambda^2/
            # (4*pi)^2 is applied only when exporting (see report Phase 6).
            # The grid must contain both Theta=180 and Phi=0 (all production
            # scat_params grids do; the pipeline relies on this).
            it=np.flatnonzero(np.abs(thetas_grid-180.)<1e-9)
            ip=np.flatnonzero(np.abs(phis_grid)<1e-9)
            if it.size!=1 or ip.size!=1:
                raise RuntimeError('adda_call: scattering grid lacks '
                                   'Theta=180 / Phi=0 row '
                                   '(thetas {}...{}, phis {}...{}), cannot '
                                   'extract backscatter matrix'
                                   .format(thetas_grid[0],thetas_grid[-1],
                                           phis_grid[0],phis_grid[-1]))

            # full Theta=180 ring: (nphi,4,4), needed for mirror fill-in
            back_mat_ring=mueller_array[it[0],:,:].reshape(phis_grid.size,4,4)

            # the backscatter matrix itself: ring row at Phi=0
            back_matrix=mueller_array[it[0],ip[0],:].reshape(4,4)



            #look after the adda output
            if  np.size(theta)==1:
                filelist = glob.glob(path_name+'/temp/*')
            else:
                
#                np.save(path_name+'/temp/mueller_array', mueller_array)
#                np.save(path_name+'/temp/ext_mat', ext_mat)
#                np.save(path_name+'/temp/phis_grid', phis_grid)
#                np.save(path_name+'/temp/thetas_grid', thetas_grid)
#                                
                filelist = glob.glob(path_name+'/temp/*')


            #and now put it into a zipfile
            #uncomment the lines if all calculations should be saved

            if not 'zipping' in input_data.keys():
                zipping=False
            else:
                zipping=input_data['zipping']


            if zipping==True:
                for filename in filelist:

                    if  np.size(theta)==1:
                        folder_structure=os.path.join('theta_'+str(theta),
                        'beta_'+str(beta),'alpha_'+str(alpha),
                        'gamma_'+str(gamma),os.path.basename(filename))
                    else:
                        
                        zipstrings={'XV':(theta[0]*1e5), 'YV':(theta[1]*1e5),'ZV':(theta[2]*1e5)}
                        folder_structure=os.path.join(zipname_vid(**zipstrings),
                                                      os.path.basename(filename))
                        

                    zf.write(filename, folder_structure, compress_type=compression)
                    
            else:
                os.remove(zipname)
                

        #clean up
        finally:
            if path_name != "":
                shutil.rmtree(path_name)

#        pdb.set_trace()

        #Put output in dictionary
        output_data={}
        output_data['mueller_array']=mueller_array
        output_data['phis_grid']=phis_grid
        output_data['thetas_grid']=thetas_grid
        output_data['ext_matrix']=ext_mat
        output_data['back_matrix']=back_matrix
        output_data['back_mat_ring']=back_mat_ring





    return output_data


#==============================================================================
#  Romberg integration over gamma angle (Euler angle)
#==============================================================================
def romberg_gamma(input_data):

    #Break up limit/criteria
    limit=input_data['limit_gamma']
    delta=input_data['delta']

    #gamma intervall
    gamma_ab=input_data['gamma_ab']

#    pdb.set_trace()

    #Make the grid refinement only if there is an intervall
    if len(gamma_ab)>1:

        #First step

        #counter
        Nidx=1


        x_add=gamma_ab

        #Get the results for first two values
        for i in range(0,x_add.size):



            input_data['gamma']=x_add[i]

            #calling Adda
            output_data=adda_call(input_data)

#            pdb.set_trace()

            mueller_matrix=output_data['mueller_array']
            ext_mat=output_data['ext_matrix']

            #size of Mueller Matrix
            q=mueller_matrix.shape

            #Add forth dimension
            mueller_matrix=np.reshape(mueller_matrix,(q[0],q[1],q[2],1))

            #size of extinction matrix
            q_ext=ext_mat.shape

            #Add third dimension
            ext_mat=np.reshape(ext_mat,(q_ext[0],q_ext[1],1))



            if i==0:
                y_add=mueller_matrix
                y_ext_add=ext_mat
            else:
                y_add=np.concatenate((y_add,mueller_matrix),3)
                y_ext_add=np.concatenate((y_ext_add,ext_mat),2)




        #put the new Values in our storage
        x=x_add
        y=y_add
        y_ext=y_ext_add

        #sort the values
        ins=np.argsort(x)
        x_sort=x[ins]
        y_sort=y[:,:,:,ins]
        y_ext_sort=y_ext[:,:,ins]



        #integrate over gamma orientation
        trapz_int=np.sum(np.diff(x_sort)*(y_sort[:,:,:,0:-1]+y_sort[:,:,:,1:])/2,3)
        trapz_int=np.reshape(trapz_int,(q[0],q[1],q[2]))

        trapz_ext_int=np.sum(np.diff(x_sort)*(y_ext_sort[:,:,0:-1]+y_ext_sort[:,:,1:])/2,2)
        trapz_ext_int=np.reshape(trapz_ext_int,(q_ext[0],q_ext[1]))


        #First result
        R=np.zeros((q[0],q[1],q[2],limit,limit))
        R[:,:,:,0,0]=trapz_int;

        R_ext=np.zeros((q_ext[0],q_ext[1],limit,limit))
        R_ext[:,:,0,0]=trapz_ext_int;

        #start value for break up checker
        krit=1.

        #allocate matrices
        krit_ctrl=np.ones((limit))
        krit_ext_ctrl=np.ones((limit))

    #    pdb.set_trace()

        #let us do some refinement
        min_its=int(np.abs(np.log10(delta)-1))
        if min_its>limit:
            raise Exception('maximum Number of Iteration is too low for the give precision');


        while (not(krit<delta and Nidx> min_its) and Nidx<=limit):


            #Counter
            Nidx=Nidx+1;

    #        pdb.set_trace()

            #The new sample points
            if( np.mod(Nidx ,2) ==0):
                n = 2**(Nidx/2)
            else:
                n = 3*2**(((Nidx-1)/2)-1)

            x_new=np.linspace(gamma_ab[0],gamma_ab[1],n+1)
            x_add=np.array([])
            y_new=np.array([])

            #Look for already existing values
            for i in range(0,x_new.size):

    #            pdb.set_trace()

                horst=np.where(x_new[i]==x)

                fidx=horst[0]

                #If a values does not exist...
                if fidx.size<1:

                    input_data['gamma']=x_new[i]

                    #calling Adda
                    output_data=adda_call(input_data)

                    mueller_matrix=output_data['mueller_array']
                    ext_mat=output_data['ext_matrix']

                    #size of Mueller Matrix
                    q=mueller_matrix.shape

                    #Add forth dimension
                    mueller_matrix=np.reshape(mueller_matrix,(q[0],q[1],q[2],1))

                    #size of extinction matrix
                    q_ext=ext_mat.shape

                    #Add third dimension
                    ext_mat=np.reshape(ext_mat,(q_ext[0],q_ext[1],1))


    #                pdb.set_trace()

                    #Add additional values
                    if x_add.size==0:
                        x_add=np.array([x_new[i]]);
                        y_add=mueller_matrix
                        y_ext_add=ext_mat
                    else:
                        x_add=np.concatenate((x_add,[x_new[i]]),0)
                        y_add=np.concatenate((y_add,mueller_matrix),3)
                        y_ext_add=np.concatenate((y_ext_add,ext_mat),2)

                    if y_new.size==0:
                        y_new=mueller_matrix
                        y_ext_new=ext_mat

                    else:
                        y_new=np.concatenate((y_new,mueller_matrix),3)
                        y_ext_new=np.concatenate((y_ext_new,ext_mat),2)

                else:
                    if y_new.size==0:
                        y_new=y[:,:,:,fidx]
                        y_ext_new=y_ext[:,:,fidx]

                    else:
                        y_new=np.concatenate((y_new,y[:,:,:,fidx]),3)
                        y_ext_new=np.concatenate((y_ext_new,y_ext[:,:,fidx]),2)


    #        pdb.set_trace()


            #Put the new values into storage
            x=np.concatenate((x,x_add),0)
            y=np.concatenate((y,y_add),3)
            y_ext=np.concatenate((y_ext,y_ext_add),2)


            #sort the values
            ins=np.argsort(x_new)
            x_sort=x_new[ins]
            y_sort=y_new[:,:,:,ins]
            y_ext_sort=y_ext_new[:,:,ins]

            #integrate over gamma orientation...
            #Mueller matrix
            trapz_int=np.sum(np.diff(x_sort)*(y_sort[:,:,:,0:-1]+y_sort[:,:,:,1:])/2,3)
            trapz_int=np.reshape(trapz_int,(q[0],q[1],q[2]))

            #Extinction matrix
            trapz_ext_int=np.sum(np.diff(x_sort)*(y_ext_sort[:,:,0:-1]+y_ext_sort[:,:,1:])/2,2)
            trapz_ext_int=np.reshape(trapz_ext_int,(q_ext[0],q_ext[1]))


            #result
            R[:,:,:,Nidx-1,0]=trapz_int
            R_ext[:,:,Nidx-1,0]=trapz_ext_int;

            #Extrapolation for better results ;-)
            for k in range(2,Nidx+1):

    #            pdb.set_trace()

                R[:,:,:,Nidx-1,k-1]=(4**(k-1)*R[:,:,:,Nidx-1,k-2]-R[:,:,:,Nidx-2,k-2]) / (4**(k-1)-1);
                R_ext[:,:,Nidx-1,k-1]=(4**(k-1)*R_ext[:,:,Nidx-1,k-2]-R_ext[:,:,Nidx-2,k-2]) / (4**(k-1)-1);



            #compare old and new value
            # Element 1,1 of the Mueller matrix is taken as control value.
            krit=abs((R[0,0,0,Nidx-1,Nidx-1]-R[0,0,0,Nidx-2,Nidx-2])) / abs(R[0,0,0,Nidx-1,Nidx-1])
            krit_ctrl[Nidx-1]=krit
            krit_ext=abs((R_ext[0,0,Nidx-1,Nidx-1]-R_ext[0,0,Nidx-2,Nidx-2])) / abs(R_ext[0,0,Nidx-1,Nidx-1])
            krit_ext_ctrl[Nidx-1]=krit_ext



            #Mueller matrix averaged over gamma
            mueller_matrix_avg=np.reshape(R[:,:,:,Nidx-1,Nidx-1],(q[0],q[1],q[2]))/(gamma_ab[1]-gamma_ab[0])

            #Extinction matrix averaged over gamma
            ext_matrix_avg=np.reshape(R_ext[:,:,Nidx-1,Nidx-1],(q_ext[0],q_ext[1]))/(gamma_ab[1]-gamma_ab[0])


    #    pdb.set_trace()
#            output_data['Rg']=R
#            output_data['Rg_ext']=R_ext
#            output_data['yg']=y
#            output_data['xg']=x


    else:
        input_data['gamma']=gamma_ab[0]
        x_sort=gamma_ab[0]


        #calling Adda
        output_data=adda_call(input_data)


        mueller_matrix_avg=output_data['mueller_array']
        ext_matrix_avg=output_data['ext_matrix']
        Nidx=1
        krit_ctrl=0.
        krit_ext_ctrl=0.




    #Put output in dictionary
    output_data['mueller_matrix_avg']=mueller_matrix_avg
    output_data['ext_matrix_avg']=ext_matrix_avg
    output_data['gamma_grid']=x_sort
    output_data['Nidx_gamma']=Nidx
    output_data['limit_gamma']=limit
    output_data['krit_gamma']=krit_ctrl
    output_data['krit_ext_gamma']=krit_ext_ctrl
    output_data['gamma_ab']=gamma_ab




    return output_data



#==============================================================================
#  Romberg integration over alpha angle (Euler angle)
#==============================================================================
def romberg_alpha(input_data):

    #Break up limit/criteria
    limit=input_data['limit_alpha']
    delta=input_data['delta']


    #alpha intervall
    alpha_ab=input_data['alpha_ab']

#    pdb.set_trace()

    #Allocate matrix for number of evaluation per step
    nx=np.zeros(limit)


    #Make the grid refinement only if there is an intervall
    if len(alpha_ab)>1:
        #First step

        Nidx_gamma_list=[]

        #Counter
        Nidx=1
        x_add=alpha_ab

        #Get the results for first two values
        for i in range(0,x_add.size):


            #Alpha angle
            input_data['alpha']=x_add[i]

            #romberg_gamma
            output_data=romberg_gamma(input_data)
            Nidx_gamma_list=Nidx_gamma_list+[len(output_data['gamma_grid'])]

            mueller_matrix=output_data['mueller_matrix_avg']
            ext_mat=output_data['ext_matrix_avg']

            #size of Mueller Matrix
            q=mueller_matrix.shape

            #Add forth dimension
            mueller_matrix=np.reshape(mueller_matrix,(q[0],q[1],q[2],1))

            #size of extinction matrix
            q_ext=ext_mat.shape

            #Add third dimension
            ext_mat=np.reshape(ext_mat,(q_ext[0],q_ext[1],1))

            if i==0:
                y_add=mueller_matrix
                y_ext_add=ext_mat


            else:
                y_add=np.concatenate((y_add,mueller_matrix),3)
                y_ext_add=np.concatenate((y_ext_add,ext_mat),2)



        #put the new Values in our storage
        x=x_add
        y=y_add
        y_ext=y_ext_add

        #sort the values
        ins=np.argsort(x)
        x_sort=x[ins]
        y_sort=y[:,:,:,ins]
        y_ext_sort=y_ext[:,:,ins]

        nx[0]=x_add.size;


        #integrate over alpha orientation
        trapz_int=np.sum(np.diff(x_sort)*(y_sort[:,:,:,0:-1]+y_sort[:,:,:,1:])/2,3)
        trapz_int=np.reshape(trapz_int,(q[0],q[1],q[2]))

        trapz_ext_int=np.sum(np.diff(x_sort)*(y_ext_sort[:,:,0:-1]+y_ext_sort[:,:,1:])/2,2)
        trapz_ext_int=np.reshape(trapz_ext_int,(q_ext[0],q_ext[1]))


        #First result
        R=np.zeros((q[0],q[1],q[2],limit,limit))
        R[:,:,:,0,0]=trapz_int;

        R_ext=np.zeros((q_ext[0],q_ext[1],limit,limit))
        R_ext[:,:,0,0]=trapz_ext_int;

        #Start value for break up checker
        krit=1.

        #Allocate matrices
        krit_ctrl=np.ones((limit))*-1
        krit_ext_ctrl=np.ones((limit))*-1


        min_its=int(np.abs(np.log10(delta)-1))
        if min_its>limit:
            raise Exception('maximum Number of Iteration is too low for the give precision');


        while (not(krit<delta and Nidx> min_its) and Nidx<=limit):


            #Counter
            Nidx=Nidx+1;


            #The new sample points
            if( np.mod(Nidx ,2) ==0):
                n = 2**(Nidx/2);
            else:
                n = 3*2**(((Nidx-1)/2)-1) ;

            x_new=np.linspace(alpha_ab[0],alpha_ab[1],n+1);
            x_add=np.array([])
            y_new=np.array([])

            #Look for already existing values
            for i in range(0,x_new.size):
                horst=np.where(x_new[i]==x);

                fidx=horst[0];

                #If a values does not exist...
                if fidx.size<1:

                    input_data['alpha']=x_new[i]

                    #romberg_alpha
                    output_data=romberg_gamma(input_data)
                    Nidx_gamma_list=Nidx_gamma_list+[len(output_data['gamma_grid'])]

                    mueller_matrix=output_data['mueller_array']
                    ext_mat=output_data['ext_matrix']

                    #size of Mueller Matrix
                    q=mueller_matrix.shape

                    #Add forth dimension
                    mueller_matrix=np.reshape(mueller_matrix,(q[0],q[1],q[2],1))

                    #size of extinction matrix
                    q_ext=ext_mat.shape

                    #Add third dimension
                    ext_mat=np.reshape(ext_mat,(q_ext[0],q_ext[1],1))


    #                pdb.set_trace()

                    #Add additional values
                    if x_add.size==0:
                        x_add=np.array([x_new[i]]);
                        y_add=mueller_matrix
                        y_ext_add=ext_mat

                    else:
                        x_add=np.concatenate((x_add,[x_new[i]]),0)
                        y_add=np.concatenate((y_add,mueller_matrix),3)
                        y_ext_add=np.concatenate((y_ext_add,ext_mat),2)

                    if y_new.size==0:
                        y_new=mueller_matrix
                        y_ext_new=ext_mat

                    else:
                        y_new=np.concatenate((y_new,mueller_matrix),3)
                        y_ext_new=np.concatenate((y_ext_new,ext_mat),2)

                else:
                    if y_new.size==0:
                        y_new=y[:,:,:,fidx]
                        y_ext_new=y_ext[:,:,fidx]

                    else:
                        y_new=np.concatenate((y_new,y[:,:,:,fidx]),3)
                        y_ext_new=np.concatenate((y_ext_new,y_ext[:,:,fidx]),2)



            #Put the new values into storage
            x=np.concatenate((x,x_add),0)
            y=np.concatenate((y,y_add),3)
            y_ext=np.concatenate((y_ext,y_ext_add),2)

            #Number of evaluation at specific step
            nx[Nidx-1]=x.size;

            #sort the values
            ins=np.argsort(x_new)
            x_sort=x_new[ins]
            y_sort=y_new[:,:,:,ins]
            y_ext_sort=y_ext_new[:,:,ins]

            #integrate over alpha orientation...
            #Mueller matrix
            trapz_int=np.sum(np.diff(x_sort)*(y_sort[:,:,:,0:-1]+y_sort[:,:,:,1:])/2,3)
            trapz_int=np.reshape(trapz_int,(q[0],q[1],q[2]))

            #Extinction matrix
            trapz_ext_int=np.sum(np.diff(x_sort)*(y_ext_sort[:,:,0:-1]+y_ext_sort[:,:,1:])/2,2)
            trapz_ext_int=np.reshape(trapz_ext_int,(q_ext[0],q_ext[1]))


            #result
            R[:,:,:,Nidx-1,0]=trapz_int
            R_ext[:,:,Nidx-1,0]=trapz_ext_int;

            #Extrapolation for better results ;-)
            for k in range(2,Nidx+1):

    #            pdb.set_trace()

                R[:,:,:,Nidx-1,k-1]=(4**(k-1)*R[:,:,:,Nidx-1,k-2]-R[:,:,:,Nidx-2,k-2]) / (4**(k-1)-1);
                R_ext[:,:,Nidx-1,k-1]=(4**(k-1)*R_ext[:,:,Nidx-1,k-2]-R_ext[:,:,Nidx-2,k-2]) / (4**(k-1)-1);


            #compare old and new value
#            krit=np.amax(abs((R[:,:,:,Nidx-1,Nidx-1]-R[:,:,:,Nidx-2,Nidx-2])) / abs(R[:,:,:,Nidx-1,Nidx-1]))
            krit=abs((R[0,0,0,Nidx-1,Nidx-1]-R[0,0,0,Nidx-2,Nidx-2])) / abs(R[0,0,0,Nidx-1,Nidx-1])
            krit_ctrl[Nidx-1]=krit
#            krit_ext=np.amax(abs((R_ext[:,:,Nidx-1,Nidx-1]-R_ext[:,:,Nidx-2,Nidx-2]))) / abs(R_ext[:,:,Nidx-1,Nidx-1]))
            krit_ext=abs((R_ext[0,0,Nidx-1,Nidx-1]-R_ext[0,0,Nidx-2,Nidx-2])) / abs(R_ext[0,0,Nidx-1,Nidx-1])
            krit_ext_ctrl[Nidx-1]=krit_ext

            #Mueller matrix averaged over alpha
            mueller_matrix_avg=np.reshape(R[:,:,:,Nidx-1,Nidx-1],(q[0],q[1],q[2]))/(alpha_ab[1]-alpha_ab[0])

            #Extinction matrix averaged over alpha
            ext_matrix_avg=np.reshape(R_ext[:,:,Nidx-1,Nidx-1],(q_ext[0],q_ext[1]))/(alpha_ab[1]-alpha_ab[0])


#            output_data['Ra']=R
#            output_data['R_ext']=R_ext
#            output_data['ya']=y
#            output_data['xa']=x



    else:
        input_data['alpha']=alpha_ab[0]
        x=alpha_ab[0]

        #romberg_alpha
        output_data=romberg_gamma(input_data)

        mueller_matrix_avg=output_data['mueller_matrix_avg']
        ext_matrix_avg=output_data['ext_matrix_avg']

#    pdb.set_trace()

    #Put output in dictionary
    output_data['mueller_matrix_avg_alpha']=mueller_matrix_avg
    output_data['ext_matrix_avg_alpha']=ext_matrix_avg
    output_data['alpha_grid']=x
    output_data['Nidx_alpha']=Nidx
    output_data['Nidx_gamma']=Nidx_gamma_list
    output_data['limit_alpha']=limit
    output_data['krit_alpha']=krit_ctrl
    output_data['krit_ext_alpha']=krit_ext_ctrl
    output_data['alpha_ab']=alpha_ab
    output_data['Nx']=nx

    #remove old stuff
    del output_data['mueller_matrix_avg']
    del output_data['ext_matrix_avg']
    del output_data['mueller_array']


    return output_data

#==============================================================================
# Adda call for random orientation
#==============================================================================

def adda_call_randomorient(input_data):
    #adda wrapper for running random oriented particles simulations


#    pdb.set_trace()




    #wavelength
    lamb=input_data['lambda']

    #Particle Size
    D=input_data['radius']

    #refractive Index
    m=input_data['refractive_index']

    #precision
    eps=input_data['eps']

    #ramdrive
    main_path=input_data['main_path']

    #main command for adda
    main_command=input_data['main_command']

    #Adda's static command line options
    static_clo=input_data['static_clo']

    calc_type=input_data['calc_type']

    batch_time=input_data['batch_time']


    #Make arguments for adda
    #-----------------------


    #refractive index
    m_str=['-m',str(m.real),str(m.imag)]

    #Wavelength
    wavelen_str=['-lambda',str(lamb)]

    #Size
    size_str=['-eq_rad',str(D)]

    #precision
    eps_str=['-eps',str(eps)]


    #Output directory
    try:

        dir_str=['-dir',main_path]



#        pdb.set_trace()


        #select the right command to run adda==================================
        if calc_type=='mistral':

            print('mistral')

            #adda batch call
            batch_adda='/mnt/lustre01/work/um0878/users/mbrath/adda/adda_python_scripts/sbatch_adda'

            #batch string (queue on mistral)
#            sbatch_str=['echo',batch_adda,'-d',
#                        'ADDA-lamb'+str(np.round(lamb*10.)/10.)+'-dmax'+str(np.round(D*10.)/10.),
#                        str(batch_time),str(main_command[2])]
            sbatch_str=[batch_adda,
                        'ADDA-lamb'+str(np.round(lamb*10.)/10.)+'-dmax'+str(np.round(D*10.)/10.),
                        str(int(batch_time)),str(main_command[2])]

            #adda string
            adda_str=static_clo+\
                      m_str+wavelen_str+size_str+dir_str+eps_str

            #Command string (queue on mistral)
#            commando=sbatch_str+adda_str
            # Phase 3: this branch never defined `commando` (silent NameError
            # swallowed by the bare except at the end of this function). Fail
            # loudly instead; the sbatch_adda path was never completed.
            raise NotImplementedError(
                'adda_call_randomorient: mistral batch submission was never '
                'implemented; use calc_type=noqueue/local or run on thunder')
            
#        elif calc_type=='thunder':
#
#            print('queue')
#
#            #batch string for thunder7
#            sbatch_str=['/scratch/uni/u237/sw/bin/sbatch_simple','-x', 'ctc[132,137,138,139]',
#                        'ADDA-lamb'+str(np.round(lamb*10.)/10.)+'-dmax'+str(np.round(D*10.)/10.),
#                        str(main_command[2])]
#
#            adda_str=main_command+static_clo+\
#                      m_str+wavelen_str+size_str+dir_str
#
#            #Command string (queue)
#            commando=sbatch_str+adda_str
            

        else:
            print('noqueue')

            #adda string
            adda_str=main_command+static_clo+\
                      m_str+wavelen_str+size_str+dir_str+eps_str

            #Command string (non queue)
            commando=adda_str;


#        if calc_type=='noqueue':
#
#            print('noqueue')
#
#            #adda string
#            adda_str=main_command+static_clo+\
#                      m_str+wavelen_str+size_str+dir_str
#
#            #Command string (non queue)
#            commando=adda_str;
#
#        elif calc_type=='queue':
#
#            print('queue')
#
#            #batch string for thunder7
#            sbatch_str=['/scratch/uni/u237/sw/bin/sbatch_simple','-x', 'ctc[132,137,138,139]',
#                        'ADDA-lamb'+str(np.round(lamb*10.)/10.)+'-dmax'+str(np.round(D*10.)/10.),
#                        str(main_command[2])]
#
#            adda_str=main_command+static_clo+\
#                      m_str+wavelen_str+size_str+dir_str
#
#            #Command string (queue)
#            commando=sbatch_str+adda_str
#
#        elif calc_type=='mistral':
#
#            print('mistral')
#
#            #adda batch call
#            batch_adda='/mnt/lustre01/work/um0878/users/mbrath/adda/adda_python_scripts/sbatch_adda'
#
#            #batch string (queue on mistral)
##            sbatch_str=['echo',batch_adda,'-d',
##                        'ADDA-lamb'+str(np.round(lamb*10.)/10.)+'-dmax'+str(np.round(D*10.)/10.),
##                        str(batch_time),str(main_command[2])]
#            sbatch_str=[batch_adda,
#                        'ADDA-lamb'+str(np.round(lamb*10.)/10.)+'-dmax'+str(np.round(D*10.)/10.),
#                        str(int(batch_time)),str(main_command[2])]
#
#            #adda string
#            adda_str=static_clo+\
#                      m_str+wavelen_str+size_str+dir_str
#
#            #Command string (queue on mistral)
#            commando=sbatch_str+adda_str



        #run adda==========================================================

        #sub.call(commando)
        ret=spawn_command(' '.join(commando))

        #check ADDAs exit status (Phase 3 — used to be ignored entirely)
        if int(ret)!=0:
            raise RuntimeError('ADDA terminated abnormally '
                               '(wait status {}, signal {})'
                               .format(int(ret), int(ret)&0x7f))

        #==================================================================

#        pdb.set_trace()
    except Exception as exc:
        print('there was something wrong...')
        print('input data:')
        print(input_data)
        print(' ')
        print('I hope, it was useful.')
        # Phase 3: re-raise — swallowing this made mass TRO runs zip
        # empty output dirs without any signal (report §1.6-8)
        raise


#==============================================================================
#  additional functions
#==============================================================================

def eps_ice_matzler06(f,TK):
    #Permittivity of fresh water ice according to Maetzler 2006
    #stolen from atmlab
    # input:
    # f     Frequency [Hz]
    # TK    Temperature [K]
    #
    #output:
    # z     permittivity of fresh water ice [no unit]


    #Frequency
    f = f/1e9;

    #some checks
    if (f<0.01 or f>3000):
      raise Exception('Valid range for frequency is 0.01-3000 GHz');

    if (TK<20 or TK>273.15):
      raise Exception('Valid range for temperature is 20-273.15 K');


    B1 = 0.0207
    B2 = 1.16e-11
    b = 335
    deltabeta = np.exp(-9.963 + 0.0372*(TK-273.));
    betam = (B1/TK)* ( np.exp(b/TK)/ ((np.exp(b/TK)-1)**2.) ) + B2*f**2.;
    beta = betam + deltabeta;
    theta = 300./TK - 1.;
    alfa = (0.00504 + 0.0062*theta)*np.exp(-22.1*theta);
    z = 3.1884 + 9.1e-4*(TK-273);

    #permittivity
    z = z + 1j*(alfa/f + beta*f);

    return z



def chunking(time,ncores,duration):
    #function to combine the different adda jobs into
    #chunks of a chosen time length
    #
    # input:
    #   time        a list of the duration of each job in seconds
    #   ncores      the number of core whih will be used
    #   duration    maximum length of a chunk in hours

    #sort accorting to time
    sort_idx=np.argsort(time,axis=-1)
    time_sorted=[time[i] for i in sort_idx]


    #convert to matrix and convert it to node hours
    time_sorted=np.matrix(time_sorted)/ncores/3600 #[h]
    time_sorted = np.squeeze(np.asarray(time_sorted))

    time=np.matrix(time)/ncores/3600 #[h]
    time = np.squeeze(np.asarray(time))

    #cumulative sum
    time_summed=np.cumsum(time_sorted)
    time_summed = np.squeeze(np.asarray(time_summed))
    
    if np.size(time)==1:
        time=np.array([time])
        time_summed=np.array([time_summed])


    #some start values
    n=0;
    index_sum=np.inf;
    counter=0;

    chunks=[];
    marker=np.zeros(np.size(time_summed))
    test=[]

    start_time=0

    # now cut the list into pieces of 8 hours
    while index_sum>0:

        if n==41:
            print('grrr')

        n=n+1;
        end_time=start_time+duration

        #take the data between start and end time
        logic=np.logical_and(start_time<time_summed,time_summed<=end_time);

        if np.sum(logic)<1:
            break
       
        #indices of chunk
        indices=sort_idx[logic];

        #check the clactime
        test=test+[np.sum(time[indices])]

        # mark the indices, which already have been taken
        marker[logic]=1

        #store the indices of a chunk
        chunks=chunks+[indices]

        #sum up the amount of indices
        index_sum=np.size(indices);

        #just a counter
        counter=counter+index_sum

        #next start time
        dummy=time_summed[logic]
        start_time=dummy.max()

    #now get all the longer runs
    rest_stuff=[sort_idx[i] for i in sort_idx if marker[i]==0]

    #append the rest stuff to chunks
    chunks=chunks+rest_stuff

    #check
    ctrl_sum=0
    calc_time=np.zeros(len(chunks))
    for chnk in range(0,len(chunks)):

        ctrl_sum=ctrl_sum+np.sum(np.size(chunks[chnk]))


        calc_time[chnk]=np.sum(time[chunks[chnk]])

    return chunks




## functions to load matlab files 

import scipy.io as spio

try:
    # scipy >= 1.12 public location (mio5_params namespace is deprecated,
    # removed in SciPy 2.0)
    from scipy.io.matlab import mat_struct as _mat_struct
except ImportError:  # older scipy
    from scipy.io.matlab.mio5_params import mat_struct as _mat_struct

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], _mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, _mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



#==============================================================================
#  ADDA mirror-symmetry transforms (Phase 4)
#  Empirically verified for ADDA 1.2 by analysis/phase4_grid_symmetry.py
#  (generic oblique propagation pairs (p, S p) on an oblate spheroid,
#  residuals <=3e-11 at -eps 3, i.e. solver noise). ADDA's mueller_scatgrid
#  lives on a propagation-ATTACHED (theta,phi) angle grid, so theta is NEVER
#  remapped; each particle mirror plane S contributes
#     mueller_S(theta,phi) = mueller(theta, phi_map(phi)) * sgn16
#     ext_matrix_S         = ext_matrix * sgn4             (elementwise)
#  with per-mirror phi maps listed below and the sign matrix outer(D,D),
#  D = diag(1,1,-1,-1) (attached-frame linear polarization basis relabelling;
#  composing two mirrors cancels the signs, D^2=1). This matches the sign
#  convention already used by aoa.construct_by_symmetry's extinction branch.
#==============================================================================
MIRROR_AXES = {'XY': 2, 'XZ': 1, 'YZ': 0}
MIRROR_PHIMAP = {'XY': lambda p: 180.0 - p, 'XZ': lambda p: -p, 'YZ': lambda p: -p}
MIRROR_D = np.array([1.0, 1.0, -1.0, -1.0])

def mirror_signs(nmirrors):
    """16-component (row-major) sign pattern of nmirrors composed mirrors"""
    if nmirrors % 2:
        return np.outer(MIRROR_D, MIRROR_D).reshape(16)
    return np.ones(16)

def mirror_phi_index_map(phis, subset, tol=1e-6):
    """column index map j -> source column implementing the composed phi maps
    of `subset` (iterable over 'XY'/'XZ'/'YZ') on the phi grid `phis` (deg);
    raises if the grid cannot represent a mapped value exactly"""
    phis = np.asarray(phis, dtype=float)
    idx = np.arange(phis.size)
    for f in subset:
        m = MIRROR_PHIMAP[f](phis) % 360.0
        j = np.abs(phis[None, :] - m[:, None]).argmin(axis=1)
        if np.max(np.abs(phis[j] - m)) > tol:
            raise ValueError('phi grid cannot represent mirror %s (max dev %.3g deg)'
                             % (f, np.max(np.abs(phis[j] - m))))
        idx = idx[j]
    return idx


#==============================================================================
#  Spherical-harmonic (l,m) index arrays
#  The NetCDF stores keep the shtns coefficient arrays as-is (m-blocks m=0
#  ..ntrunc, each block l=m..ntrunc, orthonormal + SHT_NO_CS_PHASE, m<0 via
#  complex conjugate). The pairing coeff[j] <-> (l[j],m[j]) was only implied
#  by that layout; sph_lm_indices() materializes it so it can be stored
#  next to the coefficients. Verified against shtns.idx/l/m at import time
#  when shtns is available (sph_lm_indices_verify).
#==============================================================================

def sph_lm_indices(n_coeffs):
    """(l, m) index arrays (int32) for a shtns coefficient array of length
    n_coeffs = (ntrunc+1)(ntrunc+2)/2, in storage order."""
    r = int(round(np.sqrt(8.0 * n_coeffs + 1.0)))
    if r * r != 8 * n_coeffs + 1 or r % 2 == 0:
        raise ValueError(
            'coefficient array length %d is not a shtns m>=0 triangular '
            'count (ntrunc+1)(ntrunc+2)/2' % n_coeffs)
    ntrunc = (r - 3) // 2
    ls = np.concatenate([np.arange(m, ntrunc + 1) for m in range(ntrunc + 1)])
    ms = np.concatenate([np.full(ntrunc + 1 - m, m)
                         for m in range(ntrunc + 1)])
    return ls.astype(np.int32), ms.astype(np.int32)


def sph_lm_indices_verify(n=5):
    """Assert the generated index scheme matches shtns (l, m, idx) for
    ntrunc 1..n-1 (shtns requires lmax >= 1). Returns False when shtns
    is unavailable."""
    if _shtns is None:
        return False
    for ntrunc in range(1, n):
        s = _shtns.sht(ntrunc, ntrunc, 1,
                       _shtns.sht_orthonormal + _shtns.SHT_NO_CS_PHASE)
        ls, ms = sph_lm_indices(s.nlm)
        assert np.array_equal(ls, np.asarray(s.l, dtype=np.int32)), \
            'shtns l mismatch at ntrunc %d' % ntrunc
        assert np.array_equal(ms, np.asarray(s.m, dtype=np.int32)), \
            'shtns m mismatch at ntrunc %d' % ntrunc
        for j in range(s.nlm):
            assert s.idx(int(ls[j]), int(ms[j])) == j, \
                'shtns idx mismatch at ntrunc %d j %d' % (ntrunc, j)
    return True


if _shtns is not None:
    sph_lm_indices_verify()


#==============================================================================
#  Truncate an m>=0 triangular shtns coefficient array to a smaller ntrunc
#  (m-major block order: block m covers l=m..ntrunc, cf. sph_lm_indices).
#  Used by the Phase-8 export side to cap the SSDB sph_coeffs_scat dimension
#  at an x-dependent budget instead of the worst-cell padding.
#==============================================================================

def sph_array_truncate(coeffs, ntrunc_target):
    """return coeffs (1-D shtns m>=0 triangular array) truncated to
    ntrunc_target; arrays already at or below the target pass through."""
    coeffs = np.asarray(coeffs)
    n = coeffs.size
    r = int(round(np.sqrt(8.0 * n + 1.0)))
    if r * r != 8 * n + 1 or r % 2 == 0:
        raise ValueError(
            'coefficient array length %d is not a shtns m>=0 triangular '
            'count (ntrunc+1)(ntrunc+2)/2' % n)
    ntrunc = (r - 3) // 2
    L = int(ntrunc_target)
    if L >= ntrunc:
        return coeffs
    out = np.empty((L + 1) * (L + 2) // 2, dtype=coeffs.dtype)
    src = 0
    dst = 0
    for m in range(L + 1):
        keep = L + 1 - m
        out[dst:dst + keep] = coeffs[src:src + keep]
        src += ntrunc + 1 - m   # source block length
        dst += keep
    return out


#==============================================================================
#  NetCDF routines
#==============================================================================


def read_sph_indices(group):
    """Read the (l,m) index pairing of a group's mueller_sph.

    Returns a list of per-element int32 pair arrays [l0,m0, l1,m1, ...]
    (aligned element-by-element and positionally coefficient-by-coefficient
    with mueller_sph), or None when the group carries no indices.

    Current schema: separate sph_l / sph_m vlen variables (self-describing).
    The short-lived first schema (single interleaved 'sph_indices'
    variable) is still accepted for stores written with it.
    """
    if "sph_l" in group.variables and "sph_m" in group.variables:
        ls = group["sph_l"][:]
        ms = group["sph_m"][:]
        return [np.stack([np.asarray(l, dtype=np.int32),
                          np.asarray(m, dtype=np.int32)], axis=1).ravel()
                for l, m in zip(ls, ms)]
    if "sph_indices" in group.variables:  # legacy interleaved single var
        return [np.asarray(p, dtype=np.int32) for p in group["sph_indices"][:]]
    return None


def read_data_from_separate_netcdf(filename, particle_id):
    """Try to read particle data from file.
    Returns:
         dict or None: {mueller_sph, relerror} or None if no data is available.
    """
    filename += particle_id + ".nc"
    return read_data_from_netcdf(filename, particle_id)

def write_data_to_separate_netcdf(filename, particle_id, mueller_sph, relerror,
                                  max_len_coeffs, max_ntrunc,ext_mat, nstokes=4,
                                  back_mat=None, back_mat_ring=None):
    filename += particle_id + ".nc"
    append_data_to_netcdf(filename, particle_id, mueller_sph, relerror,
                          max_len_coeffs, max_ntrunc,ext_mat, nstokes,
                          back_mat, back_mat_ring)

def read_data_from_netcdf(filename, particle_id):
    """Try to read particle data from file.
    Returns:
         dict or None: {mueller_sph, relerror} or None if no data is available.
    """
    if os.path.exists(filename):
        with Dataset(filename, "r") as rootgrp:
            try:
                ret = {}
                mueller_sph_array = rootgrp[f"/{particle_id}/mueller_sph"][:]
                
                ret["mueller_sph"]=[c.view(np.complex64).astype(
                        np.complex128) for c in mueller_sph_array]   
                ret["relerror"] = rootgrp[f"/{particle_id}/relerror"][:]
                ret["ext_matrix"] = rootgrp[f"/{particle_id}/ext_mat"][:]

                # explicit l/m index arrays (returned as int32 [l,m] pair
                # arrays per element, see read_sph_indices). Stores written
                # before this schema addition do not contain them -> None;
                # the triangular order of mueller_sph is unchanged.
                ret["sph_indices"] = read_sph_indices(
                    rootgrp[f"/{particle_id}"])

                group = rootgrp[f"/{particle_id}"]
                ret["max_len_coeffs"] = group.max_len_coeffs
                ret["max_ntrunc"] = group.max_ntrunc

                # Phase 6: backscatter matrix + Theta=180 ring. Stores written
                # before Phase 6 do not contain them -> None (callers must
                # treat that as a cache miss if they need backscatter data).
                # netCDF4 raises IndexError (older versions KeyError) for a
                # missing variable -> catch both.
                try:
                    ret["back_matrix"] = rootgrp[
                        f"/{particle_id}/back_mat"][:]
                    ret["back_mat_ring"] = rootgrp[
                        f"/{particle_id}/back_mat_ring"][:]
                except (KeyError, IndexError):
                    ret["back_matrix"] = None
                    ret["back_mat_ring"] = None
                return ret
            except KeyError:
                pass
    return None

def read_fulldata_from_netcdf(filename):
    """Try to read all data from file.
    Returns:
         dict or None: {mueller_sph, relerror} or None if no data is available.
    """
    
    if os.path.exists(filename):
        with Dataset(filename, "r") as rootgrp:
            try:
                ret = {}
                
                
                ret["mueller_sph_array"]=[[c.view(np.complex64).astype(np.complex128) 
                    for c in g["mueller_sph"][:]] 
                        for g in rootgrp.groups.values()] 
    
                ret["relerror"] = [g["relerror"][:] 
                    for g in rootgrp.groups.values()]
                ret["ext_matrix"] = np.array([g["ext_mat"][:] 
                    for g in rootgrp.groups.values()])

                # l/m index arrays per element; None for pre-addition stores
                ret["sph_indices"] = [read_sph_indices(g)
                                      for g in rootgrp.groups.values()]

                ret["max_len_coeffs"] = np.array([g.max_len_coeffs 
                    for g in rootgrp.groups.values()])
                ret["max_ntrunc"] = np.array([g.max_ntrunc 
                    for g in rootgrp.groups.values()])

                # Phase 6: tolerate stores without backscatter variables
                ret["back_matrix"] = [g["back_mat"][:] if "back_mat" in g
                                      else None
                                      for g in rootgrp.groups.values()]
                ret["back_mat_ring"] = [g["back_mat_ring"][:]
                                        if "back_mat_ring" in g else None
                                        for g in rootgrp.groups.values()]
                return ret
            except KeyError:
                pass
    return None


def write_variables(rootgrp, particle_id, mueller_sph, relerror, max_len_coeffs,
                    max_ntrunc, ext_mat, back_mat=None, back_mat_ring=None):
    """Create mueller and relerror in NetCDF group."""

    vlen_t = rootgrp.vltypes["float_vlen"]

    # int32 vlen type for the (l,m) indices (created once per file, where
    # the float_vlen type is created: append_data_to_netcdf /
    # write_mueller_array_to_netcdf)
    if "int32_vlen" not in rootgrp.vltypes:
        rootgrp.createVLType(np.int32, "int32_vlen")
    vlen_i = rootgrp.vltypes["int32_vlen"]

    mueller_sph_array = np.ones(len(mueller_sph), dtype=object)
    sph_l_array = np.ones(len(mueller_sph), dtype=object)
    sph_m_array = np.ones(len(mueller_sph), dtype=object)
    for i, c in enumerate(mueller_sph):
        mueller_sph_array[i] = c.astype(np.complex64).view(np.float32)
        # explicit l / m index arrays for every stored coefficient
        ls, ms = sph_lm_indices(len(c))
        sph_l_array[i] = ls
        sph_m_array[i] = ms

    var_mueller = rootgrp.createVariable(f"/{particle_id}/mueller_sph",
                                         vlen_t, ("stokes2",))
    var_mueller[:] = mueller_sph_array

    # sph_l[k] / sph_m[k] = l- and m-indices of element k, positionally
    # aligned with mueller_sph[k]: mueller_sph[k][j] multiplies Y_{sph_l[k][j],
    # sph_m[k][j]} (triangular m>=0 shtns order; m<0 via complex conjugate)
    var_sph_l = rootgrp.createVariable(f"/{particle_id}/sph_l",
                                       vlen_i, ("stokes2",))
    var_sph_l[:] = sph_l_array
    var_sph_m = rootgrp.createVariable(f"/{particle_id}/sph_m",
                                       vlen_i, ("stokes2",))
    var_sph_m[:] = sph_m_array

    var_relerror = rootgrp.createVariable(f"/{particle_id}/relerror", "f4",
                                          ("stokes2",))
    var_relerror[:] = relerror
    
    var_ext_mat = rootgrp.createVariable(f"/{particle_id}/ext_mat", "f4",
                                          ("stokes","stokes",))
    var_ext_mat[:] = ext_mat

    # Phase 6: per-vertex backscatter matrix and its Theta=180 ring
    # (raw mueller-grid values). Optional: old call sites omit them.
    if back_mat is not None:
        var_back_mat = rootgrp.createVariable(f"/{particle_id}/back_mat",
                                              "f4", ("stokes", "stokes",))
        var_back_mat[:] = back_mat

    if back_mat_ring is not None:
        if "backring" not in rootgrp.dimensions:
            rootgrp.createDimension("backring", np.shape(back_mat_ring)[0])
        var_back_ring = rootgrp.createVariable(f"/{particle_id}/back_mat_ring",
                                               "f4",
                                               ("backring", "stokes", "stokes"))
        var_back_ring[:] = back_mat_ring

    group = rootgrp[f"/{particle_id}"]
    group.max_len_coeffs = np.int32(max_len_coeffs)
    group.max_ntrunc = np.int32(max_ntrunc)


# @typhon.utils.Timer()
def append_data_to_netcdf(filename, particle_id, mueller_sph, relerror,
                          max_len_coeffs, max_ntrunc,ext_mat, nstokes=4,
                          back_mat=None, back_mat_ring=None):
    """Appends data to NetCDF file.
    File is created if it doesn't exist.
    """
    if not os.path.exists(filename):
        print(f"Writing {particle_id} to file {filename}")
        with Dataset(filename, "w") as rootgrp:
            rootgrp.createDimension("stokes2", nstokes ** 2)
            rootgrp.createDimension("stokes", nstokes )
            vlen_t = rootgrp.createVLType(np.float32, "float_vlen")

            write_variables(rootgrp, particle_id, mueller_sph, relerror,
                            max_len_coeffs, max_ntrunc, ext_mat,
                            back_mat, back_mat_ring)
    else:
        print(f"Appending {particle_id} to file {filename}")
        with Dataset(filename, "a") as rootgrp:
            write_variables(rootgrp, particle_id, mueller_sph, relerror,
                            max_len_coeffs, max_ntrunc, ext_mat,
                            back_mat, back_mat_ring)


#==============================================================================
#  Mueller-array artifact: one NetCDF4 file per (T, f, size) run
#  (replaces the *_mueller_data.pkl internal artifact, see
#  plans/DDA-mueller-netcdf-migration-plan.md). Schema mirrors the per-vertex
#  temp-cache stores: root dims stokes/stokes2 + float_vlen + int32_vlen, one
#  group per COMPUTED icosphere vertex (uncomputed = None -> no group),
#  ordering via the group attr vertex_index (row index into the run's
#  `vertices` array). Group-name-sorted iteration is explicitly NOT trusted
#  for order.
#  Each mueller_sph element has parallel sph_l / sph_m elements (int32 vlen)
#  giving the l- and m-index of every stored coefficient, positionally
#  aligned with the coefficients (see sph_lm_indices / read_sph_indices).
#  Stores predating this addition simply lack the variables -> readers
#  return None. (The first schema iteration used a single interleaved
#  'sph_indices' variable; it is still accepted by the readers.)
#==============================================================================

MUELLER_UNITS_NOTE = ('raw k-normalized mueller-grid units, m^2/sr '
                      'conversion in rawdata2average')


def mueller_vertex_id(vertex):
    """NetCDF group name for one icosphere vertex — identical to the
    temp-cache file/group ids (coords x1e5, micrometres)."""
    return ('vertex_X{XV:07.0f}_Y{YV:07.0f}_Z{ZV:07.0f}').format(
        XV=vertex[0] * 1e5, YV=vertex[1] * 1e5, ZV=vertex[2] * 1e5)


def write_mueller_array_to_netcdf(filename, mueller_array, vertices,
                                  rel_err_array, ext_mat_array,
                                  back_mat_array=None, meta=None,
                                  max_len_array=None, max_ntrunc_array=None,
                                  back_mat_ring_array=None):
    """Save the per-vertex mueller_array of one ARO run as a single NetCDF4
    file (final *_mueller_data.nc artifact).

    Args:
        mueller_array: list of len(vertices) entries, each None (vertex not
            computed) or a list of 16 ragged complex64 spherical-harmonic
            coefficient arrays.
        vertices: (N,3) vertex-coordinate array of the run (row order is the
            list index; ids come from mueller_vertex_id).
        rel_err_array: (N,16) per-Stokes-element expansion relerrors.
        ext_mat_array: (N,4,4) per-vertex extinction matrices (raw units).
        back_mat_array: optional (N,4,4) per-vertex backscatter matrices.
        meta: dict of provenance attrs written at root level; recognized
            keys: shape_name, temperature, frequency, wavelength, refr_index
            (complex -> stored as float64 [re, im]), adda_version,
            adda_binary, max_len_coeffs, max_ntrunc.
        max_len_array / max_ntrunc_array: optional per-vertex SHT diagnostics
            (group attrs); max_len falls back to len(coeff) otherwise.
        back_mat_ring_array: optional list of per-vertex Theta=180 rings.

    Complex convention (lossless, same as the temp-cache path): coefficients
    are stored complex64 .view(np.float32) as float_vlen arrays.

    Note: like write_variables, back_mat rings share one root dim
    'backring' -- rings of differing length cannot coexist in one file.
    The final run-end artifact normally omits rings entirely (sub_control
    only aggregates ext/back 4x4 arrays; rings live in the per-vertex
    temp cache).
    """
    n_vertices = len(mueller_array)
    if np.shape(vertices)[0] != n_vertices:
        raise ValueError('vertices has %d rows but mueller_array has %d '
                         'entries' % (np.shape(vertices)[0], n_vertices))

    meta = dict(meta) if meta else {}

    with Dataset(filename, "w", format="NETCDF4") as rootgrp:
        rootgrp.createDimension("stokes", 4)
        rootgrp.createDimension("stokes2", 16)
        rootgrp.createVLType(np.float32, "float_vlen")

        # --- root provenance attrs (mirror the *_overview_data.mat fields) ---
        for key in ('shape_name', 'adda_version', 'adda_binary'):
            if key in meta:
                setattr(rootgrp, key, str(meta[key]))
        for key in ('temperature', 'frequency', 'wavelength'):
            if key in meta:
                setattr(rootgrp, key, float(meta[key]))
        if 'refr_index' in meta:
            m = complex(meta['refr_index'])
            rootgrp.refr_index = np.array([m.real, m.imag], dtype='f8')
        rootgrp.mueller_units = MUELLER_UNITS_NOTE
        rootgrp.n_vertices = np.int32(n_vertices)
        rootgrp.max_len_coeffs = np.int32(meta.get('max_len_coeffs', 0))
        rootgrp.max_ntrunc = np.int32(meta.get('max_ntrunc', 0))
        # Phase 8 provenance: size parameter + adaptive scatgrid cap
        # (-1 = historic static grid); optional keys, readers must tolerate
        # their absence in pre-Phase-8 stores
        if 'size_parameter' in meta:
            rootgrp.size_parameter = float(meta['size_parameter'])
        if 'scatgrid_ntrunc_cap' in meta:
            rootgrp.scatgrid_ntrunc_cap = np.int32(meta['scatgrid_ntrunc_cap'])
        if 'scatgrid_adaptive' in meta:
            rootgrp.scatgrid_adaptive = int(bool(meta['scatgrid_adaptive']))
        rootgrp.created = datetime.datetime.now(
            datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

        # --- one group per computed vertex ---
        n_computed = 0
        for i, sph in enumerate(mueller_array):
            if sph is None:
                continue

            pid = mueller_vertex_id(vertices[i])

            mlen = (max_len_array[i] if max_len_array is not None
                    else max(len(c) for c in sph))
            mntr = (max_ntrunc_array[i] if max_ntrunc_array is not None
                    else int(meta.get('max_ntrunc', 0)))

            backm = (back_mat_array[i]
                     if back_mat_array is not None else None)
            backring = (back_mat_ring_array[i]
                        if back_mat_ring_array is not None else None)

            write_variables(rootgrp, pid, sph, rel_err_array[i], mlen, mntr,
                            ext_mat_array[i], backm, backring)

            group = rootgrp[f"/{pid}"]
            group.vertex_index = np.int32(i)
            n_computed += 1

        print(f"Writing mueller array ({n_computed}/{n_vertices} vertices) "
              f"to file {filename}")


def load_mueller_array_from_netcdf(filename, n_vertices=None,
                                   return_indices=False):
    """Load a *_mueller_data.nc written by write_mueller_array_to_netcdf.

    Returns:
        list of length n_vertices, indexed EXACTLY like the old pickle /
        the run's `vertices` array: entry i is None for an uncomputed
        vertex, else a list of 16 complex64 coefficient arrays (raw
        stored data, no resampling).

    return_indices=True: returns (mueller_array, sph_indices) instead,
        where sph_indices[i] is None (uncomputed vertex or a store written
        before the sph_l/sph_m schema addition) else a list of 16 int32
        arrays [l0,m0, l1,m1, ...] aligned element-by-element with
        mueller_array[i] (de-interleaved from the stored sph_l / sph_m
        variables; triangular m>=0 shtns order).

    n_vertices defaults to the file's n_vertices root attr.
    Raises ValueError on missing / duplicate / out-of-range vertex_index
    group attrs (ordering is never inferred from group order).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'no mueller artifact: {filename}')

    with Dataset(filename, "r") as rootgrp:
        if n_vertices is None:
            try:
                n_vertices = int(rootgrp.n_vertices)
            except AttributeError:
                raise ValueError(
                    f'{filename} has no n_vertices root attr -- pass '
                    f'n_vertices explicitly')

        mueller_array = [None] * n_vertices
        sph_indices = [None] * n_vertices
        seen = {}
        for gname, group in rootgrp.groups.items():
            try:
                idx = int(group.vertex_index)
            except AttributeError:
                raise ValueError(
                    f'group {gname} of {filename} has no vertex_index attr '
                    f'-- refusing to infer vertex order from group order')
            if idx < 0 or idx >= n_vertices:
                raise ValueError(
                    f'group {gname} vertex_index {idx} out of range '
                    f'(n_vertices={n_vertices})')
            if idx in seen:
                raise ValueError(
                    f'duplicate vertex_index {idx}: groups {seen[idx]} '
                    f'and {gname}')
            seen[idx] = gname
            mueller_array[idx] = [c.view(np.complex64)
                                  for c in group["mueller_sph"][:]]
            if return_indices:
                sph_indices[idx] = read_sph_indices(group)

    if return_indices:
        return mueller_array, sph_indices
    return mueller_array

