# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:24:44 2015

@author: u242031
"""


import subprocess as sub
import numpy as np
#import scipy as sp
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
            spawn_command(' '.join(commando))

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
            print('Mistral is not yet implemented, sorry!')
            
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
        spawn_command(' '.join(commando))

        #==================================================================

#        pdb.set_trace()
    except:
        print('there was something wrong...')
        print('input data:')
        print(input_data)
        print(' ')
        print('I hope, it was useful.')


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
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



#==============================================================================
#  NetCDF routines
#==============================================================================


def read_data_from_separate_netcdf(filename, particle_id):
    """Try to read particle data from file.
    Returns:
         dict or None: {mueller_sph, relerror} or None if no data is available.
    """
    filename += particle_id + ".nc"
    return read_data_from_netcdf(filename, particle_id)

def write_data_to_separate_netcdf(filename, particle_id, mueller_sph, relerror,
                                  max_len_coeffs, max_ntrunc,ext_mat, nstokes=4):
    filename += particle_id + ".nc"
    append_data_to_netcdf(filename, particle_id, mueller_sph, relerror,
                          max_len_coeffs, max_ntrunc,ext_mat, nstokes)

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

                group = rootgrp[f"/{particle_id}"]
                ret["max_len_coeffs"] = group.max_len_coeffs
                ret["max_ntrunc"] = group.max_ntrunc
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

                ret["max_len_coeffs"] = np.array([g.max_len_coeffs 
                    for g in rootgrp.groups.values()])
                ret["max_ntrunc"] = np.array([g.max_ntrunc 
                    for g in rootgrp.groups.values()])
                return ret
            except KeyError:
                pass
    return None


def write_variables(rootgrp, particle_id, mueller_sph, relerror, max_len_coeffs,
                    max_ntrunc, ext_mat):
    """Create mueller and relerror in NetCDF group."""

    vlen_t = rootgrp.vltypes["float_vlen"]

    mueller_sph_array = np.ones(len(mueller_sph), dtype=object)
    for i, c in enumerate(mueller_sph):
        mueller_sph_array[i] = c.astype(np.complex64).view(np.float32)

    var_mueller = rootgrp.createVariable(f"/{particle_id}/mueller_sph",
                                         vlen_t, ("stokes2",))
    var_mueller[:] = mueller_sph_array

    var_relerror = rootgrp.createVariable(f"/{particle_id}/relerror", "f4",
                                          ("stokes2",))
    var_relerror[:] = relerror
    
    var_ext_mat = rootgrp.createVariable(f"/{particle_id}/ext_mat", "f4",
                                          ("stokes","stokes",))
    var_ext_mat[:] = ext_mat

    group = rootgrp[f"/{particle_id}"]
    group.max_len_coeffs = np.int32(max_len_coeffs)
    group.max_ntrunc = np.int32(max_ntrunc)


# @typhon.utils.Timer()
def append_data_to_netcdf(filename, particle_id, mueller_sph, relerror,
                          max_len_coeffs, max_ntrunc,ext_mat, nstokes=4):
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
                            max_len_coeffs, max_ntrunc, ext_mat)
    else:
        print(f"Appending {particle_id} to file {filename}")
        with Dataset(filename, "a") as rootgrp:
            write_variables(rootgrp, particle_id, mueller_sph, relerror,
                            max_len_coeffs, max_ntrunc, ext_mat)

