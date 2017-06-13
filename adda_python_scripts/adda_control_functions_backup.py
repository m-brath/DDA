# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:24:44 2015

@author: u242031
"""


import subprocess as sub
import numpy as np
import scipy as sp
import glob
import os
import shutil
import pdb

import zipfile
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED

modes = { zipfile.ZIP_DEFLATED: 'deflated',
          zipfile.ZIP_STORED:   'stored'}

#==============================================================================
# additional function
#==============================================================================

def particle2lab_ref(mueller_raw_array,theta_p,phi_p,alpha,beta,gamma):
# Function to transform the scattering angle from the particle reference system
# to the laboratory system, see also :
# Mishenko et al.; Scattering, Absorption and Emission of Light by Small Particles;
# 2002, pp. 42-46
# IMPORTANT the angles have to be in DEGREE!

    from scipy.interpolate import interp2d
    from scipy.interpolate import griddata
    import matplotlib.pyplot as plt

    #define combined grid in DEGREE!
    pg, tg = np.meshgrid(phi_p,theta_p)  #this will act as regualar scattering
                                         #grid in the lab system, because the
                                         #lag scattering grid should have the
                                         #same dimensions as the particle
                                         #scattering grid.



    #convert input to radian
    phi_p=phi_p*np.pi/180.
    theta_p=theta_p*np.pi/180.
    alpha=alpha*np.pi/180.
    beta=beta*np.pi/180.
    gamma=gamma*np.pi/180.


    #Allocate matrix
    theta_l=tg.copy()*np.nan
    phi_l=pg.copy()*np.nan

#    pdb.set_trace()

    #Loop over all scattering angles
    for j in xrange(0,theta_p.size):
        for i in xrange(0,phi_p.size):

            #define some abrevations
            tp=theta_p[j]
            pp=phi_p[i]

            cb=np.cos(-beta)
            cg=np.cos(-gamma)
            sb=np.sin(-beta)
            sg=np.sin(-gamma)

            cppa=np.cos(pp+alpha)
            sppa=np.sin(pp+alpha)

            ctp=np.cos(-tp)
            stp=np.sin(-tp+1e-5) # the addition of 1e-5 is there to avoid that stp
                                # can be exact 0

            #cos(theta_l) EQ (2.80)
            ctl=ctp*cb+stp*sb*cppa

            #sin(phi_l) EQ (2.81)
            spl=(-cb*sg*stp*cppa+cg*stp*sppa+sb*sg*ctp)/stp

            #cos(phi_l) EQ (2.82)
            cpl=(cb*cg*stp*cppa+sg*stp*sppa-sb*cg*ctp)/stp

            theta_l[j,i]=np.arccos(ctl)*180./np.pi
            phi_l[j,i]=np.arctan2(spl,cpl)*180./np.pi

            if phi_l[j,i]<0:
                phi_l[j,i]=360.+phi_l[j,i]
            elif phi_l[j,i]>360:
                phi_l[j,i]=phi_l[j,i]-360.


    #--------------------------------------------------------------------------
    # Interpolate to regualar scattering grid in laboratory reference system
    #--------------------------------------------------------------------------

    #Get size of the mueller matrix array
    pq=mueller_raw_array.shape

    #Allocate matrix
    mueller_int_array=mueller_raw_array.copy()*np.nan;



    points=np.transpose(np.vstack((phi_l.flatten(),theta_l.flatten())))

    #Loop over Mueller matrix components
    for m in xrange(0,pq[2]):

        #generate interpolation grid
        temp=mueller_raw_array[:,:,m]
#        func=interp2d(theta_l,phi_l,temp,kind='linear', copy=True)


        values=mueller_raw_array[:,:,m].flatten()
#        pdb.set_trace()
        test=griddata(points, values,(pg,tg), method='nearest')

        #now interpolate on regular scattering grid in lab system


        mueller_int_array[:,:,m]=test
#            #Mueller matrix in particle reference polarisation
#            mueller_mat_P=np.reshape(mueller_raw_array[j,i,:],(4,4))

    return mueller_int_array




def rotate_mueller(M,phi1,phi2):
#    Function to rotate the Mueller matrix by an angle of phi.
#    Angles are in RADIAN!


    rot1=np.matrix( [[1.,0.,0.,0.],
                    [0.,np.cos(2.*phi1), np.sin(2.*phi1),0.],
                    [0.,-np.sin(2.*phi1), np.cos(2.*phi1),0.],
                    [0.,0.,0.,1.]] )

    rot2=np.matrix( [[1.,0.,0.,0.],
                    [0.,np.cos(2.*phi2), np.sin(2.*phi2),0.],
                    [0.,-np.sin(2.*phi2), np.cos(2.*phi2),0.],
                    [0.,0.,0.,1.]] )


    #Let's rotate
    Mrot=rot2*np.mat(M)*rot1

    return Mrot


#    return M


def extinction_matrix(s11,s12,s21,s22,lamb):
    #Function for calculation of extinction matrix from scattering amplitude matrix
    #according to Mishenko et al.,2002

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


    #wavelength
    lamb=input_data['lambda']

    #Particle Size
    D=input_data['Size']

    #refractive Index
    m=input_data['refrative_ndex']

    #incidence angle
    theta=input_data['theta']

    #alpha
    alpha=input_data['alpha']

    #beta angle
    beta=input_data['beta']

    #gamma angle
    gamma=input_data['gamma']

    #ramdrive
    ramdrive=input_data['ramdrive']

    #simulation basename
    simbasename=input_data['simbasename']

    #Name of the zipname
    zipname=input_data['zipname']

    #main command for adda
    main_command=input_data['main_command']

    #Adda's static command line options
    static_clo=input_data['static_clo']



    with zipfile.ZipFile(zipname, mode='a') as zf:


        #Make arguments for adda
        #-----------------------


        #refractive index
        m_str=['-m',str(m.real),str(m.imag)]

        #Wavelength
        wavelen_str=['-lambda',str(lamb)]

        #Size
        size_str=['-eq_rad',str(D)]

        #Particle orientation
        pori_str=['-orient',str(alpha),str(beta),str(gamma)]

        #Incidence direction
        px=[np.sin(theta*np.pi/180),0,np.cos(theta*np.pi/180)]
        prop_str=['-prop',str(px[0]),str(px[1]),str(px[2])]

        #Output directory
        temp_dir=simbasename
        path_name=ramdrive+'/'+temp_dir
        dir_str=['-dir',path_name]

        #command_str
        commando=main_command+static_clo+\
                  m_str+wavelen_str+size_str+pori_str+\
                  prop_str+dir_str


        print('  ')
        print(commando)

        sub.call(commando)

#        pdb.set_trace()

        #Load amplitude matrix
        ampl_data_temp=np.loadtxt(path_name+'/ampl_scatgrid', skiprows=1)

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
        data_temp=np.loadtxt(path_name+'/mueller_scatgrid', skiprows=1)

        phis_grid=data_temp[:,1]
        thetas_grid=data_temp[:,0]

        test_phi=data_temp[:,1]
        test_theta=data_temp[:,0]

        #make it unique
        phis_grid=np.unique(phis_grid)
        thetas_grid=np.unique(thetas_grid)



        #Mueller matrix
        mueller_raw_array=data_temp[:,2:]
        qmm=mueller_raw_array.shape
        mueller_raw_array=np.reshape(mueller_raw_array,(thetas_grid.size,
                                            phis_grid.size,qmm[1]))

        #test
#        test_phi2=np.reshape(test_phi,(phis_grid.size,thetas_grid.size))
#        test_theta2=np.reshape(test_theta,(phis_grid.size,thetas_grid.size))


#        pdb.set_trace()
        #transfor the mueller matrix from particle reference system to laborator system
#        mueller_array=particle2lab_ref(mueller_raw_array,thetas_grid,phis_grid,alpha,beta,gamma)
        mueller_array=mueller_raw_array

#        #Unit vector incidence v-pol (phi=0)
#        vi=np.array([np.cos(theta*np.pi/180),
#                     0.,
#                     -np.sin(theta*np.pi/180)])
#
#        #Unit vector incidence direction
#        ki=np.array([np.sin(theta*np.pi/180)*np.cos(0.*np.pi/180),
#                     np.sin(theta*np.pi/180)*np.sin(0.*np.pi/180),
#                     np.cos(theta*np.pi/180)])
#
#
#        test=mueller_raw_array[:,:,0]
#        pdb.set_trace()
#
#        #Loop over all scattering angles
#        for j in xrange(0,thetas_grid.size):
#            for i in xrange(0,phis_grid.size):
#
#
#                cts=np.cos(thetas_grid[j]*np.pi/180.)
#                cti=np.cos(theta*np.pi/180.)
#
#                sts=np.sin(thetas_grid[j]*np.pi/180.)
#                sti=np.sin(theta*np.pi/180.)
#
#                cpsi=np.cos(phis_grid[i]-0.)
#
#                cT=cts*cti+sts*sti*cpsi
#
#                cs1=(cts-cti*cT)/(sti*cT)
#                cs2=(cti-cts*cT)/(sts*cT)
#
#
#
#                #transform the polararization to lab system
#
#                #Unit vector scattering direction
#                ks=np.array([np.sin(thetas_grid[j]*np.pi/180)*np.cos(phis_grid[i]*np.pi/180),
#                             np.sin(thetas_grid[j]*np.pi/180)*np.sin(phis_grid[i]*np.pi/180),
#                             np.cos(thetas_grid[j]*np.pi/180)])
#
#                #Unit vector scattered v-pol (phi=0)
#                vs=np.array([np.cos(thetas_grid[j]*np.pi/180)*np.cos(phis_grid[i]*np.pi/180),
#                             np.cos(thetas_grid[j]*np.pi/180)*np.sin(phis_grid[i]*np.pi/180),
#                             -np.sin(thetas_grid[j]*np.pi/180)])
#
#
#
#
#
#                #rotation angle
#                sig1=np.arccos(cs1)
#                psi1=np.arcsin(np.dot(vi,ks))
#
#                #rotation angle
#                sig2=np.arccos(cs2)
#                psi2=np.arcsin(np.dot(vs,ki))
#
#                #Mueller matrix in particle reference polarisation
#                mueller_mat_P=np.reshape(mueller_raw_array[j,i,:],(4,4))
#
#                #Mueller matrix in lab reference polarisation
#                mueller_mat_L=rotate_mueller(mueller_mat_P,psi1,-psi2)
#
#                mueller_array[j,i,:]=mueller_mat_L.reshape(4*4)
#
#
#        #make mueller to stokes matrix, even though the name is still Mueller, here
#        mueller_array=mueller_array*lamb**2/(4*np.pi**2)


        #look after the adda output
        filelist = glob.glob(path_name+'/*')



        #and now put it into a zipfile
        for filename in filelist:


            folder_structure=os.path.join('theta_'+str(theta),
            'beta_'+str(beta),'gamma_'+str(gamma),
            'alpha_'+str(alpha),os.path.basename(filename))

            zf.write(filename, folder_structure, compress_type=compression)


        #clean up
        shutil.rmtree(path_name)

#        pdb.set_trace()

        #Put output in dictionary
        output_data={}
        output_data['mueller_array']=mueller_array
        output_data['phis_grid']=phis_grid
        output_data['thetas_grid']=thetas_grid
        output_data['ext_matrix']=ext_mat
#        output_data['test_phi']=test_phi2
#        output_data['test_theta']=test_theta2



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


    #First step


    #counter
    Nidx=1


    x_add=alpha_ab

    #Get the results for first two values
    for i in xrange(0,x_add.size):



        input_data['alpha']=x_add[i]

        #calling Adda
        output_data=adda_call(input_data)

#        pdb.set_trace()

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


    krit=1


    krit_ctrl=np.ones((limit))
    krit_ext_ctrl=np.ones((limit))

#    pdb.set_trace()

    #let us do some refinement
    while (not(krit<delta and Nidx>2 ) and Nidx<limit):


        #Counter
        Nidx=Nidx+1;

#        pdb.set_trace()

        #The new sample points
        if( np.mod(Nidx ,2) ==0):
            n = 2**(Nidx/2)
        else:
            n = 3*2**(((Nidx-1)/2)-1)

        x_new=np.linspace(alpha_ab[0],alpha_ab[1],n+1)
        x_add=np.array([])
#        y_add=np.array([])



        #Look for already existing values
        for i in xrange(0,x_new.size):

#            pdb.set_trace()

            horst=np.where(x_new[i]==x)

            fidx=horst[0]

            #If a values does not exist...
            if fidx.size<1:

                input_data['alpha']=x_new[i]

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



#        pdb.set_trace()


        #Put the new values into storage
        x=np.concatenate((x,x_add),0)
        y=np.concatenate((y,y_add),3)
        y_ext=np.concatenate((y_ext,y_ext_add),2)


        #sort the values
        ins=np.argsort(x)
        x_sort=x[ins]
        y_sort=y[:,:,:,ins]
        y_ext_sort=y_ext[:,:,ins]

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
        for k in xrange(2,Nidx+1):

#            pdb.set_trace()

            R[:,:,:,Nidx-1,k-1]=(4**(k-1)*R[:,:,:,Nidx-1,k-2]-R[:,:,:,Nidx-2,k-2]) / (4**(k-1)-1);
            R_ext[:,:,Nidx-1,k-1]=(4**(k-1)*R_ext[:,:,Nidx-1,k-2]-R_ext[:,:,Nidx-2,k-2]) / (4**(k-1)-1);


        #compare old and new value
#        krit=np.amax(abs((R[:,:,:,Nidx-1,Nidx-1]-R[:,:,:,Nidx-2,Nidx-2])) / abs(R[:,:,:,Nidx-1,Nidx-1]))
#        krit_ctrl[Nidx-1]=krit
#        krit_ext=np.amax(abs((R_ext[:,:,Nidx-1,Nidx-1]-R_ext[:,:,Nidx-2,Nidx-2])) / abs(R_ext[:,:,Nidx-1,Nidx-1]))
#        krit_ext_ctrl[Nidx-1]=krit_ext

        #compare old and new value
        # Element 1,1 of the Mueller matrix is taken as control value.
        krit=abs((R[0,0,0,Nidx-1,Nidx-1]-R[0,0,0,Nidx-2,Nidx-2])) / abs(R[0,0,0,Nidx-1,Nidx-1])
        krit_ctrl[Nidx-1]=krit
        krit_ext=abs((R_ext[0,0,Nidx-1,Nidx-1]-R_ext[0,0,Nidx-2,Nidx-2])) / abs(R_ext[0,0,Nidx-1,Nidx-1])
        krit_ext_ctrl[Nidx-1]=krit_ext



        #Mueller matrix averaged over alpha
        mueller_matrix_avg=np.reshape(R[:,:,:,Nidx-1,Nidx-1],(q[0],q[1],q[2]))/(alpha_ab[1]-alpha_ab[0])

        #Extinction matrix averaged over alpha
        ext_matrix_avg=np.reshape(R_ext[:,:,Nidx-1,Nidx-1],(q_ext[0],q_ext[1]))/(alpha_ab[1]-alpha_ab[0])

#        pdb.set_trace()

#    pdb.set_trace()

    #Put output in dictionary
    output_data['mueller_matrix_avg']=mueller_matrix_avg
    output_data['ext_matrix_avg']=ext_matrix_avg
    output_data['alpha_grid']=x_sort
    output_data['Nidx_alpha']=Nidx
    output_data['limit_alpha']=limit
    output_data['krit_alpha']=krit_ctrl
    output_data['krit_ext_alpha']=krit_ext_ctrl
#    output_data['R']=R
#    output_data['R_ext']=R_ext
#    output_data['y']=y_sort
#    output_data['x']=x_sort

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


        #Zählindex
        Nidx=1
        x_add=gamma_ab

        #Get the results for first two values
        for i in xrange(0,x_add.size):



            input_data['gamma']=x_add[i]

            #romberg_alpha
            output_data=romberg_alpha(input_data)

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




        #let us do some refinement
        while (not(krit<delta and Nidx>2 ) and Nidx<limit):


            #Counter
            Nidx=Nidx+1;


            #The new sample points
            if( np.mod(Nidx ,2) ==0):
                n = 2**(Nidx/2);
            else:
                n = 3*2**(((Nidx-1)/2)-1) ;

            x_new=np.linspace(gamma_ab[0],gamma_ab[1],n+1);
            x_add=np.array([])

            #Look for already existing values
            for i in xrange(0,x_new.size):
                horst=np.where(x_new[i]==x);

                fidx=horst[0];

                #If a values does not exist...
                if fidx.size<1:

                    input_data['gamma']=x_new[i]

                    #romberg_alpha
                    output_data=romberg_alpha(input_data)

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







            #Put the new values into storage
            x=np.concatenate((x,x_add),0)
            y=np.concatenate((y,y_add),3)
            y_ext=np.concatenate((y_ext,y_ext_add),2)


            #sort the values
            ins=np.argsort(x)
            x_sort=x[ins]
            y_sort=y[:,:,:,ins]
            y_ext_sort=y_ext[:,:,ins]

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
            for k in xrange(2,Nidx+1):

    #            pdb.set_trace()

                R[:,:,:,Nidx-1,k-1]=(4**(k-1)*R[:,:,:,Nidx-1,k-2]-R[:,:,:,Nidx-2,k-2]) / (4**(k-1)-1);
                R_ext[:,:,Nidx-1,k-1]=(4**(k-1)*R_ext[:,:,Nidx-1,k-2]-R_ext[:,:,Nidx-2,k-2]) / (4**(k-1)-1);


            #compare old and new value
            krit=np.amax(abs((R[:,:,:,Nidx-1,Nidx-1]-R[:,:,:,Nidx-2,Nidx-2])) / abs(R[:,:,:,Nidx-1,Nidx-1]))
            krit_ctrl[Nidx-1]=krit
            krit_ext=np.amax(abs((R_ext[:,:,Nidx-1,Nidx-1]-R_ext[:,:,Nidx-2,Nidx-2])) / abs(R_ext[:,:,Nidx-1,Nidx-1]))
            krit_ext_ctrl[Nidx-1]=krit_ext

            #Mueller matrix averaged over gamma
            mueller_matrix_avg=np.reshape(R[:,:,:,Nidx-1,Nidx-1],(q[0],q[1],q[2]))/(gamma_ab[1]-gamma_ab[0])

            #Extinction matrix averaged over gamma
            ext_matrix_avg=np.reshape(R_ext[:,:,Nidx-1,Nidx-1],(q_ext[0],q_ext[1]))/(gamma_ab[1]-gamma_ab[0])





    else:
        input_data['gamma']=gamma_ab[0]
        x_sort=gamma_ab[0]

        #romberg_alpha
        output_data=romberg_alpha(input_data)

        mueller_matrix_avg=output_data['mueller_matrix_avg']
        ext_matrix_avg=output_data['ext_matrix_avg']

#    pdb.set_trace()

    #Put output in dictionary
    output_data['mueller_matrix_avg_gamma']=mueller_matrix_avg
    output_data['ext_matrix_avg_gamma']=ext_matrix_avg
    output_data['gamma_grid']=x_sort


    return output_data






#==============================================================================
#  additional functions
#==============================================================================

def eps_ice_matzler06(f,TK):
    #Permittivity of fresh water ice according to Maetzler 2006
    #stolen from atmlab

    #Frequency
    f = f/1e9;

    #some checks
    if (f<0.01 or f>3000):
      raise StandardError('Valid range for frequency is 0.01-3000 GHz');

    if (TK<20 or TK>273.15):
      raise StandardError('Valid range for temperature is 20-273.15 K');


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