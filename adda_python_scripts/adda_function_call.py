# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 17:24:44 2015

@author: u242031
"""


import subprocess as sub
import numpy as np

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

def rotate_mueller(M,phi):
#    Function to rotate the Mueller matrix by an angle of phi.

    rot=np.matrix( [[1.,0.,0,,0],
                    [0.,np.cos(2.*phi), np.sin(2.*phi),0.],
                    [0.,-np.sin(2.*phi), np.cos(2.*phi),0.],
                    [0.,0.,0.,1.]] )


    #Let's rotate
    Mrot=np.mat(M)*rot

    return Mrot


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



    with zipfile.ZipFile(zipname, mode='w') as zf:


        #Make arguments for adda
        #-----------------------


        #refractive index
        m_str=['-m',str(m.real),str(m.imag)]

        #Wavelength
        wavelen_str=['-lambda',str(lamb)]

        #Size
        size_str=['-size',str(D)]

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



        #Load mueller_scatgrid
        data_temp=np.loadtxt(path_name+'/mueller_scatgrid', skiprows=1)

        phis_grid=data_temp[:,1]
        thetas_grid=data_temp[:,0]

        #make it unique
        phis_grid=np.unique(phis_grid)
        thetas_grid=np.unique(thetas_grid)

        #Mueller matrix
        mueller_raw_array=data_temp[:,2:]
        qmm=mueller_raw_array.shape
        mueller_raw_array=np.reshape(mueller_raw_array,(phis_grid.size,
                                            thetas_grid.size,qmm[1]))


    #    if alpha_index==1:
        mueller_array=mueller_raw_array*0.

         #Unit vector incidence v-pol (phi=0)
        vi=np.array([np.sin(theta*np.pi/180),
                     0.,
                     np.cos(theta*np.pi/180)])


        #Loop over all scattering angles
        for i in xrange(0,phis_grid.size):
            for j in xrange(0,thetas_grid.size):




                #Unit vector scattering direction
                ks=np.array([np.sin(thetas_grid[j]*np.pi/180)*np.cos(phis_grid[i]*np.pi/180),
                             np.sin(thetas_grid[j]*np.pi/180)*np.sin(phis_grid[i]*np.pi/180),
                             np.cos(thetas_grid[j]*np.pi/180)])

                #rotation angle
                psi=np.arcsin(np.dot(vi,ks))

                #Mueller matrix in particle reference polarisation
                mueller_mat_P=reshape(mueller_raw_array[i,j,:],(4,4))

                #Mueller matrix in lab reference polarisation
                mueller_mat_L=rotate_mueller(mueller_mat_P,psi)

                mueller_array[i,j,:]=mueller_mat_L.reshape(4*4)






        #look after the adda output
        filelist = glob.glob(path_name+'/*')


        #and now put it into a zipfile
        for filename in filelist:


            folder_structure=os.path.join('theta_'+str(theta),
            'gamma_'+str(gamma),'beta_'+str(beta),
            'alpha_'+str(alpha),os.path.basename(filename))

            zf.write(filename, folder_structure, compress_type=compression)


        #clean up
        shutil.rmtree(path_name)

        #Put output in dictionary
        output_data['mueller_array']=mueller_array
        output_data['phis_grid']=phis_grid
        output_data['thetas_grid']=thetas_grid


        return output_data


#==============================================================================
#  Romberg integration over alpha angle (Euler angle)
#==============================================================================
def romberg_alpha(input_data):


    #alpha intervall
    alpha_ab=input_data['alpha_intervall']


    #First step


    #Zählindex
    Nidx=1
    x_add=alpha_ab

    #Get the results for first two values
    for i in xrange(0,x_add.size):



        input_data['alpha']=x_add[i]

        #calling Adda
        [output_data]=adda_call(input_data)

        #size of Mueller Matrix
        q=mueller_matrix.shape

        #Add forth dimension
        mueller_matrix=np.reshape(mueller_matrix,(q[0],q[1],q[2],1))

        if i==0:
            y_add=mueller_matrix
        else:
            y_add=np.concatenate((y_add,mueller_matrix),3)



    #put the new Values in our storage
    x=x_add
    y=y_add

    #sort the values
    ins=np.argsort(x)
    x_sort=x[ins]
    y_sort=y[:,:,:,ins]



    #integrate over alpha orientation
    trapz_int=np.sum(np.diff(x_sort)*(y_sort[:,:,:,0:-1]+y_sort[:,:,:,1:])/2)
    trapz_int=np.reshape(trapz_int,q[0],q[1],q[2])

    #First result
    R=np.zeros(q[0],q[1],q[2],limit,limit)
    R[:,:,:,0,0]=trapz_int;

    krit=1




    #let us do some refinement
    while (not(krit<eps and Nidx>2 ) and Nidx<limit):


        #Counter
        Nidx=Nidx+1;


        #The new sample points
        if( mod(Nidx ,2) ==0):
            n = 2^(Nidx/2);
        else:
            n = 3*2^(((Nidx-1)/2)-1) ;

        x_new=np.linspace(p_orientation_alpha[0],p_orientation_alpha[1],n+1);


        #Look for already existing values
        for i in xrange(0,x_new.size):
            horst=where(x_new(i)==x);

            fidx=horst[0];

            #If a values does not exist...
            if fidx.size<1:

                input_data['alpha']=x_new[i]

                #calling Adda
                [output_data]=adda_call(input_data)

#                [mueller_array, phis_grid, thetas_grid]=adda_call(lamb, theta,\
#                    D, m, x_new[i], beta, gamma, ramdrive, simbasename)

                #size of Mueller Matrix
                q=mueller_matrix.shape

                #Add forth dimension
                mueller_matrix=np.reshape(mueller_matrix,(q[0],q[1],q[2],1))


                #Add additional values
                if x_add.size==0:
                    x_add=x_new[i];
                    y_add=mueller_matrix
                else:
                    x_add=np.concatenate((x_add,x_new(i)),0)
                    y_add=np.concatenate((y_add,mueller_matrix),3)






        #Put the new values into storage
        x=np.concatenate((x,x_add),0)
        y=np.concatenate((y,y_add),3)

        #sort the values
        ins=np.argsort(x)
        x_sort=x[ins]
        y_sort=y[:,:,:,ins]

        #integrate over alpha orientation
        trapz_int=np.sum(np.diff(x_sort)*(y_sort[:,:,:,0:-1]+y_sort[:,:,:,1:])/2)
        trapz_int=np.reshape(trapz_int,q[0],q[1],q[2])

        #integral
        R[:,:,:,Nidx-1,0]=trapz_int

        #Extrapolation for better results ;-)
        for k in xrange(2,x_new.size+1):
#
            R[Nidx-1,k-1]=(4**(k-1)*R[:,:,:,Nidx-1,k-2]-R[:,:,:,Nidx-2,k-2]) / (4**(k-1)-1);


        #compare old and new value
        krit=np.amax(abs((R[:,:,:,Nidx-1,Nidx-1]-R[:,:,:,Nidx-2,Nidx-2])) / abs(R[:,:,:,Nidx-1,Nidx-1]))


        #Mueller matrix averaged over alpha
        mueller_matrix_avg=np.reshape(R[:,:,:,Nidx-1,Nidx-1],(q[0],q[1],q[2]))


        #Put output in dictionary
        output_data['mueller_matrix_avg']=mueller_matrix_avg
        output_data['phis_grid']=phis_grid
        output_data['thetas_grid']=thetas_grid


    return output_data