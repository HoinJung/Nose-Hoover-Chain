#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
import math
import pandas as pd
import os


# In[ ]:


# create a box and assign particles on a cubic lattice 
# The box will be divided into cubic lattices with one atom in each lattice.
# Consider mimium image criterion
def create_box(bx, by, bz, nparticles, n_3):
    # initialize coordinates
    ix, iy, iz = [0, 0 ,0]
    coord=np.zeros(shape=[nparticles,3], dtype="float")
    # assign particle postions
    for i in range(nparticles):
        coord[i,0]=float((ix+0.5)*bx/n_3)
        coord[i,1]=float((iy+0.5)*by/n_3)
        coord[i,2]=float((iz+0.5)*bz/n_3)
        ix += 1
        if ix == n_3:
            ix = 0
            iy += 1
        if iy == n_3:
            iy = 0
            iz +=1
    print("Creating a Cubic box and Coordinate Completed")
    return coord


# In[ ]:



def init_vel(nparticles, T): 
    # initialize velocity by uniform distribution
    vel = np.random.uniform(-0.5, 0.5, nparticles*3).reshape(nparticles,3) 
    
    # define center of mass 
    cm_x=vel[:,0].sum()/nparticles
    cm_y=vel[:,1].sum()/nparticles
    cm_z=vel[:,2].sum()/nparticles
    
    # initialize kinetic energy
    ke = 0
    
    # elminate center of mass drift (make zero momentum)
    for i in range(nparticles):    
        vel[i,0] -= cm_x
        vel[i,1] -= cm_y
        vel[i,2] -= cm_z
    
    # obtain kinetic energy from velocity
    ke = 0.5 * np.square(vel).sum()
    
    # define 'scale velocity'
    T_temp = ke*2 / (3*nparticles)
    scale = math.sqrt(T/T_temp)
    vel=np.multiply(scale, vel)
    ke=0.5*np.square(vel).sum()
    
    print("Velocity Initialization Completed")
    
    return vel    


# In[4]:


#Lennard-Jones potetial : calculate potential energy and forces 
def LJ_pe_f(rc, bx, by, bz, nparticles, coord):
    # Zero the forces
    force=np.zeros(shape=[nparticles,3], dtype="float")
    pe = 0 
    vir = 0.0
    hL_x = bx/2
    hL_y = by/2
    hL_z = bz/2
    for i in range(nparticles-1):
        for j in range(i+1,nparticles):
            rx=coord[i,0]-coord[j,0]
            ry=coord[i,1]-coord[j,1]
            rz=coord[i,2]-coord[j,2]
            # adapt Periodic Boundary Condition for each coordinates
            if rx > hL_x :
                rx -= bx
            elif rx < -hL_x :
                rx += bx
            if ry > hL_y :
                ry -= by
            elif ry < -hL_y :
                ry += by
            if rz > hL_z :
                rz -= bz
            elif rz < -hL_z :
                rz += bz
            r = math.sqrt(rx**2 + ry**2 + rz**2)
            # calculate Lennard-Jones Potential Energy and Force
            if r < rc:
                pe+=4*(1/r**12-1/r**6)  #calculate potential energy
                f = 48*(1/r**14-0.5/r**8) # calculate forces
                
                # once we have f_ij, we automatically have f_ji = -f_ij
                force[i,0]+=rx*f
                force[j,0]-=rx*f
                force[i,1]+=ry*f
                force[j,1]-=ry*f
                force[i,2]+=rz*f
                force[j,2]-=rz*f
                
                # obtain vir to calculate pressure
                vir += f * r**2

    return pe, force, vir


# In[5]:


# Nose-hoover Chain Algorithm from Frenkel & Smit's book:
def nhchain(Q, dt, dt_2, dt_4, dt_8, nparticles, vxi, xi, ke, vel):
    G2 = (Q[0]*vxi[0]*vxi[0]-T) 
    vxi[1] += G2*dt_4
    vxi[0] *= math.exp(-vxi[1]*dt_8)
    G1 = (2*ke-3*nparticles*T)/Q[0]   
    vxi[0] += G1*dt_4
    vxi[0] *= math.exp(-vxi[1]*dt_8)
    xi[0]+=vxi[0]*dt_2
    xi[1]+=vxi[1]*dt_2
    s=math.exp(-vxi[0]*dt_2)
    vel=np.multiply(s,vel)
    ke*=(s*s)
    vxi[0]*=math.exp(-vxi[1]*dt_8)
    G1=(2*ke-3*nparticles*T)/Q[0] 
    vxi[0]+=G1*dt_4
    vxi[0]*=math.exp(-vxi[1]*dt_8)
    G2=(Q[0]*vxi[0]*vxi[0]-T)/Q[1]
    vxi[1]+=G2*dt_4

    return ke


# In[6]:


""" parameters """

# cutoff
rc = 8.5 
# time step
dt = 0.001
# total number of steps
nsteps = 10000
# reduced density of the gas
rho=0.8442

# compute parameters
dt_2=0.5*dt
dt_4=0.5*dt_2
dt_8=0.5*dt_4

#frequency to save result
freq = 10

# mass Q and 
Q=[1.0, 1.0]
xi=[0.0, 0.0]
vxi=[0.0, 0.0]

""" Fixed NVT """
# Volumne
V = 1000

# calculate number of particles
nparticles=int(V*rho)

# to make perfecet cube to compute easier, round down it and get the number of particles again
# related to def creat_box
n_3=int(math.floor(nparticles**(1/3)+0.5))
nparticles=n_3**3

# initial temperature (reduced)
T = 1.0

# initialize velocity by def 'init'
vel=init_vel(nparticles, T)

# assume cubic lattice and define the cubic box
# calculate side-length of the box
bx=by=bz=V**(1.0/3)
x_l = y_l = z_l = 0.0
x_u = y_u = z_u = bx

# initialize coordinate by def 'create_box'
coord=create_box(bx, by, bz, nparticles, n_3)

print("side-length : %.3f" % bx)
print("Volume of the system : %.1f" % V)
print("Density of the system : %.4f" % rho)
print("Initial Temperature : %.3f" % T)
print("Number of particles : %i " % nparticles)


# In[7]:


# initialize dataframe to save coordinates
coord_df=pd.DataFrame()
vel_df=pd.DataFrame()
# define ke
ke = 0
result_list = []

# start main iteration MD code
for i in range(nsteps):
    print("Executing NVT MD step {}.".format(i))
  
    # update kinetic energy by Nose-Hoover Chain 
    ke= nhchain(Q, dt, dt_2, dt_4, dt_8, nparticles, vxi, xi, ke, vel)
    
    # first-half of Nose-Hoover Chain Algorithm.
    for j in range(nparticles):
        coord[j,0]+=vel[j,0]*dt_2
        coord[j,1]+=vel[j,1]*dt_2
        coord[j,2]+=vel[j,2]*dt_2
        if coord[j,0] < x_l: 
            coord[j,0] = coord[j,0]+ bx
        if coord[j,0] > x_u:
            coord[j,0]= coord[j,0] - bx
        if coord[j,1] < y_l:
            coord[j,1] = coord[j,1]+ by
        if coord[j,1] > y_u:
            coord[j,1] = coord[j,1]- by
        if coord[j,2] < z_l:
            coord[j,2] = coord[j,2]+ bz
        if coord[j,2] > z_u:
            coord[j,2] = coord[j,2]- bz
            
    # update potential energy, force, virial pressure
    pe, force, vir=LJ_pe_f(rc, bx, by, bz, nparticles, coord)
    
    # Second-half of Nose-Hoover Chain Algorithm.
    for j in range(nparticles):
        vel[j,0]+=dt*force[j,0]  
        vel[j,1]+=dt*force[j,1]
        vel[j,2]+=dt*force[j,2]
        coord[j,0]+=vel[j,0]*dt_2
        coord[j,1]+=vel[j,1]*dt_2
        coord[j,2]+=vel[j,2]*dt_2
        if coord[j,0] < x_l:
            coord[j,0] = coord[j,0]+ bx
        if coord[j,0] > x_u:
            coord[j,0]= coord[j,0] - bx
        if coord[j,1] < y_l:
            coord[j,1] = coord[j,1]+ by
        if coord[j,1] > y_u:
            coord[j,1] = coord[j,1]- by
        if coord[j,2] < z_l:
            coord[j,2] = coord[j,2]+ bz
        if coord[j,2] > z_u:
            coord[j,2] = coord[j,2]- bz

    # calculate kinetic energy from velocity of particles
    ke=0.5*np.square(vel).sum()
    
    # calculate kinetic energy again from Nose-Hoover Chain
    ke= nhchain(Q, dt, dt_2, dt_4, dt_8, nparticles, vxi, xi, ke, vel)
    
    # frequency to save result
    if i%freq == 0:          
        # calculate the variables (total energy, temperature, pressure)
        te = pe + ke
        temperature = ke*2 /(3*nparticles)
        pressure = rho*ke*2 /(3*nparticles) + vir/(3*V)
        
        # initialize a list to save result
        result = [i,pe,ke,te,temperature,pressure]
        result_list.append(result)

        # print all kind of energy and caclulate Temperature,
        print("Nstep: %i , PE = %.2f , KE = %.2f , TE = %.2f , T = %.4f ,  P = %.3f \n" %(i,pe,ke,te,temperature,pressure))


        # make dataframe to save coordiates for each step to make animation
        temp_coord = pd.DataFrame(columns=['index','x','y','z'])
        for j in range(nparticles) :
            index = j
            x,y,z = coord[j,0], coord[j,1], coord[j,2]
            temp_coord = temp_coord.append(pd.DataFrame([[index, x, y, z]], columns=['index','x','y','z']), ignore_index = True)
        temp_coord.set_index('index', inplace=True)
        coord_df = pd.concat([coord_df, temp_coord], axis=1)
        
        # make dataframe to save velocity
        temp_vel = pd.DataFrame(columns=['index','vx','vy','vz'])
        for k in range(nparticles) :
            index2 = k
            vx,vy,vz = vel[k,0], vel[k,1], vel[k,2]
            temp_vel = temp_vel.append(pd.DataFrame([[index2, vx, vy, vz]], columns=['index','vx','vy','vz']), ignore_index = True)
        temp_vel.set_index('index', inplace=True)
        vel_df = pd.concat([vel_df, temp_vel], axis=1)
            

    
# make a directory to save result 
if not os.path.exists('result'):
     os.makedirs('result')   

# save the coordinate and result as *.csv file
coord_df.to_csv('./result/coord_result.csv')
vel_df.to_csv('./result/vel_result.csv')
result_df=pd.DataFrame(result_list,columns=['Step', 'PE', 'KE', 'TE', 'TEMP', 'PRESSURE'])
result_df.to_csv('./result/result.csv')


print("Simulation Completed")


# In[ ]:




