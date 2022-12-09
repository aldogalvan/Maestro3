# Author: Aldo Galvan
# Co-Author: Lourdes Reyes
# HERO Lab / Reneu Lab

import numpy as np
import scipy as sp
import sympy as sym
from scipy.optimize import minimize
import matplotlib as plt
import cmath
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cvxpy as cvxpy
from Maestro3 import Maestro3


# initalize the maestro
M3 = Maestro3()

pi = 3.14

# check for valid regions


# test the forward kinematics
phi3 = np.linspace(pi/2,pi,1000)
psy2 = np.linspace(pi/2,pi,1000)

PIP = pi
MCP = pi
for i in range(1000):
    ret_pip,ret_mcp = M3.forward_kinematics(phi3[i],psy2[i])
    #print(ret_pip,ret_mcp)

#M3.visualizer(phi3,psy2)

"""
output = np.zeros((1000,1000,6),dtype=complex)

for i in range(1000):
    for j in range(1000):
        output[i,j,0],output[i,j,1] = M3.idx.position_KL1(phi3[i])
        output[i,j,2],output[i,j,3] = M3.idx.intermediate_constraints(output[i,j,1])
        output[i,j,4],output[i,j,5] = M3.idx.position_KL2(psy2[j], output[i,j,3])
        
def surface_plot (matrix, **kwargs):
    # acquire the cartesian coordinate matrices from the matrix
    # x is cols, y is rows
    (x, y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, matrix, **kwargs)
    return (fig, ax, surf)


(fig, ax, surf) = surface_plot(output[:,:,5], cmap=plt.cm.coolwarm)

fig.colorbar(surf)

ax.set_xlabel('X (cols)')
ax.set_ylabel('Y (rows)')
ax.set_zlabel('Z (values)')

plt.show()

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

M3.visualizer(phi3,psy2)

# check for valid regions



# mcp appears to be valid for the whole region
for i in range(360):
    PIP[i],MCP[i]= np.real(M3.forward_kinematics(phi3[i], psy2[i]))

# Physics informed neural networks
# first define any symbolic variables for KL1
a1,b1,c1,d1,l1,phi1,phi3,phi4 = sym.symbols('a1,b1,c1,d1,l1,phi1,phi3,phi4')
# next define symbolic variables for KL2
a2,b2,c2,d2,psy1,psy2,psy3,psy4,psys = sym.symbols('a2,b2,c2,d2,psy1,psy2,psy3,psy4,psys')

phi4 = 2*sym.atan((a1 - b1 - ((a1**2 + b1**2 - c1**2 + 2*a1*b1*(2*sym.cos(phi3/2)**2 - 1))/sym.cos(phi3/2)**4)**(1/2)*(sym.cos(phi3)/2 + 1/2) + b1*(sym.cos(phi3) + 1))/(c1 - b1*sym.sin(phi3)))
d1 = ((a1**2 + b1**2 - c1**2 + 2*a1*b1*(2*sym.cos(phi3/2)**2 - 1))/sym.cos(phi3/2)**4)**(1/2)*(sym.cos(phi3)/2 + 1/2)
psy1 = sym.atan(c1**2 / (l1 - d1))
c2 = sym.sqrt(c1**2 + (l1 - d1)**2)
psy3 = -2*sym.atan((((sym.cos(psy2) + 1)*(-(a2**4 + b2**4 + c2**4 + d2**4 - 2*a2**2*b2**2 - 2*a2**2*c2**2 + 2*a2**2*d2**2 - 2*b2**2*c2**2 - 2*b2**2*d2**2 - 2*c2**2*d2**2 + 4*a2**2*d2**2*sym.cos(psy2)**2 + 4*a2*d2**3*sym.cos(psy2) + 4*a2**3*d2*sym.cos(psy2) - 4*a2*b2**2*d2*sym.cos(psy2) - 4*a2*c2**2*d2*sym.cos(psy2))/sym.cos(psy2/2)**4)**(1/2))/2 - 2*a2*b2*sym.sin(psy2))/(a2**2 + 2*sym.cos(psy2)*a2*b2 + 2*sym.cos(psy2)*a2*d2 + b2**2 + 2*b2*d2 - c2**2 + d2**2))
psy4 = 2*sym.atan((((sym.cos(psy2) + 1)*(-(a2**4 + b2**4 + c2**4 + d2**4 - 2*a2**2*b2**2 - 2*a2**2*c2**2 + 2*a2**2*d2**2 - 2*b2**2*c2**2 - 2*b2**2*d2**2 - 2*c2**2*d2**2 + 4*a2**2*d2**2*sym.cos(psy2)**2 + 4*a2*d2**3*sym.cos(psy2) + 4*a2**3*d2*sym.cos(psy2) - 4*a2*b2**2*d2*sym.cos(psy2) - 4*a2*c2**2*d2*sym.cos(psy2))/sym.cos(psy2/2)**4)**(1/2))/2 + 2*a2*c2*sym.sin(psy2))/(a2**2 + 2*sym.cos(psy2)*a2*c2 + 2*sym.cos(psy2)*a2*d2 - b2**2 + c2**2 + 2*c2*d2 + d2**2))

PIP = (psys + psy1 + 2*sym.pi-psy3+psy4 - sym.pi)
MCP = phi4 - sym.pi

X = sym.Matrix([MCP,PIP])

X_jacobian = X.jacobian(sym.Matrix([phi3,psy2]))

print(X_jacobian)


# total length of trajectory
theta_traj_MCP = np.linspace(pi/2,pi,100)
theta_traj_PIP = np.linspace(pi/2,pi,100)
jacobian_det = np.zeros((100,100),dtype = complex)
        
X_jacobian = X_jacobian.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])
X_jacobian = X_jacobian.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])

for idx_MCP in range(100):
    for idx_PIP in range(100):
        jacobian_temp = X_jacobian.subs([(phi3,theta_traj_MCP[idx_MCP]),(psy2,theta_traj_PIP[idx_PIP])])
        jacobian_np = np.array(jacobian_temp).astype(np.cdouble)
        jacobian_det[idx_MCP,idx_PIP] = np.linalg.det(jacobian_np)


# Now we can input a torque and get an output torque
# for a specific trajectory
max_torque_MCP = 1
max_torque_PIP = 1

# length of torque trajectory (s)
t_traj = 10

#number of test points
N = 100
desired_input_torque_MCP = np.linspace(0,max_torque_MCP,N)
desired_input_torque_PIP = np.linspace(0,max_torque_PIP,N)

# measured from the SEA
actual_input_torque_MCP = desired_input_torque_MCP
actual_input_torque_PIP = desired_input_torque_PIP

# get the estimated torque from the static analysis
#for i in range(100):
    

(fig, ax, surf) = surface_plot(jacobian_det, cmap=plt.cm.coolwarm)

fig.colorbar(surf)

ax.set_xlabel('X (cols)')
ax.set_ylabel('Y (rows)')
ax.set_zlabel('Z (values)')

plt.show()

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

""" 

################################################
# MODEL PREDICTIVE CONTROL
# GOD HELP ME
################################################

# first define any symbolic variables for KL1
pi = sym.pi

a1,b1,c1,d1,l1,phi1,phi3,phi4 = sym.symbols('a1,b1,c1,d1,l1,phi1,phi3,phi4')
# next define symbolic variables for KL2
a2,b2,c2,d2,psy1,psy2,psy3,psy4,psys = sym.symbols('a2,b2,c2,d2,psy1,psy2,psy3,psy4,psys')

# forward functions
phi4 = 2*sym.atan((a1 - b1 - ((a1**2 + b1**2 - c1**2 + 2*a1*b1*(2*sym.cos(phi3/2)**2 - 1))/sym.cos(phi3/2)**4)**(1/2)*(sym.cos(phi3)/2 + 1/2) + b1*(sym.cos(phi3) + 1))/(c1 - b1*sym.sin(phi3)))
d1 = ((a1**2 + b1**2 - c1**2 + 2*a1*b1*(2*sym.cos(phi3/2)**2 - 1))/sym.cos(phi3/2)**4)**(1/2)*(sym.cos(phi3)/2 + 1/2)
psy1 = sym.atan(c1**2 / (l1 - d1))
c2 = sym.sqrt(c1**2 + (l1 - d1)**2)
psy3 = -2*sym.atan((((sym.cos(psy2) + 1)*(-(a2**4 + b2**4 + c2**4 + d2**4 - 2*a2**2*b2**2 - 2*a2**2*c2**2 + 2*a2**2*d2**2 - 2*b2**2*c2**2 - 2*b2**2*d2**2 - 2*c2**2*d2**2 + 4*a2**2*d2**2*sym.cos(psy2)**2 + 4*a2*d2**3*sym.cos(psy2) + 4*a2**3*d2*sym.cos(psy2) - 4*a2*b2**2*d2*sym.cos(psy2) - 4*a2*c2**2*d2*sym.cos(psy2))/sym.cos(psy2/2)**4)**(1/2))/2 - 2*a2*b2*sym.sin(psy2))/(a2**2 + 2*sym.cos(psy2)*a2*b2 + 2*sym.cos(psy2)*a2*d2 + b2**2 + 2*b2*d2 - c2**2 + d2**2))
psy4 = 2*sym.atan((((sym.cos(psy2) + 1)*(-(a2**4 + b2**4 + c2**4 + d2**4 - 2*a2**2*b2**2 - 2*a2**2*c2**2 + 2*a2**2*d2**2 - 2*b2**2*c2**2 - 2*b2**2*d2**2 - 2*c2**2*d2**2 + 4*a2**2*d2**2*sym.cos(psy2)**2 + 4*a2*d2**3*sym.cos(psy2) + 4*a2**3*d2*sym.cos(psy2) - 4*a2*b2**2*d2*sym.cos(psy2) - 4*a2*c2**2*d2*sym.cos(psy2))/sym.cos(psy2/2)**4)**(1/2))/2 + 2*a2*c2*sym.sin(psy2))/(a2**2 + 2*sym.cos(psy2)*a2*c2 + 2*sym.cos(psy2)*a2*d2 - b2**2 + c2**2 + 2*c2*d2 + d2**2))
theta_PIP = psy3
theta_MCP = phi4

phi4 = phi4.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])
psy3 = psy3.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])

print(phi4)
print(psy3)


M_jacobian = sym.Matrix([theta_MCP,theta_PIP]).jacobian(sym.Matrix([phi3,psy2]))
M_jacobian = M_jacobian.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])

a1,b1,c1,d1,l1,phi1,phi3,phi4 = sym.symbols('a1,b1,c1,d1,l1,phi1,phi3,phi4')
# next define symbolic variables for KL2
a2,b2,c2,d2,psy1,psy2,psy3,psy4,psys = sym.symbols('a2,b2,c2,d2,psy1,psy2,psy3,psy4,psys')

#inverse functions
phi3 = 2*sym.atan(((-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2) - b1*sym.cos(phi4) + sym.cos(phi4)*(-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2) + sym.sin(phi4)*(-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2) + sym.cos(phi4)*sym.sin(phi4)*(-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/(c1 - a1*sym.sin(phi4) + b1*sym.sin(phi4)))
#phi3  =-2*sym.atan((b1*sym.cos(phi4) + (-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2) + sym.cos(phi4)*(-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2) + sym.sin(phi4)*(-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2) + sym.cos(phi4)*sym.sin(phi4)*(-(a1**2*sym.sin(phi4)**2 - b1**2 + c1**2 - 2*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/(c1 - a1*sym.sin(phi4) + b1*sym.sin(phi4)))
d1 = a1*sym.cos(phi4) + (2**(1/2)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2 + (2**(1/2)*sym.cos(phi4)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2 + (2**(1/2)*sym.sin(phi4)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2 + (2**(1/2)*sym.cos(phi4)*sym.sin(phi4)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2
#d1 = a1*sym.cos(phi4) - (2**(1/2)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2 - (2**(1/2)*sym.cos(phi4)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2 - (2**(1/2)*sym.sin(phi4)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2 - (2**(1/2)*sym.cos(phi4)*sym.sin(phi4)*(-(2*a1**2*sym.sin(phi4)**2 - 2*b1**2 + 2*c1**2 - 4*a1*c1*sym.sin(phi4))/((sym.cos(phi4) + 1)**2*(sym.sin(phi4) + 1)**2))**(1/2))/2
psy1 = sym.atan(c1**2 / (l1 - d1))
c2 = sym.sqrt(c1**2 + (l1 - d1)**2) 
psy2 = -2*sym.atan((((sym.cos(psy3) + 1)*(-(a2**4 + b2**4 + c2**4 + d2**4 - 2*a2**2*b2**2 - 2*a2**2*c2**2 - 2*a2**2*d2**2 - 2*b2**2*c2**2 + 2*b2**2*d2**2 - 2*c2**2*d2**2 + 4*b2**2*d2**2*sym.cos(psy3)**2 - 4*b2*d2**3*sym.cos(psy3) - 4*b2**3*d2*sym.cos(psy3) + 4*a2**2*b2*d2*sym.cos(psy3) + 4*b2*c2**2*d2*sym.cos(psy3))/sym.cos(psy3/2)**4)**(1/2))/2 - 2*a2*b2*sym.sin(psy3))/(a2**2 + 2*sym.cos(psy3)*a2*b2 - 2*a2*d2 + b2**2 - 2*sym.cos(psy3)*b2*d2 - c2**2 + d2**2))
#psy2 = 2*sym.atan((((sym.cos(psy3) + 1)*(-(a2**4 + b2**4 + c2**4 + d2**4 - 2*a2**2*b2**2 - 2*a2**2*c2**2 - 2*a2**2*d2**2 - 2*b2**2*c2**2 + 2*b2**2*d2**2 - 2*c2**2*d2**2 + 4*b2**2*d2**2*sym.cos(psy3)**2 - 4*b2*d2**3*sym.cos(psy3) - 4*b2**3*d2*sym.cos(psy3) + 4*a2**2*b2*d2*sym.cos(psy3) + 4*b2*c2**2*d2*sym.cos(psy3))/sym.cos(psy3/2)**4)**(1/2))/2 - 2*a2*b2*sym.sin(psy3))/(a2**2 + 2*sym.cos(psy3)*a2*b2 - 2*a2*d2 + b2**2 - 2*sym.cos(psy3)*b2*d2 - c2**2 + d2**2))

phi3 = phi3.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])
psy2 = psy2.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])

M_jacobian_inverse = sym.Matrix([phi3,psy2]).jacobian(sym.Matrix([phi4,psy3]))
M_jacobian_inverse = M_jacobian_inverse.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])

#testing jacobian
phi4_op = phi4.subs([(phi3,sym.pi/2),(psy2,2.5)])
print(phi4_op)
psy3_op = psy3.subs([(phi3,sym.pi/2),(psy2,2.5)])
print(psy3_op)
M_jacobian = M_jacobian.subs([(phi3,sym.pi/2),(psy2,2.5)])
M_jacobian_numpy = np.asarray(M_jacobian)
print(M_jacobian_numpy)
torque_out = M_jacobian_numpy.transpose()@np.array([1,1])
print(torque_out)
    
# for input phi3 = pi, psy2 = 2.5
# phi4 = 1.22 , psy3 = -0.34
# for a torque input [1,1] torque out = [2.75,-1.76]
# now we test backwards works
phi3_op = phi3.subs([(phi4,1.22),(psy3,-0.34)])
print(phi3_op)
psy2_op = psy2.subs([(phi4,1.22),(psy3,-0.34)])
print(psy2_op)
M_jacobian_inverse = M_jacobian_inverse.subs([(phi4,1.22),(psy3,-0.34)])
M_jacobian_inverse_numpy = np.asarray(M_jacobian_inverse)
print(M_jacobian_numpy)
torque_out = M_jacobian_inverse_numpy.transpose()@np.array([2.75,-1.76])
print(torque_out)

print(M_jacobian_numpy)
print(M_jacobian_inverse_numpy)
x1_dot,x2_dot,x3_dot,x4_dot,x5_dot,x6_dot,x7_dot,x8_dot = sym.symbols('x1_dot,x2_dot,x3_dot,x4_dot,x5_dot,x6_dot,x7_dot,x8_dot ')
x1,x2,x3,x4,x5,x6,x7,x8 = sym.symbols('x1,x2,x3,x4,x5,x6,x7,x8')
tao_MCP_in,tao_PIP_in = sym.symbols('tao_MCP_in,tao_PIP_in')


x5_bar = theta_MCP.subs([(phi3,x1),(psy2,x2)])
x5_bar = x5_bar.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])
x7_bar = theta_PIP.subs([(phi3,x1),(psy2,x3)])
x7_bar = x7_bar.subs([(a1,40),(b1,75),(c1,12),(a2,41.50),(b2,math.sqrt(15**2+12**2)),(c2,48.0),(psys,math.atan(15/12)),(phi1,50*pi/180),(d2,48.00),(l1,55.0)])
x_jacobian = M_jacobian.subs([(phi3,x1),(psy2,x3)])
x_jacobian_inverse = x_jacobian.inv()
ret = x_jacobian*sym.Matrix([x2,x4])
x6_bar = ret[0]
x8_bar = ret[1]
ret = x_jacobian_inverse*sym.Matrix([x6,x8])
x2_bar = ret[0]
x4_bar = ret[1]
print(x_jacobian_inverse,flush=True)

# check for initial values

#parameters 
b_sea = 10
k_sea = 29.25
k_finger_MCP = 1
b_finger_MCP = 1
k_finger_PIP = 1
b_finger_PIP = 1
m_finger_MCP = 1
m_finger_PIP = 1
r_pulley = 0.025/2
dt = 0.01

"""
#continuous time
x1_dot = x2
x2_dot = -2*k_sea*r_pulley*x1 - b_sea*r_pulley*x2 + sym.Matrix(x_jacobian_inverse@sym.Matrix([x5*k_finger_MCP + x6*b_finger_MCP, x7*k_finger_PIP + x8*b_finger_PIP]))[0] + tao_MCP_in
x3_dot = x4
x4_dot = -2*k_sea*r_pulley*x3 - b_sea*r_pulley*x4 +  sym.Matrix(x_jacobian_inverse@sym.Matrix([x5*k_finger_MCP + x6*b_finger_MCP, x7*k_finger_PIP + x8*b_finger_PIP]))[1]  + tao_PIP_in
x5_dot = x6
x6_dot = -k_finger_MCP*(x5_bar)/m_finger_MCP - b_finger_MCP*(x6_bar)/m_finger_MCP 
x7_dot = x8
x8_dot = -k_finger_PIP*(x7_bar)/m_finger_PIP - b_finger_PIP*(x8_bar)/m_finger_PIP
"""

#discrete time
x1_kplus1 = x1 + x2*dt
x2_kplus1 = x2 + (-k_sea*x1 - b_sea*x2 + (M_jacobian_inverse_numpy.transpose()@sym.Matrix([x5*k_finger_MCP + x6*b_finger_MCP, x7*k_finger_PIP + x8*b_finger_PIP]))[0] + tao_MCP_in)*dt
#x2_kplus1 = x2 + (k_sea*(x1 - x5) + b_sea*(x2 - x6) + tao_MCP_in)*dt
x3_kplus1 = x3 + x4*dt
#x4_kplus1 = x4 + (k_sea*(x3) + b_sea*(x4) + tao_PIP_in)*dt
x4_kplus1 = x4 + (-k_sea*x3 - b_sea*x4 + (M_jacobian_inverse_numpy.transpose()@sym.Matrix([x5*k_finger_MCP + x6*b_finger_MCP, x7*k_finger_PIP + x8*b_finger_PIP]))[1]  + tao_PIP_in)*dt
x5_kplus1 = x5 + x6*dt
x6_kplus1 = x6 + (-(M_jacobian_numpy.transpose()@sym.Matrix([k_sea*(x1) + b_sea*(x2),k_sea*x3 + b_sea*x4]))[0] + k_finger_MCP*x5 + b_finger_MCP*x6)*dt
x7_kplus1 = x7 + x8*dt
x8_kplus1 = x8 + (-(M_jacobian_numpy.transpose()@sym.Matrix([k_sea*(x1) + b_sea*(x2),k_sea*(x3) + b_sea*(x4)]))[1] + k_finger_PIP*x7 + b_finger_PIP*x8 )*dt

print(M_jacobian_inverse_numpy.transpose()@sym.Matrix([x5*k_finger_MCP + x6*b_finger_MCP, x7*k_finger_PIP + x8*b_finger_PIP])[0])
"""
k1 = 10
k2 = 10
b1 = 100
b2 = 100
m1 = 10
m2 = 10
fx2,fx4 = sym.symbols('fx2,fx4')
#double spring damper
x1_kplus1 = x1 + x2*dt
#x2_kplus1 = x2 + (-k_sea*x1 - b_sea*x2 + sym.Matrix(x_jacobian_inverse@sym.Matrix([x5*k_finger_MCP + x6*b_finger_MCP, x7*k_finger_PIP + x8*b_finger_PIP]))[0] + tao_MCP_in)*dt
x2_kplus1 = x2 + (-(k1+k2)/m1*x1 + k2/m1*x3 -(b1+b2)/m1*x2 + b2/m1*x4 + fx2)*dt
x3_kplus1 = x3 + x4*dt
x4_kplus1 = x4 + (k2/m2*x1 - k2/m2*x3 + b2/m2*x2 - b2/m2*x4)*dt
#x4_kplus1 = x4 + (-k_sea*x3 - b_sea*x4 + sym.Matrix(x_jacobian_inverse@sym.Matrix([x5*k_finger_MCP + x6*b_finger_MCP, x7*k_finger_PIP + x8*b_finger_PIP]))[1]  + tao_PIP_in)*dt
x5_kplus1 = x5 + x6*dt
x6_kplus1 = x6 + (-(k1+k2)/m1*x5 + k2/m1*x7 - (b1+b2)/m1*x6 + b2/m1*x8 + fx4)*dt
x7_kplus1 = x7 + x8*dt
x8_kplus1 = x8 + (k2/m2*x5 - k2/m2*x7 + b2/m2*x6 - b2/m2*x8  )*dt
"""
#model
model = sym.Matrix([x1_dot,x2_dot,x3_dot,x4_dot,x5_dot,x6_dot,x7_dot,x8_dot])
model_discrete = sym.Matrix([x1_kplus1,x2_kplus1,x3_kplus1,x4_kplus1,x5_kplus1,x6_kplus1,x7_kplus1,x8_kplus1])
#model_discrete_simplified =  sym.Matrix([x1_kplus1,x2_kplus1,x3_kplus1,x4_kplus1])
x_simplified = sym.Matrix([x1,x2,x3,x4])
x = sym.Matrix([x1,x2,x3,x4,x5,x6,x7,x8])
u = sym.Matrix([tao_MCP_in,tao_PIP_in])
A = model_discrete.jacobian(x)
B = model_discrete.jacobian(u)
#A_simp = model_discrete_simplified.jacobian(x_simplified)
#B_simp = model_discrete_simplified.jacobian(sym.Matrix([fx2]))

A_discrete = model_discrete.jacobian(x)
B_discrete = model_discrete.jacobian(u)

x_init = np.array([0,0,1,0])
x_sim = np.zeros([4,5001])
u_in = np.zeros(1)
u_in[0] = 0 
A_numpy = np.asarray(A)
B_numpy = np.asarray(B)
print(A_numpy)
print(B_numpy)

x_sim[:,0] = x_init
for i in range(1,5001):
    x_sim[:,i] =  A_numpy@x_sim[:,i-1] + B_numpy@u_in

for i in range(8):
    plt.plot(x_sim[i,:])

plt.legend([0,1,2,3,4,5,6,7])


# Model Predictive control
N_max = 100
n = 8
m = 2
T = 100
x_0 = [0,0,0,0,pi/4,0,pi/4,0]
u = np.zeros([2,101])
x = np.zeros([8,101])
x[:,0] = x_0
A_sub = np.array(A_numpy)
B_sub = np.array(B_numpy)


for N in range(1,N_max+1):
    x_temp = cvxpy.Variable((n, T+1))
    u_temp = cvxpy.Variable((m, T))
    x_0 = x[:,N-1]
    cost = 0
    constr = []
    
    for t in range(T):
        cost += 10*cvxpy.sum_squares(x_temp[:, t + 1]) + 0.01*cvxpy.sum_squares(u_temp[:, t])
        constr += [x_temp[:, t + 1] == A_sub @ x_temp[:, t] + B_sub @ u_temp[:, t], cvxpy.norm(u_temp[:, t], "inf") <= 100]
    
    # sums problem objectives and concatenates constraints.
    constr += [x_temp[:, T] == 0, x_temp[:, 0] == x_0]
    problem = cvxpy.Problem(cvxpy.Minimize(cost), constr)
    problem.solve(solver=cvxpy.ECOS)
    print(u_temp[:,0].value)
    u[:,N] = u_temp[:,0].value
    x[:,N] = A_sub@x[:,N-1] + B_sub@u[:,N]

fig, ax = plt.subplots(2)
t = np.linspace(0,1000,100)
for i in range(2):
    ax[0].plot(t,u[i,0:100])
    
ax[0].set_xlabel("Time (ms)")
ax[0].set_ylabel("Torque (Nm)")
ax[0].legend(['Motor MCP','Motor PIP'])
    
for i in range(8):
    ax[1].plot(t,x[i,0:100])
    
ax[1].set_xlabel("Time (ms)")
ax[1].set_ylabel("Deflection (Rad)")
ax[1].legend(['x1','x2','x3','x4','x5','x6','x7','x8'])

fig.suptitle('Passive MCP & PIP Joint Displacement')
fig.show()