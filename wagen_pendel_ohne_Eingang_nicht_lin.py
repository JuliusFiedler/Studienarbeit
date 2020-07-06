# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Wagen Pendel Systemidentification mit SINDy
# ##  nicht-linearisierte DGL
# ##  autonomes System 
# 

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
import pysindy as ps

from pysindy.feature_library import FourierLibrary, CustomLibrary
from pysindy.feature_library import ConcatLibrary


import pickle
import random

# %% [markdown]
# ### Systemgleichungen
# 

# %%
system_read = pickle.load(open("wagen_pendel_f.p","rb"))
get_ipython().run_line_magic('time', 'rhs = system_read.create_simfunction()')

# %% [markdown]
# ### Trainingsdaten generieren

# %%
tt = np.linspace(0, 5, 1000)
data_sequence = []
for i in range(0,10):
    random.seed(i)
    xx0=[random.uniform(-3,3),random.uniform(-3,3),random.uniform(-3,3),random.uniform(-3,3)]
    
    data_set = odeint(rhs, xx0, tt) 
    data_sequence.append(data_set)

# print(data_sequence)

# %% [markdown]
# ### Modellparameter einstellen

# %%
library_functions = [
    lambda x : 1./x,
    lambda x : x,
    lambda x : np.sin(x),
    lambda x : np.cos(x),
    lambda x : 1
  ]


verkettung =True

def factory_1(i,k):
    return lambda x,y : library_functions[i](x) * library_functions[k](y)
def factory_2(i,k):
    return lambda x,y,z , i=i,k=k: library_functions[i](x) * library_functions[k](y,z)

if verkettung: #Multiplikative Verkettung von Elementarfunktionen bis zu gewünschter Tiefe
    library_functions.pop() # Absolutglied entfernen, später wieder zur Liste hinzufügen
    depth = 2
    lengths = []
    lengths.append(0)
    original_size=len(library_functions)
    for d in range (0,depth):
        lengths.append(len(library_functions))
        for i in range (0,original_size):
            for k in range (lengths[d],lengths[d+1]):
                if d==0:
                    # print(i,k)
                    #func = lambda x,y, i=i,k=k : library_function_names[i](x) +'*'+ library_function_names[k](y)
                    func = factory_1(i,k)
                    library_functions.append(func)
                    # print(library_function_names[-1]('a','b'))

                elif d==1:
                    # library_function_names.append(lambda x,y,z , i=i,k=k: library_function_names[i](x) +'*'+ library_function_names[k](y,z))
                    func = factory_2(i,k)
                    library_functions.append(func)
    print(len(library_functions))   
        # print(library_functions) 
    library_functions.append(lambda x : 1)


def factory_1(i,k):
    return lambda x,y : library_function_names[i](x) +'*'+ library_function_names[k](y)
def factory_2(i,k):
    return lambda x,y,z , i=i,k=k: library_function_names[i](x) +'*'+ library_function_names[k](y,z)

library_function_names = [
    lambda x : '1/' + x,
    lambda x : x,
    lambda x : 'sin(' + x + ')',
    lambda x : 'cos(' + x + ')',
    lambda x : ''     
]

if verkettung: #Multiplikative Verkettung von Elementarfunktionen bis zu gewünschter Tiefe
    library_function_names.pop() # Absolutglied entfernen, später wieder zur Liste hinzufügen
    lengths = []
    lengths.append(0)
    original_size=len(library_function_names)
    for d in range (0,depth):
        lengths.append(len(library_function_names))
        for i in range (0,original_size):
            for k in range (lengths[d],lengths[d+1]):
                if d==0:
                    print(i,k)
                    #func = lambda x,y, i=i,k=k : library_function_names[i](x) +'*'+ library_function_names[k](y)
                    func = factory_1(i,k)
                    library_function_names.append(func)
                    # print(library_function_names[-1]('a','b'))

                elif d==1:
                    # library_function_names.append(lambda x,y,z , i=i,k=k: library_function_names[i](x) +'*'+ library_function_names[k](y,z))
                    func = factory_2(i,k)
                    library_function_names.append(func)
    print(len(library_function_names))   
        # print(library_function_names) 
    library_function_names.append(lambda x : '')


custom_library = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)
differentiation_method = ps.FiniteDifference(order=2)
# feature_library = ps.PolynomialLibrary(degree=5)
# feature_library = ps.FourierLibrary() 
feature_library = custom_library
# feature_library = ps.IdentityLibrary()


optimizer = ps.STLSQ(threshold=0.1)
# optimizer = ps.SR3(threshold=0.1, nu=1)
# optimizer = Lasso(alpha=100, fit_intercept=False)

# %% [markdown]
# ### fit model

# %%
x_train = data_sequence


model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=["phi", "x", "phidot","xdot"]
)

dt= 0.005
model.fit(x_train, t=dt, multiple_trajectories=True)
model.print()

# %% [markdown]
# ### Vergleichsmodell ply=5, th=.1
# phi' = 1.001 phidot  
# x' = 1.001 xdot  
# phidot' = -36.374 phi + -4.181 phi xdot + 5.626 phi^3 + -0.123 phi phidot^2 + -0.355 phi xdot^2 + 1.466 phi^3 xdot + 0.295 phi xdot^3 + 0.230 phi^3 xdot^2  
# xdot' = 4.518 phi + 0.597 phi xdot + -1.496 phi^3 + -0.199 phi xdot^2 + -0.525 phi^3 xdot

# %%
t_test = np.linspace(0, 5, 1000)
x0_test = [1,-1,0.1,1]
x_test = odeint(rhs, x0_test, t_test)
# print(x_test)


# %%
# # Compare SINDy-predicted derivatives with finite difference derivatives
print('Model score: %f' % model.score(x_test, t=dt))

# %% [markdown]
# ### Taylor Simulation

# %%
system_read = pickle.load(open("wagen_pendel_f_taylor.p","rb"))
get_ipython().run_line_magic('time', 'rhs = system_read.create_simfunction()')

x_taylor_test = odeint(rhs, x0_test, t_test)


# %%
# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)
x_dot_taylor_test_computed = model.differentiate(x_taylor_test, t=dt)
#t_test = np.linspace(0, 15, 3000)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i],
                'k', label='numerical derivative')
    axs[i].plot(t_test, x_dot_test_predicted[:, i],
                'r--', label='model prediction')
    # taylor
    axs[i].plot(t_test, x_dot_taylor_test_computed[:, i],
                'g-.', label='numerical derivative taylor')

    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$\dot x_{}$'.format(i))
fig.show()


# %%
# Evolve the new initial condition in time with the SINDy model
#x0_test = [1,-1,0,1]
t_evolve = np.linspace(0, 10, 2000)
x_test_sim = model.simulate(x0_test, t_evolve)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_evolve, x_test_sim[:, i], 'r--', label='model simulation')
    # taylor
    axs[i].plot(t_test, x_taylor_test[:, i], 'g-.', label='taylor simulation')

    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$x_{}$'.format(i))

fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(x_test[:, 0], x_test[:, 1],  'k')
ax1.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='true simulation')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], 'r--')
ax2.set(xlabel='$x_0$', ylabel='$x_1$',
        zlabel='$x_2$', title='model simulation')

fig.show()


# %%
g = 9.81
s2 = 0.26890449

system_read = pickle.load(open("wagen_pendel_f.p","rb"))
get_ipython().run_line_magic('time', 'rhs = system_read.create_simfunction()')
t_ori_model = np.linspace(0, 5, 1000)
x0_ori_model = [1,1,0,0]

ori_model_data = odeint(rhs, x0_ori_model, t_ori_model)
print(ori_model_data)



plt.plot(t_ori_model, ori_model_data[:,0], color= 'red', label="phi")
plt.plot(t_ori_model, ori_model_data[:,1], color= 'blue', label="x")
plt.plot(t_ori_model, ori_model_data[:,2], color= 'black', linestyle='dashed', label="phidot")
plt.plot(t_ori_model, ori_model_data[:,3], color= 'green', linestyle='dashed', label="xdot")
plt.xlabel('Zeit [s]')
plt.ylabel('Amplitude')

plt.grid()
plt.legend()
plt.show()


# %%


