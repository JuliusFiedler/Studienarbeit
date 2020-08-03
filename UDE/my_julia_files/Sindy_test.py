# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import csv
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
import debugpy

import pysindy as ps
import pickle
import random
import pandas as pd

# %%
system = 1 # 1 = volterra, 2= lorenz

# %%
def read(name):
    A = pd.read_csv(name, sep=',',header=None)
    A = np.array(A.values)
    A = np.delete(A, 0, axis= 0)
    A = np.array(A, dtype=np.float64)
    for x in A:
        for y in range(len(x)):
            for z in range(len(A)):
                A[z][y] = float(A[z][y])
    print(A.shape)
    t_end = 3
    tt = np.linspace(0, t_end, A.shape[0])
    plt.plot(tt, A)
    return A

# %%
X = read('X.csv')
x_dot = read('x_dot.csv')
DX_ = read('DX_.csv')#
L = read('L.csv')
X_high_res_0_01 = read('X_hr_0_01.csv')
L_high_res_0_01 = read('L_hr_0_01.csv')
X_high_res_0_001 = read('X_hr_0_001.csv')
L_high_res_0_001 = read('L_hr_0_001.csv')
t_end = 3
tt = np.linspace(0, t_end, X.shape[0])
if system == 1:
    p_nom = np.array([[0, 1.3, 0, 0, -0.9, 0], [0, 0, -1.8, 0, 0.8, 0]])


# %%
a = 1.3
c = -1.8

DX_K = np.copy(x_dot)
DX_K[:, 0] = x_dot[:, 0] - a*X[:, 0]    
DX_K[:, 1] = x_dot[:, 1] - c*X[:, 1]  
print(DX_K.shape)
plt.plot(tt, DX_K)

# %%
library_functions = [
    lambda x : x,
    # lambda x : x**3,
    lambda x : np.sin(x),
    lambda x : np.cos(x),
    # lambda x : np.tanh(x)
]
library_function_names = [
    lambda x : x,
    # lambda x : x + "^3",
    lambda x : 'sin(' + x + ')',
    lambda x : 'cos(' + x + ')',
    # lambda x : 'tanh(' + x + ')'

]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)
differentiation_method = ps.FiniteDifference(order=2)
feature_library = ps.PolynomialLibrary(degree=2)
# feature_library = custom_library
optimizer = ps.SR3(threshold=0.2)


# %%
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=["x", "y"]
)
dt= t_end/(X.shape[0]-1)
print("nominal")
model.fit(X, x_dot=DX_, t=dt, multiple_trajectories=False) 
model.print()
p_ident_nominal = model.coefficients()

print("Zentral")
model.fit(X, x_dot=x_dot, t=dt, multiple_trajectories=False) 
model.print()
p_ident_zentral = model.coefficients()

print("NN 0.1s")
model.fit(X, x_dot=L, t=dt, multiple_trajectories=False) 
model.print()
p_ident_NN_0_1 = model.coefficients()

print("NN 0.01s")
model.fit(X_high_res_0_01, x_dot=L_high_res_0_01, t=dt, multiple_trajectories=False) 
model.print()
p_ident_NN_0_01 = model.coefficients()

print("NN 0.001s")
model.fit(X_high_res_0_001, x_dot=L_high_res_0_001, t=dt, multiple_trajectories=False) 
model.print()
p_ident_NN_0_001 = model.coefficients()


# %%
def calc_relative_error(p_n, p_i):
    s = 0
    i = 0
    for x in p_n:
        for y in range(len(x)):
            for z in range(len(p_n)):
                if p_n[z][y]!=0:
                    i += 1
                    s += ((p_n[z][y] - p_i[z][y]) / p_n[z][y])**2
    return np.sqrt(s/i)

# %%
print("RMS relative error Nominalableitung:")
print(calc_relative_error(p_nom, p_ident_nominal))

print("RMS relative error Zentraldifferenz:")
print(calc_relative_error(p_nom, p_ident_zentral))

print("RMS relative error NN Sample interval = 0.1s :")
print(calc_relative_error(p_nom, p_ident_NN_0_1))

print("RMS relative error NN Sample interval = 0.01s :")
print(calc_relative_error(p_nom, p_ident_NN_0_01))

print("RMS relative error NN Sample interval = 0.001s :")
print(calc_relative_error(p_nom, p_ident_NN_0_001))

# %%
t_test = np.linspace(0, 3, X.shape[0])
random.seed(12)
x0_test = [0.44249296,4.6280594]
x_test = X
print('Model score: %f' % model.score(x_test, t=dt))


# %%
# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test)  

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)
#t_test = np.linspace(0, 15, 3000)


fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))

for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i],
                'k', label='numerical derivative')
    axs[i].plot(t_test, x_dot_test_predicted[:, i],
                'r--', label='model prediction')
    axs[i].legend()
    axs[i].set(xlabel='t', ylabel='$\dot x_{}$'.format(i))

fig.show()


# %%
# Evolve the new initial condition in time with the SINDy model

t_evolve = np.linspace(0, 5, 1000)
x_test_sim = model.simulate(x0_test, t_evolve)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], 'k', label='true simulation')
    axs[i].plot(t_evolve, x_test_sim[:, i], 'r--', label='model simulation')
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


