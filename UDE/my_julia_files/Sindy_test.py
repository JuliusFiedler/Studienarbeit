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
system = 3 # 1 = volterra, 2 = lorenz, 3 = roessler
NN = True
if (system == 1):
    sys_name = "volterra"
    p_nom = np.array([[0, 1.3, 0, 0, -0.9, 0], [0, 0, -1.8, 0, 0.8, 0]])
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y"]
elif (system == 2):
    sys_name = "lorenz"
    p_nom = np.array(   [[0, -10, 10, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 28, -1, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0]])
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y", "z"]
elif (system == 3):
    sys_name = "roessler"
    p_nom = np.array(   [[0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.1, 0.0, 0.0, -5.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y", "z"]

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
X = read('X_' + sys_name + '.csv')
x_dot = read('x_dot_' + sys_name + '.csv')
DX_ = read('DX__' + sys_name + '.csv')#
if NN:
    L = read('L_' + sys_name + '.csv')
    X_high_res_0_01 = read('X_hr_0_01_' + sys_name + '.csv')
    L_high_res_0_01 = read('L_hr_0_01_' + sys_name + '.csv')
    # X_high_res_0_001 = read('X_hr_0_001_' + sys_name + '.csv')
    # L_high_res_0_001 = read('L_hr_0_001_' + sys_name + '.csv')
t_end = 3
tt = np.linspace(0, t_end, X.shape[0])
    


# %%
# a = 1.3
# c = -1.8

# DX_K = np.copy(x_dot)
# DX_K[:, 0] = x_dot[:, 0] - a*X[:, 0]    
# DX_K[:, 1] = x_dot[:, 1] - c*X[:, 1]  
# print(DX_K.shape)
# plt.plot(tt, DX_K)

# %%
differentiation_method = ps.FiniteDifference(order=2)
optimizer = ps.SR3(threshold=0.05)


# %%
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=feature_names
)
dt= t_end/(X.shape[0]-1)
print("nominal")
model.fit(X, x_dot=DX_, t=dt, multiple_trajectories=False) 
model.print()
p_ident_nominal = model.coefficients()
# print(p_ident_nominal)

print("Zentral")
model.fit(X, x_dot=x_dot, t=dt, multiple_trajectories=False) 
model.print()
p_ident_zentral = model.coefficients()

if NN:
    print("NN 0.1s")
    model.fit(X, x_dot=L, t=dt, multiple_trajectories=False) 
    model.print()
    p_ident_NN_0_1 = model.coefficients()

    print("NN 0.01s")
    model.fit(X_high_res_0_01, x_dot=L_high_res_0_01, t=dt, multiple_trajectories=False) 
    model.print()
    p_ident_NN_0_01 = model.coefficients()

    # print("NN 0.001s")
    # model.fit(X_high_res_0_001, x_dot=L_high_res_0_001, t=dt, multiple_trajectories=False) 
    # model.print()
    # p_ident_NN_0_001 = model.coefficients()


# %%
def calc_param_ident_error(p_n, p_i):
    s = 0
    i = 0
    k = 0
    t = 0
    msg = ""
    for x in p_n:
        for y in range(len(x)):
            for z in range(len(p_n)):
                if p_n[z][y]!=0:
                    i += 1
                    s += ((p_n[z][y] - p_i[z][y]) / p_n[z][y])**2
                elif p_i[z][y] != 0:
                    msg = "*"
                    k += 1
                else:
                    k += 1
                t += (p_n[z][y] - p_i[z][y])**2
    rel_error = round(np.sqrt(s/i), 5)
    abs_error = round(np.sqrt(t/(i+k)), 5)
    print("   RMS absoluter Fehler: " + str(abs_error))
    print("   RMS relativer Fehler: " + str(rel_error) + msg)
    return [str(abs_error), str(rel_error) + msg]

# %%
print("\nNominalableitung:")
calc_param_ident_error(p_nom, p_ident_nominal)

print("\nZentraldifferenz:")
calc_param_ident_error(p_nom, p_ident_zentral)

if NN:
    print("\nNN Sample interval = 0.1s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_1)

    print("\nNN Sample interval = 0.01s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_01)

    # print("\nNN Sample interval = 0.001s :")
    # calc_param_ident_error(p_nom, p_ident_NN_0_001)

# %%
def export_to_csv(l=5):
    p_ident = [p_ident_nominal, p_ident_zentral, p_ident_NN_0_1, p_ident_NN_0_01, p_ident_NN_0_001]
    heads = ["nominal", "zentral", "NN", "NN x10", "NN x100"]
    row_head = [sys_name]
    row_abs = ['PySINDy abs. Fehler']
    row_rel = ['PySINDy rel. Fehler']
    for i in range(l):
        row_head.append(heads[i])
        row_abs.append(calc_param_ident_error(p_nom, p_ident[i])[0])
        row_rel.append(calc_param_ident_error(p_nom, p_ident[i])[1])

    folder = 'C:\\Users\\Julius\\Documents\\Studium_Elektrotechnik\\Studienarbeit\\github\\Studienarbeit\\Latex\\RST-DiplomMasterStud-Arbeit\\images\\'
    with open(folder + 'errors_' + sys_name + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(row_head)
        spamwriter.writerow(row_abs)
        spamwriter.writerow(row_rel)




# %%
