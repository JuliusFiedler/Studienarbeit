# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import csv
import numpy as np 
import sympy as sp
import itertools as it
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
# System auswählen
system = 2 # 1 = volterra, 2 = lorenz, 3 = roessler, 4 = wp
# NN auswerten?
NN = False


t_end = 3
th = 0.2 # coef threshold
if (system == 1):
    sys_name = "volterra"
    p_nom = np.array([[0, 1.3, 0, 0, -0.9, 0], [0, 0, -1.8, 0, 0.8, 0]])
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y"]
    n = 2
elif (system == 2):
    sys_name = "lorenz"
    p_nom = np.array(   [[0, -10, 10, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 28, -1, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0]])
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y", "z"]
    n = 3
elif (system == 3):
    sys_name = "roessler"
    p_nom = np.array(   [[0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.1, 0.0, 0.0, -5.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y", "z"]
    n = 3
    th = 0.05
elif system == 4:
    sys_name = "wp"
    n = 4
    t_end = 5
    m1 = 3.34
    m2 = 0.8512
    library_functions = [
        lambda x : x,
        lambda x : np.sin(x),
        lambda x : np.cos(x)    ]       
    library_function_names = [
        lambda x : x,
        lambda x : 'sin(' + x + ')',
        lambda x : 'cos(' + x + ')' ]
    custom_library = ps.CustomLibrary(library_functions=library_functions, function_names=library_function_names)
    feature_library = custom_library
    feature_names=["phi", "x", "dphi","dx"]
    p_nom = np.array([[  0.        ,   0.        ,   1,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ,   1,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ,   0.        , -9.81/0.26890449,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ]])

# %%
# import csv file and convert to nice datatype
def read(name):
    A = pd.read_csv(name, sep=',',header=None)
    A = np.array(A.values)
    A = np.delete(A, 0, axis= 0)
    A = np.array(A, dtype=np.float64)
    # for x in A:
    #     for y in range(len(x)):
    #         for z in range(len(A)):
    #             A[z][y] = float(A[z][y])
    print(A.shape)
    
    # tt = np.linspace(0, t_end, A.shape[0])
    # plt.plot(tt, A)
    return A

# %%
X = read('X_' + sys_name + '.csv')
x_dot = read('x_dot_' + sys_name + '.csv')
DX_ = read('DX__' + sys_name + '.csv')#
if NN:
    L = read('L_' + sys_name + '.csv')
    X_high_res_0_01 = read('X_hr_0_01_' + sys_name + '.csv')
    L_high_res_0_01 = read('L_hr_0_01_' + sys_name + '.csv')
    X_high_res_0_001 = read('X_hr_0_001_' + sys_name + '.csv')
    L_high_res_0_001 = read('L_hr_0_001_' + sys_name + '.csv')
# t_end = 3
tt = np.linspace(0, t_end, X.shape[0])

# %%


poly_oder = 2
x1, x2, x3, x4 = sp.symbols('x1, x2, x3, x4')
x_list = [x1, x2, x3, x4]
polys = list()
variables = x_list[0:n]
for i in range(1, poly_oder+1):
    comb = list(it.combinations_with_replacement(variables, i))
    for j in range(len(comb)):
        monom = comb[j][0]
        for k in range(1, len(comb[j])):
            monom *= comb[j][k]
        polys.append(monom)
polys.insert(0, 1)
print(polys)
print(len(polys))
# %%
def calc_param_ident_error(p_n, p_i):
    assert p_n.shape == p_i.shape
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
                    s += ((p_n[z][y] - p_i[z][y]) / p_i[z][y])**2
                    msg = "*"
                    k += 1
                else:
                    k += 1
                t += (p_n[z][y] - p_i[z][y])**2
    rel_error = round(np.sqrt(s/(p_n.size)), 9)
    # abs_error = round(np.sqrt(t/(i+k)), 5)
    # print("   RMS absoluter Fehler: " + str(abs_error))
    print(s)
    print(p_n.size)
    print("   RMS relativer Fehler: " + str(rel_error) + msg)
    return [ str(rel_error) + msg] #str(abs_error),

# %%
# Chaining
def mult(x):
    temp = 1
    for i in range(len(x)):
        temp *= x[i]
    return temp
def verkettung():
    depth = 4 
    h = [sp.sin(x1), sp.cos(x1), x3]
    # Zähler bauen
    nominator = []
    for i in range(depth):
        comb = list(it.combinations_with_replacement(h, i+1))
        for j in range(len(comb)):
            nominator = np.append(nominator, mult(comb[j]))
    nominator = np.append(nominator, 1)
    # Bruch bauen
    denom = m1+m2*sp.sin(x1)**2
    basis = np.copy(variables)
    for z_i in nominator:
        basis = np.append(basis, z_i/denom)
    
    return basis

# %%
def library(h, u, X):
    f = sp.lambdify(u, h)
    if n == 2:
        l = f(X[:, 0], X[:, 1])
    elif n == 3:
        l = f(X[:, 0], X[:, 1], X[:, 2])
    elif n == 4:
        l = f(X[:, 0], X[:, 1], X[:, 2], X[:, 3])
    lib = np.ones((X.shape[0], len(l)))
    for i in range(len(l)):
        lib[:, i] = l[i]
    return lib
# %%
def make_sparse(Xi, lam=0.2):
    was_sparse_already = True
    for i in range(Xi.shape[0]):
            for j in range(Xi.shape[1]):
                if abs(Xi[i][j]) < lam:
                    if Xi[i][j] != 0:
                        was_sparse_already = False
                    Xi[i][j] = 0                    
    return Xi, was_sparse_already
# %%
def make_sparse_2(Xi, lam=0.2):
    A = np.copy(Xi)
    A[abs(Xi)<lam] = 0
    was_sparse_already = not (A-Xi).any()                
    return A, was_sparse_already
# %%
def calc_Xi_select(Xi, theta, DX):
    Xi_select = []
    for i in range(Xi.shape[1]):
        a = np.array([], dtype=int)
        for j in range(Xi.shape[0]):
            if Xi[j][i] != 0:
                a = np.append(a, j)
        theta_select = theta[:, a]
        DX_select = DX[:, i]
        Xi_i = np.linalg.lstsq(theta_select, DX_select, rcond=None)[0]
        Xi_select = np.append(Xi_select, Xi_i)
    return Xi_select
# %%
def calc_Xi_select_2(Xi, theta, DX):
    Xi_select = []
    for i in range(Xi.shape[1]):
        theta_select = theta[:, np.nonzero(Xi[:, i])[0]]
        DX_select = DX[:, i]
        Xi_i = np.linalg.lstsq(theta_select, DX_select, rcond=None)[0]
        Xi_select = np.append(Xi_select, Xi_i)
    return Xi_select
# %%
def reshape_Xi(Xi, Xi_select):
    count = 0
    for i in range(Xi.shape[1]):
        for j in range(Xi.shape[0]):
            if Xi[j][i] != 0:
                Xi[j, i] = Xi_select[count]
                count += 1
    return Xi
# %%
def reshape_Xi_2(Xi, Xi_select):
    indices = np.argwhere(Xi.T)
    indices[:, [0, 1]] = indices[:, [1, 0]]
    for i in range(indices.shape[0]):
        Xi[indices[i, 0], indices[i, 1]] = Xi_select[i]
    return Xi
# %%
def SIR(Xi, h):
    for i in range(Xi.shape[1]):
        eq_ = None
        for j in range(Xi.shape[0]):
            if Xi[j][i] != 0:
                if eq_ == None:
                    eq_ = round(Xi[j][i], 5) * h[j]
                else:
                    eq_ += round(Xi[j][i], 5) * h[j]
        print("f" + str(i+1) + " = " + str(eq_))
        print(" = " + str(sp.simplify(eq_)) + "\n")
# %%
def sel_lin_indep(theta, basis):
    is_ = []
    rangabfall = 0
    for i in range(1, theta.shape[1]):
        r_ = np.linalg.matrix_rank(theta[:, 0:i])
        if r_ < i+1 - rangabfall:
            is_ = np.append(is_, i)
            rangabfall += 1
    basis_ = np.copy(basis)
    basis_ = np.delete(basis_, is_)
    return basis_
# %%
def sindy_simple(X, DX, h, u, lam=0.2, maxiter=30):
    theta = library(h, u, X)
    
    # initial guess
    Xi = np.linalg.lstsq(theta, DX, rcond=None)[0]
    # set small coef to 0
    Xi, _ = make_sparse_2(Xi, lam)
    # iterate
    iters = 0
    for i in range(maxiter):
        Xi_select = calc_Xi_select_2(Xi, theta, DX)
        Xi = reshape_Xi_2(Xi, Xi_select)
        Xi, wsa = make_sparse_2(Xi, lam)
        iters += 1
        if wsa:
            break
    print("converged after " + str(iters) + " iterations")
    # SIR(Xi, h)
    return Xi

# %%
# simple systems
%time p_simple_ident_nom = sindy_simple(X, DX_, polys, variables, th)
%time SIR(p_simple_ident_nom, polys)
calc_param_ident_error(p_nom, p_simple_ident_nom.T)
%time p_simple_ident_zen = sindy_simple(X, x_dot, polys, variables, th)
%time SIR(p_simple_ident_zen, polys)
calc_param_ident_error(p_nom, p_simple_ident_zen.T)
# %%
# cart-pendulum
basis = verkettung()
theta = library(basis, variables, X)
# for i in range(1, theta.shape[1]):
#     print(np.linalg.matrix_rank(theta[:, 0:i]))
better_basis = sel_lin_indep(theta, basis)
%time p_simple_ident = sindy_simple(X, DX_, basis, variables, 0.5)
%time SIR(p_simple_ident_nom, basis)

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
optimizer = ps.SR3(threshold=th)
# %%
feature_library = ps.PolynomialLibrary(degree=5)

# %%
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=feature_names
)
dt= t_end/(X.shape[0]-1)
print("nominal")
%time model.fit(X, x_dot=DX_, t=dt, multiple_trajectories=False) 
%time model.print()
%time p_ident_nominal = model.coefficients()
# print(p_ident_nominal)

print("Zentral")
%time model.fit(X, x_dot=x_dot, t=dt, multiple_trajectories=False) 
%time model.print()
%time p_ident_zentral = model.coefficients()

if NN:
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
print("\nNominalableitung:")
calc_param_ident_error(p_nom, p_ident_nominal)

print("\nZentraldifferenz:")
calc_param_ident_error(p_nom, p_ident_zentral)

if NN:
    print("\nNN Sample interval = 0.1s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_1)

    print("\nNN Sample interval = 0.01s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_01)

    print("\nNN Sample interval = 0.001s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_001)

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
