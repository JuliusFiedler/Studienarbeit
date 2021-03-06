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
import time


# %%
# System auswählen
system = 1 # 1 = volterra, 2 = lorenz, 3 = roessler, 4 = wp
# NN auswerten?
NN = False

poly_order = 2
t_end = 3
th = 0.2 # coef threshold
if (system == 1):
    sys_name = "Volterra"
    p_nom = np.array([[0, 1.3, 0, 0, -0.9, 0], [0, 0, -1.8, 0, 0.8, 0]]).T
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y"]
    n = 2
elif (system == 2):
    sys_name = "Lorenz"
    p_nom = np.array(   [[0, -10, 10, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 28, -1, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0]]).T
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y", "z"]
    n = 3
elif (system == 3):
    sys_name = "Roessler"
    p_nom = np.array(   [[0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.1, 0.0, 0.0, -5.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T
    feature_library = ps.PolynomialLibrary(degree=2)
    feature_names=["x", "y", "z"]
    n = 3
    th = 0.05
elif system == 4:
    sys_name = "Wagen-Pendel"
    n = 4
    t_end = 5
    m1 = 3.34
    m2 = 0.8512
    s = 0.26890449
    g = 9.81
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
    # p_nom = np.array([[  0.        ,   0.        ,   1,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
    #    [  0.        ,   0.        ,   0.        ,   1,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
    #    [  0.        ,   0.        ,   0.        ,   0.        , -9.81/0.26890449,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ],
    #    [  0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ,   0.        ]]).T
    p_nom = np.array([ [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   1.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    1.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        , -g*(m1+m2)/ s,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    m2*g      ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    m2*s      ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,   -m2        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ],
                    [   0.        ,    0.        ,    0.        ,    0.        ]])

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
# import csv file and convert to nice datatype
def read_block(name, n):
    path = 'C:\\Users\\Julius\\Documents\\Studium_Elektrotechnik\\Studienarbeit\\github\\Studienarbeit\\Latex\\RST-DiplomMasterStud-Arbeit\\images\\'
    A = pd.read_csv(path + name, sep=',',header=None)
    A = np.array(A.values)
    A = np.delete(A, 0, axis= 0)
    A = np.array(A, dtype=np.float64)
    head = A[0:5, :]
    head = head[head!=0]
    head = np.array(head, dtype=np.int64)
    no_blocks = head.size
    A = np.delete(A, [0,1,2,3,4], axis= 0)
    Xs = []
    DX_s = []
    x_dots = []
    row_start = 0
    row_stop = 0
    for i in range(0, no_blocks):
        row_stop += head[i]
        Xs.append(A[row_start:row_stop, 0:n])
        DX_s.append(A[row_start:row_stop, n:2*n])
        x_dots.append(A[row_start:row_stop, 2*n:3*n])
        row_start = row_stop
    return Xs, DX_s, x_dots
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
def create_poly_library(poly_order = 2):
    x1, x2, x3, x4 = sp.symbols('x1, x2, x3, x4')
    x_list = [x1, x2, x3, x4]
    polys = list()
    variables = x_list[0:n]
    for i in range(1, poly_order+1):
        comb = list(it.combinations_with_replacement(variables, i))
        for j in range(len(comb)):
            monom = comb[j][0]
            for k in range(1, len(comb[j])):
                monom *= comb[j][k]
            polys.append(monom)
    polys.insert(0, 1)
    print(polys)
    print(len(polys))
    return variables, polys
# %%
def calc_param_ident_error(p_no, p_i, supress_print=False):
    if p_no.shape != p_i.shape:
        p_n = np.zeros_like(p_i)
        a, b = p_no.shape
        p_n[0:a, 0:b] = p_no
    else :
        p_n = p_no
 
    s1 = np.sum(((p_n[p_n != 0] - p_i[p_n != 0] ) /p_n[p_n!=0])**2)  
    # s2 = np.sum(((p_n[np.logical_and(p_n == 0, p_i !=0)] - p_i[np.logical_and(p_n == 0, p_i !=0)] ) / p_i[np.logical_and(p_n == 0, p_i !=0)])**2)
    # s2 = np.sum((p_i[np.logical_and(p_n == 0, p_i !=0)])**2)
    s2 = np.sum((p_i[np.logical_and(p_n == 0, abs(p_i) <=1)])**2)
    s3 = np.sum((np.ones_like(p_i)[np.logical_and(p_n == 0, abs(p_i) >1)])**2)

    s = s1 + s2 +s3
    rel_error = round(np.sqrt(s/(p_n[p_n!=0].size)), 5)
    msg = ""
    if p_n[p_n!=0].size != p_i[p_i!=0].size:
        msg = "*"
    #print(s1, s2, s3, s)
    # print(p_n.size)
    if not supress_print:
        print("   RMS relativer Fehler: " + str(rel_error) + msg )
    return rel_error #return str(rel_error) + msg
# %%
def calc_param_ident_error_div_by_ev(p_no, p_i):
    if p_no.shape != p_i.shape:
        p_n = np.zeros_like(p_i)
        a, b = p_no.shape
        p_n[0:a, 0:b] = p_no
    else :
        p_n = p_no
 
    s1 = np.sum(((p_n[p_n != 0] - p_i[p_n != 0] ) /p_n[p_n!=0])**2)  
    s2 = np.sum(((p_n[np.logical_and(p_n == 0, p_i !=0)] - p_i[np.logical_and(p_n == 0, p_i !=0)] ) / p_i[np.logical_and(p_n == 0, p_i !=0)])**2)
    s = s1 + s2
    rel_error = round(np.sqrt(s/(p_n.size)), 5)
    msg = ""
    if p_n[p_n!=0].size != p_i[p_i!=0].size:
        msg = "*"
    # print(s1, s2, s)
    # print(p_n.size)
    print("   RMS relativer Fehler: " + str(rel_error) + msg )
    return str(rel_error) + msg
# %%
# Chaining
def mult(x):
    temp = 1
    for i in range(len(x)):
        temp *= x[i]
    return temp
def chain():
    depth = 4 
    x1 = variables[0]
    x3 = variables[2]
    h = [sp.sin(x1), sp.cos(x1), x3]
    # Zähler bauen
    nominator = []
    nominator = np.append(nominator, 1)
    for i in range(depth):
        comb = list(it.combinations_with_replacement(h, i+1))
        for j in range(len(comb)):
            nominator = np.append(nominator, mult(comb[j]))
    
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
        if eq_ != None:
            print(" = " + str(sp.simplify(eq_)) + "\n")
# %%
def sel_lin_indep(theta, basis):
    is_ = []
    rangabfall = 0
    for i in range(2, theta.shape[1]):
        r_ = np.linalg.matrix_rank(theta[:, 0:i])
        if r_ < i - rangabfall:
            is_ = np.append(is_, i-1)
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
def do_sindys(X, DX_, x_dot, variables, polys, th, dt, p_nom):
    errors = []
    times = []
    
    if dt == -1:
        dt= t_end/(X.shape[0]-1)
    print("PySINDy nominal")
    start_time = time.time()
    model.fit(X, x_dot=DX_, t=dt, multiple_trajectories=False) 
    t1 = time.time() - start_time
    print("time " + str(t1))
    model.print()
    p_ident_nominal = model.coefficients().T
    print(model.coefficients().T)
    err_ps_nom = calc_param_ident_error(p_nom, model.coefficients().T)

    print("PySINDy Zentral")
    start_time = time.time()
    model.fit(X, x_dot=x_dot, t=dt, multiple_trajectories=False) 
    t2 = time.time() - start_time
    print("time " + str(t2))
    model.print()
    p_ident_zentral = model.coefficients().T
    print(model.coefficients().T)
    err_ps_zen = calc_param_ident_error(p_nom, model.coefficients().T)
    
    print("SINDy simple nominal")
    start_time = time.time()
    p_simple_ident_nom = sindy_simple(X, DX_, polys, variables, th)
    t3 = time.time() - start_time
    print("time " + str(t3))
    SIR(p_simple_ident_nom, polys)
    err_ss_nom = calc_param_ident_error(p_nom, p_simple_ident_nom)
    
    print("SINDy simple zentral")
    start_time = time.time()
    p_simple_ident_zen = sindy_simple(X, x_dot, polys, variables, th)
    t4 = time.time() - start_time
    print("time " + str(t4))
    SIR(p_simple_ident_zen, polys)
    err_ss_zen = calc_param_ident_error(p_nom, p_simple_ident_zen)

    errors = np.array([[err_ps_nom, err_ps_zen], [err_ss_nom, err_ss_zen]])
    # errors = np.array([[], [err_ss_nom, err_ss_zen]])
    times = np.array([[t1, t2], [t3, t4]])
    # times = np.array([[], [t3, t4]])
    return errors, times

# %%
# simple systems
variables, polys = create_poly_library(poly_order)
print("SINDy simple nominal")
%time p_simple_ident_nom = sindy_simple(X, DX_, polys, variables, th)
%time SIR(p_simple_ident_nom, polys)
calc_param_ident_error(p_nom, p_simple_ident_nom)
print("SINDy simple zentral")
%time p_simple_ident_zen = sindy_simple(X, x_dot, polys, variables, th)
%time SIR(p_simple_ident_zen, polys)
calc_param_ident_error(p_nom, p_simple_ident_zen)

# %%
# cart-pendulum
basis = chain()
theta = library(basis, variables, X)
# for i in range(1, theta.shape[1]):
#     print(np.linalg.matrix_rank(theta[:, 0:i]))
better_basis = sel_lin_indep(theta, basis)
%time p_simple_ident_nom = sindy_simple(X, DX_, better_basis, variables, 0.2)
%time SIR(p_simple_ident_nom, better_basis)
calc_param_ident_error(p_nom, p_simple_ident_nom)

%time p_simple_ident_zen = sindy_simple(X, x_dot, better_basis, variables, 0.2)
%time SIR(p_simple_ident_zen, better_basis)
calc_param_ident_error(p_nom, p_simple_ident_zen)

# %%
def sim_wp(x,t):
    x = np.zeros(4)
    f = np.array([1,2,3,4], dtype=object)
    for i in range(4):
        f[i] = np.sum((p_simple_ident_zen.T*better_basis)[i])
    f = sp.Subs(f, ["x1","x2","x3","x4"], [x[0], x[1], x[2], x[3]])
    f= f.doit()
    return f

# %%
def sys_wp_nom(u, t):
    r=[ u[2],
        u[3],
        -(g*m1 + g*m2 + m2*u[2]**2*s*np.cos(u[0]))*np.sin(u[0])/(s*(m1 + m2*np.sin(u[0])**2)),
        (m2*(g*np.cos(u[0]) + u[2]**2*s)*np.sin(u[0]))/(m1 + m2*np.sin(u[0])**2)]
    return r
p = p_simple_ident_zen.T[p_simple_ident_zen.T!=0].T
def sys_wp_ident(u,t):
    r=[ p[0]*u[2],
        p[1]* u[3],
        (p[2] + p[3]*u[2]**2*np.cos(u[0]))*np.sin(u[0])/((m1 + m2*np.sin(u[0])**2)),
        ((p[4]*np.cos(u[0]) + u[2]**2*p[5])*np.sin(u[0]))/(m1 + m2*np.sin(u[0])**2)]
    return r
tt = np.arange(0, 10, 0.001)
x_wp_nom=odeint(sys_wp_nom, [-10, 10, 0.5 ,0.5], tt)
x_wp_ident=odeint(sys_wp_ident, [-10, 10, 0.5 ,0.5], tt)
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE) 

fig, axs = plt.subplots(x_wp_nom.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_wp_nom.shape[1]):
    # top = np.max([np.max(x_ori[:,i]), np.max(x_ident[:,i])])

    # bot = np.min([np.min(x_ori[:,i]), np.min(x_ident[:,i])])
    # axs[i].set_ylim(top= top*1.7, bottom = bot)
    axs[i].plot(tt, x_wp_nom[:, i], 'k', label='Originalsystem')
    axs[i].plot(tt, x_wp_ident[:, i], 'r--', label='identifiziertes System')
    axs[i].legend(loc=0)
    axs[i].set(xlabel='t [s]', ylabel='$x_{}$'.format(i+1))
    

fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(7, 9))
colors = ["b", "m--", "g:", "c-."]
for i in range(x_wp_nom.shape[1]):
    axs2[0].plot(tt, np.abs(x_wp_nom[:, i] - x_wp_ident[:, i]), colors[i], label='absoluter Fehler in $x_{}$'.format(i+1))
    axs2[0].legend()
    axs2[0].set(xlabel='t [s]', ylabel="absoluter Fehler")
    
    axs2[1].plot(tt, np.abs(x_wp_nom[:, i] - x_wp_ident[:, i])/x_wp_nom[:, i], colors[i], label='relativer Fehler in $x_{}$'.format(i+1))
    axs2[1].legend()
    axs2[1].set(xlabel='t [s]', ylabel='relativer Fehler')
    axs2[1].set_yscale("log")
    

    

plt.show()


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
optimizer = ps.STLSQ(threshold=th)
feature_library = ps.PolynomialLibrary(degree=2)


# %%
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
    feature_names=feature_names
)
dt= t_end/(X.shape[0]-1)
print("PySINDy nominal")
model.fit(X, x_dot=DX_, t=dt, multiple_trajectories=False) 
%time model.print()
%time p_ident_nominal = model.coefficients().T
calc_param_ident_error(p_nom, p_ident_nominal)

print("PySINDy zentral")
%time model.fit(X, x_dot=x_dot, t=dt, multiple_trajectories=False) 
%time model.print()
%time p_ident_zentral = model.coefficients().T
calc_param_ident_error(p_nom, p_ident_zentral)

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
# import data, set parameters
system = 1 # 1 = volterra, 2 = lorenz, 3 = roessler, 4 = wp
para = "no_tr"
poly_order = 5
dt =  0.1 
t_end = 1
th = 0.2 # coef threshold
if (system == 1):
    sys_name = "Volterra"
    p_nom = np.array([[0, 1.3, 0, 0, -0.9, 0], [0, 0, -1.8, 0, 0.8, 0]]).T
    feature_names=["x", "y"]
    n = 2
elif (system == 2):
    sys_name = "Lorenz"
    p_nom = np.array(   [[0, -10, 10, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 28, -1, 0, 0, 0, -1, 0, 0, 0],
                        [0, 0, 0, -8/3, 0, 1, 0, 0, 0, 0]]).T
    feature_names=["x", "y", "z"]
    n = 3
elif (system == 3):
    sys_name = "Roessler"
    p_nom = np.array(   [[0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [ 0.1, 0.0, 0.0, -5.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T
    feature_names=["x", "y", "z"]
    n = 3
    th = 0.05
model = ps.SINDy(
    differentiation_method = ps.FiniteDifference(order=2),
    feature_library = ps.PolynomialLibrary(degree=poly_order),
    optimizer = ps.STLSQ(threshold=th),
    # optimizer = ps.SR3(threshold=th),
    feature_names=feature_names
)
variables, polys = create_poly_library(poly_order)
Xs, DX_s, x_dots=read_block("Data_"+sys_name+"_"+para+"_variation2.csv", n)

# %%
# calc models, add errors to array
Mask = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
errors = np.zeros((2, 2*len(Xs)))
errors = np.array(errors, dtype=str)
times = np.array(np.zeros((2,1)))
for i in range(len(Xs)):
    e, t = do_sindys(Xs[i], DX_s[i], x_dots[i], variables, polys, th, dt, p_nom)
    errors[:, (2*i):(2*(i+1))] = e 
    times += [[np.sum(t[0])], [np.sum(t[1])]]
print(errors)
av_time = times/(len(Xs)*2)
print(times)

# %%
errors = np.array(errors, dtype = float)
plt.plot(errors[0, np.equal(Mask[0:int((errors.shape[1]))], 1) ], 'k', label = "PySINDy")
plt.plot(errors[1, np.equal(Mask[0:int((errors.shape[1]))], 1) ], 'r--', label = "Simple")
plt.yscale("log")
plt.legend()
plt.show()

# %%
X = read('X_' + sys_name + '.csv')
x_dot = read('x_dot_' + sys_name + '.csv')
DX_ = read('DX__' + sys_name + '.csv')
# %% only for order variation
para = "order"
orders = [2,8]# [ 2, 3, 4, 5 , 6, 7, 8,15]
errors = np.zeros((2, 2*len(orders)))
errors = np.array(errors, dtype=str)
times = np.array(np.zeros((2,1)))
i=0
for ord in orders:
    model = ps.SINDy(
    differentiation_method = ps.FiniteDifference(order=2),
    feature_library = ps.PolynomialLibrary(degree=ord),
    optimizer = ps.STLSQ(threshold=th),
    feature_names=feature_names)
    variables, polys = create_poly_library(ord)
    e, t = do_sindys(X, DX_, x_dot, variables, polys, th, dt, p_nom)
    errors[:, (2*i):(2*(i+1))] = e 
    times += [[np.sum(t[0])], [np.sum(t[1])]]
    i += 1
print(errors)
av_time = times/(len(orders)*2)
print(times)
# %%

if NN:
    print("\nNN Sample interval = 0.1s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_1)

    print("\nNN Sample interval = 0.01s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_01)

    print("\nNN Sample interval = 0.001s :")
    calc_param_ident_error(p_nom, p_ident_NN_0_001)

# %%
def export_to_csv(errors, av_time, file_identifier):
      
    # row 1:
    heads1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]   # parameter values as numbers if possible
    row_head1 = [sys_name]
    for i in range(len(heads1)):
        row_head1.append(heads1[i])
        row_head1.append("")
    row_head1.append("Rechenzeit")
    # row 2:
    heads2 = ["nominal", "zentral"]
    row_head2 = ["Anzahl an Trajektorien"]    # parameter name
    for i in range(len(heads1)):
        row_head2.append(heads2[0])
        row_head2.append(heads2[1])
    row_head2.append("Durchschnitt")
    # row 3:
    row_ps = ['PySINDy']
    for i in range(errors.shape[1]):
        row_ps.append(errors[0][i])
    row_ps.append(round(av_time[0][0], 6))
    #row 4:
    row_ss = ["SINDy simple"]
    for i in range(errors.shape[1]):
        row_ss.append(errors[1][i])
    row_ss.append(round(av_time[1][0], 6))
    
    folder = 'C:\\Users\\Julius\\Documents\\Studium_Elektrotechnik\\Studienarbeit\\github\\Studienarbeit\\Latex\\RST-DiplomMasterStud-Arbeit\\images\\'
    with open(folder + 'errors_' + sys_name + '_' + file_identifier + '.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(row_head1)
        spamwriter.writerow(row_head2)
        spamwriter.writerow(row_ps)
        spamwriter.writerow(row_ss)


# %%
# graphs
def make_graph(name):
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    path = 'C:\\Users\\Julius\\Documents\\Studium_Elektrotechnik\\Studienarbeit\\github\\Studienarbeit\\Latex\\RST-DiplomMasterStud-Arbeit\\images\\'
    A = pd.read_csv(path + name + ".csv", sep=',',header=None)
    A = np.array(A.values) 
    H = np.array(A[0,1:-1], dtype=np.float64)
    Mask = np.logical_not(np.isnan(H)) # nominals 
    H = H[Mask]

    sys = A[0][0]
    x_axes = A[1][0]
    A = np.delete(A, [0, 1], axis= 0)
    A = np.delete(A, 0, axis= 1)
    A = np.array(A, dtype=np.float64)

    Times = A[:, -1]
    A = np.delete(A, -1, axis=1)

    PSn = A[0][Mask]
    PSz = A[0][np.logical_not(Mask)]
    SSn = A[1][Mask]
    SSz = A[1][np.logical_not(Mask)]
    JSn = A[2][Mask]
    JSz = A[2][np.logical_not(Mask)]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(7, 7))
    
    axs[0].plot(H, PSn, 'k', label=r'PySINDy')
    axs[0].plot(H, SSn, 'r--', label=r'SINDy vereinfacht')
    axs[0].plot(H, JSn, 'gD', label=r'SINDy aus DiffEq')
    axs[0].legend()
    axs[0].set(ylabel="relativer Fehler", title='System: '+ sys +', Nominalableitungen')
    axs[0].set_yscale("linear")
    # axs[0].set_xscale("log")

    axs[1].plot(H, PSz, 'k', label=r'PySINDy')
    axs[1].plot(H, SSz, 'r--', label=r'SINDy vereinfacht')
    axs[1].plot(H, JSz, 'gD', label=r'SINDy aus DiffEq')
    axs[1].set(xlabel=str(x_axes), ylabel="relativer Fehler", title='System: '+ sys +', Zentraldifferenz')
    axs[1].set_yscale("log")
    # axs[1].set_xscale("log")

    fig.show()


  
# %%
def time_plot(name, title):
    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    path = 'C:\\Users\\Julius\\Documents\\Studium_Elektrotechnik\\Studienarbeit\\github\\Studienarbeit\\Latex\\RST-DiplomMasterStud-Arbeit\\images\\'
    A = pd.read_csv(path + name + ".csv", sep=',',header=None)
    A = np.array(A.values) 
    H = np.array(A[0,1:-1], dtype=np.float64)
    Mask = np.logical_not(np.isnan(H)) # nominals 
    H = H[Mask]

    sys = A[0][0]
    x_axes = A[1][0]
    A = np.delete(A, [0, 1], axis= 0)
    A = np.delete(A, 0, axis= 1)
    A = np.array(A, dtype=np.float64)

    Times = A[:, -1]
    figure, ax = plt.subplots(figsize=(7, 7))
    x = ["PySINDy", "SINDy\n vereinfacht", "SINDy\n aus DiffEq"]
    y_pos = np.arange(len(x))
    plt.ylabel("Rechenzeit [s]")
    plt.bar(y_pos, Times, align= "center", color= ["k", "r", "g"])
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    # ax.set_yscale("log")
    plt.title("durchschnittliche Rechenzeit\n"+sys+"-System\n"+title)
    figure.show()
    # print(x_axes, sys)

# %%
def simulate(system, p_i, p_nom, dt):
    p1 = p_i.T[p_i.T!=0]
    if system == 1:
        u0 = [0.44249296,4.6280594]
        def sys_ident(x, t):
            dgl1 = [ p1[0] * x[0] + p1[1] * x[0] * x[1],
                     p1[2] * x[1] + p1[3] * x[0] * x[1]]
            return dgl1
        p2 = p_nom.T[p_nom.T!=0]
        def sys(x, t):
            dgl2 = [p2[0] * x[0] + p2[1] * x[0] * x[1],
                    p2[2] * x[1] + p2[3] * x[0] * x[1]]
            return dgl2
    elif system == 2:
        u0 = [-8, 8, 27]
        def sys_ident(x, t):
            dgl1 = [ p1[0] * x[0] + p1[1] * x[1],
                    p1[2] * x[0] + p1[3] * x[1] + p1[4] * x[0] * x[2],
                    p1[5] * x[2] + p1[6] * x[0] * x[1]]
            return dgl1
        p2 = p_nom.T[p_nom.T!=0]
        def sys(x, t):
            dgl2 = [ p2[0] * x[0] + p2[1] * x[1],
                    p2[2] * x[0] + p2[3] * x[1] + p2[4] * x[0] * x[2],
                    p2[5] * x[2] + p2[6] * x[0] * x[1]]
            return dgl2
    elif system == 3:
        u0 = [1, -1, -1]
        def sys_ident(x, t):
            dgl1 = [p1[0]*x[1] + p1[1]* x[2],
                    p1[2]*x[0] + p1[3] * x[1],
                    p1[4] + p1[5]* x[2] + p1[6]*x[0] * x[2] ]
            return dgl1
        p2 = p_nom.T[p_nom.T!=0]
        def sys(x, t):
            dgl2 = [p2[0]*x[1] + p2[1]* x[2],
                    p2[2]*x[0] + p2[3] * x[1],
                    p1[4] + p1[5]* x[2] + p1[6]*x[0] * x[2] ]
            return dgl2
    
    tt = np.arange(0, 15, dt)
    x_ori = odeint(sys, u0, tt)
    x_ident = odeint(sys_ident, u0, tt)

    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE) 
    
    

    fig, axs = plt.subplots(x_ori.shape[1], 1, sharex=True, figsize=(7, 9))
    for i in range(x_ori.shape[1]):
        # top = np.max([np.max(x_ori[:,i]), np.max(x_ident[:,i])])

        # bot = np.min([np.min(x_ori[:,i]), np.min(x_ident[:,i])])
        # axs[i].set_ylim(top= top*1.7, bottom = bot)
        axs[i].plot(tt, x_ori[:, i], 'k', label='Originalsystem')
        axs[i].plot(tt, x_ident[:, i], 'r--', label='identifiziertes System')
        axs[i].legend(loc=0)
        axs[i].set(xlabel='t [s]', ylabel='$x_{}$'.format(i+1))
        

    fig2, axs2 = plt.subplots(2, 1, sharex=True, figsize=(7, 9))
    colors = ["b", "m--", "g:"]
    for i in range(x_ori.shape[1]):
        axs2[0].plot(tt, np.abs(x_ori[:, i] - x_ident[:, i]), colors[i], label='absoluter Fehler in $x_{}$'.format(i+1))
        axs2[0].legend()
        axs2[0].set(xlabel='t [s]', ylabel="absoluter Fehler")
        
        axs2[1].plot(tt, np.abs(x_ori[:, i] - x_ident[:, i])/x_ori[:, i], colors[i], label='relativer Fehler in $x_{}$'.format(i+1))
        axs2[1].legend()
        axs2[1].set(xlabel='t [s]', ylabel='relativer Fehler')
        axs2[1].set_yscale("log")
        
        
    # fig3, axs3 = plt.subplots(x_ori.shape[1], 1, sharex=True, figsize=(7, 9))
    # for i in range(x_ori.shape[1]):
        

    plt.show()

# %%
def wrong_coef(p_nom, rel_err):
    max_i = 1000
    p, e = 0, 0
    n_level = 1
    for i in range(0, max_i):
        p = np.random.normal(size=p_nom.shape) * n_level * (p_nom != 0) + p_nom
        e = calc_param_ident_error(p_nom, p, True)
        if e < rel_err:
            break
        else:
            n_level *= 0.99
    print(e)
    return p, e

# %%
e = 10e-2
np.random.seed(seed=1)
plt.plot(np.random.normal(0,e,100))
plt.show()
np.random.seed(seed=1)
plt.plot(np.random.normal(0,1,100)*e)
plt.show()

# %%
names = ["volterra", "lorenz", "roessler"]
paras = ["tspan", "dt", "order", "no_tr", "noise"]
for na in names:
    for p in paras:
        time_plot("errors_"+na+"_"+p+"_variation", na+"_"+p)
# %%
