__author__ = 'libochen'

import numpy as np
from numpy import linalg as la
import os
from numpy import matlib
import matplotlib.pyplot as plt
import random


# ###################### read data from file #####################
train_file = './logistic_regression/logistic_digits_train.txt'
test_file = './logistic_regression/logistic_digits_test.txt'


def read_file(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    data_new = []
    line_num = 0
    y_pre = []
    for line in lines:
        if line_num > 0:
            line = line.rstrip(os.linesep)
            string_list = line.split(",")
            num_list = []
            for string in string_list:
                num_list.append(int(string))
            y_pre.append(num_list[-1])
            data_new.append(num_list)
            feature_num = len(num_list) - 1
        line_num += 1
    sample_num = len(data_new)

    Y = np.zeros((1, sample_num))
    X = np.zeros((feature_num, sample_num))
    for i in range(sample_num):
        Y[0, i] = data_new[i][feature_num]
        X[:, i] = np.array(data_new[i][0:feature_num])
    Y += (1 - min(y_pre))
    return X, Y

X, Y = read_file(train_file)
X_t, Y_t = read_file(test_file)
feature_num, sample_num = X.shape
print "feature number is, ", feature_num


def matrix_to_list(M, row):
    res = []
    m, n = M.shape
    for i in range(row):
        res.append(M[(i*m/row):((i+1)*m/row), 0:n])
    return res


def soft_max(X, y, B_in):
    m, n = X.shape
    m_B, n_B = B_in.shape
    B_new = np.reshape(B_in, (class_num, m_B/class_num))
    B_new = B_new.T
    E_new = - np.dot(B_new.T, X)
    E_new = np.exp(E_new)
    E_new_sum = np.sum(E_new, axis=0)
    E_new_sum = np.matlib.repmat(E_new_sum, class_num, 1)
    # print "e new sum shape", E_new_sum.shape
    s = np.divide(E_new, E_new_sum)
    # print "s shape is ", s.shape
    return s


def calc_error(X, y, B_in):
    m,n = X.shape
    nvar = n    # number of samples
    m_B, n_B = B_in.shape
    ncat = m_B/m;
    B_new = matrix_to_list(B_in, ncat)
    y_pred = np.zeros((1,nvar))
    bx = np.zeros((ncat,1))

    for j in range(nvar):
        for i in range(ncat):
            bx[i] = np.dot(B_new[i].T, X[:,j])
        Idx = np.unravel_index(bx.argmin(), bx.shape)
        y_pred[0, j] = Idx[0] + 1
    correct = np.where(y_pred == y)
    cor_len = len(correct[1])
    accuracy = float(cor_len)/float(nvar)
    err = 1 - accuracy

    return err


# input y is 1*N
def log_func(X, Y, B, u):
    feature_num, sample_num = X.shape
    length, dummy = B.shape

    # B1 C * n (feature number)
    B1 = np.reshape(B, (class_num, length / class_num))
    multi_sum = -np.dot(B1, X)
    exp_sum = np.exp(multi_sum)
    partial_sum = np.sum(exp_sum, axis=0)

    log_sum = np.sum(np.log(partial_sum))
    normal_sum = np.zeros((1, sample_num))
    b_norm = (la.norm(B, 2))**2

    b_list = matrix_to_list(B, class_num)
    for i in range(sample_num):
        normal_sum[0, i] = np.dot(b_list[int(Y[0, i] - 1)].T, X[:, i])
    res = (1.0 / sample_num) * (np.sum(normal_sum) + log_sum) + u * b_norm
    return res


def log_func_gradient(X, Y, B, S, u):
    feature_num, sample_num = X.shape
    x_sum = np.zeros((feature_num, class_num))
    for i in range(class_num):
        row, col = np.where(Y == (i+1))
        x_sum[:, i] = np.sum(X[:, col], axis=1)
    res = np.zeros((feature_num * class_num, 1))
    for i in range(class_num):
        res[(i * feature_num):((i+1) * feature_num), 0] = x_sum[:, i]
        S_new = np.dot(X, np.diag(S[i, :]))
        res[(i * feature_num):((i+1) * feature_num), 0] = \
            res[i * feature_num:(i+1) * feature_num, 0] - np.sum(S_new, axis=1)

    res = (1.0/sample_num)*res + 2 * u * B
    return res


def i_sample_gradient(X, y, B_in, S, mu, i):
    m, n = X.shape
    ymin = np.amin(y)
    ymax = np.amax(y)
    ncat = int(ymax - ymin) + 1    # number of classes
    Bmat = B_in

    res = np.zeros((m*ncat, 1))
    for j in range(ncat):
        if y[0, i] == j+1:
            res[(j*m):((j+1)*m), 0] = X[:, i]
        S_new = np.dot(X[:, i], S[j, i])
        res[(j*m):((j+1)*m), 0] = res[j*m:(j+1)*m, 0] - S_new

    res = res + 2*mu*Bmat;
    return res



# ########################### GD Method ##############################
f_GD = []
err_GD = []


def GD():
    feature_num, sample_num = X.shape
    B = 0.1 * np.ones((feature_num * class_num, 1))
    u = 0.2
    tol = 1e-10
    S = soft_max(X, Y, B)
    g = log_func_gradient(X, Y, B, S, u)
    print "g is ", g
    k = 1
    eta = 0.001
    max_iteration = max_iteration_config['GD']
    B += eta * (-g)
    print "B is ", B
    log_func(X, Y, B, u)
    while (la.norm(g))**2 > tol and k < max_iteration:
        B += eta * (-g)
        f = log_func(X, Y, B, u)
        print "k = %d: l(B) = %f" % (k, f), "in GD"
        f_GD.append(f)
        S = soft_max(X, Y, B)
        g = log_func_gradient(X, Y, B, S, u)
        k += 1
        err_GD.append(calc_error(X_t, Y_t, B))
        print "Prediction error is: %f" % calc_error(X_t, Y_t, B)

    print log_func(X_t, Y_t, B, u)


f_SGD = []
err_SGD = []
def SGD():
    feature_num, sample_num = X.shape
    B = 0.1 * np.ones((feature_num * class_num, 1))
    mu = 0.2
    max_iteration = max_iteration_config['SGD']
    tol = 1e-1
    S = soft_max(X, Y, B)
    i = random.randint(0, sample_num-1)
    g = i_sample_gradient(X, Y, B, S, mu, i)
    k = 1
    eta = 0.005

    while (la.norm(g))**2 > tol and k < max_iteration:
        B += eta * (-g)
        f = log_func(X, Y, B, mu)
        print "k = %d: l(B) = %f" % (k,f), "in SGD"
        f_SGD.append(f)
        S = soft_max(X, Y, B)
        print "s shape is ", S.shape
        i = random.randint(0, sample_num-1)
        g = i_sample_gradient(X, Y, B, S, mu, i)
        k += 1

        err_SGD.append(calc_error(X_t, Y_t, B))
        print "Prediction error is: %f" % calc_error(X_t, Y_t, B)

    print log_func(X_t, Y_t, B, mu)


def SVRG(M):
    feature_num, sample_num = X.shape
    B0 = 0.1 * np.ones((feature_num * class_num, 1))
    u = 1 / sample_num
    tol = 1e-2
    B = B0
    B_in = B0
    max_iteration = max_iteration_config['SVRG']
    S = soft_max(X, Y, B)
    g = log_func_gradient(X, Y, B, S, u)
    k = 1
    eta = 0.001
    error_list = []
    loss_list = []

    while (la.norm(g))**2 > tol and k < max_iteration:
        S_2 = soft_max(X, Y, B)
        for t in range(M):
            i = random.randint(0, sample_num-1)
            S_1 = soft_max(X, Y, B_in)
            g_1 = i_sample_gradient(X, Y, B_in, S_1, u, i)
            g_2 = i_sample_gradient(X, Y, B, S_2, u, i)
            B_in = B_in-eta*(g_1-g_2+g)
        B = np.copy(B_in)
        f = log_func(X, Y, B, u)
        print "k = %d: l(B) = %f" % (k, f), "in SVRG"
        loss_list.append(f)
        S = soft_max(X, Y, B)
        g = log_func_gradient(X, Y, B, S, u)
        k += 1

        error_list.append(calc_error(X_t, Y_t, B))
        print "Prediction error is: %f" % calc_error(X_t, Y_t, B)

    print log_func(X_t, Y_t, B, u)

    return loss_list, error_list


# ########################## set config to run three methods #############

class_num = 10
max_iteration_config = {'GD': 100, 'SGD': 3000, 'SVRG': 100}
GD()

SGD()
f_SVRG = {}
err_SVRG = {}
n_base = 10
C = [1 * n_base, 2 * n_base, 4 * n_base, 8 * n_base, 16 * n_base]
for k in range(len(C)):
    M = C[k]
    f_SVRG[k], err_SVRG[k] = SVRG(M)


# ############################ plot result of loss function ##########################
fig = plt.figure(1)
ax1 = fig.add_subplot(111)
f_res = [f_GD, f_SGD, f_SVRG[2]]
handles = []
n_base = 10
C = [sample_num, 1, 4*n_base+sample_num]
str_legend = ['GD', 'SGD', 'SVRG: m = 40']
color = ["red", "blue", "green"]
for k in range(len(str_legend)):
    x = np.array(range(len(f_res[k])))*C[k]
    plt.plot(x, f_res[k], color=color[k], linewidth=1.0, linestyle="-", label=str_legend[k])

plt.legend(loc='upper right')
ax1.set_xlim([0, 20000])
plt.xlabel('Gradient Eveluation Number')
plt.ylabel('f(B)')
ax1.grid(True)
plt.show()
plt.savefig('p_4_fB_full.png')


fig = plt.figure(2)
ax1 = fig.add_subplot(111)

handles = []
n_base = 10
C = [1*n_base, 2*n_base, 4*n_base, 8*n_base, 16*n_base]
str_legend = ['m = 10', 'm = 20', 'm = 40', 'm = 80', 'm = 160']
color = ["red", "blue", "green", "purple", "magenta"]
for k in range(len(str_legend)):
    x = np.array(range(len(f_SVRG[k])))*C[k]
    plt.plot(x, f_SVRG[k], color=color[k], linewidth=1.0, linestyle="-", label=str_legend[k])

plt.legend(loc='upper right')
ax1.set_xlim([0,1000])
plt.xlabel('Gradient Eveluation Number')
plt.ylabel('f(B)')
ax1.grid(True)
plt.show()
plt.savefig('p_4_fB_SVRG.png')

# ############################ plot result of error ###############################
fig = plt.figure(1)
ax1 = fig.add_subplot(111)

err_res = [err_GD, err_SGD, err_SVRG[2]]
handles = []
n_base = 10
C = [sample_num, 1, 4 * n_base + sample_num]
str_legend = ['GD', 'SGD', 'SVRG: m = 40']
color = ["red", "blue", "green"]
for k in range(len(str_legend)):
	x = np.array(range(len(err_res[k])))*C[k]
	plt.plot(x, err_res[k], color=color[k], linewidth=1.0, linestyle="-", label=str_legend[k])

plt.legend(loc='upper right')
ax1.set_xlim([0, 20000])
plt.xlabel('Gradient Eveluation Number')
plt.ylabel('error')
ax1.grid(True)
ax1.set_yscale('log')
plt.show()
plt.savefig('p_4_err_full.png')

fig = plt.figure(2)
ax1 = fig.add_subplot(111)

handles = []
n_base = 10
C = [1*n_base, 2*n_base, 4*n_base, 8*n_base, 16*n_base]
str_legend = ['m = 10', 'm = 20', 'm = 40', 'm = 80', 'm = 160']
color = ["red", "blue", "green", "purple", "magenta"]
for k in range(len(str_legend)):
    x = np.array(range(len(err_SVRG[k])))*C[k]  # problem
    plt.plot(x, err_SVRG[k], color=color[k], linewidth=1.0, linestyle="-", label=str_legend[k])

plt.legend(loc='upper right')
ax1.set_xlim([0, 1000])
plt.xlabel('Gradient Eveluation Number')
plt.ylabel('error')
ax1.grid(True)
plt.show()
plt.savefig('p_4_err_SVRG.png')
