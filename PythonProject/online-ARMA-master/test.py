import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt
import datetime
#import pandas_datareader.data as web
from scipy import signal, linalg

from numpy.linalg import inv
from scipy.optimize import fmin_bfgs
import pandas as pd
import time


def gen_dataset1(n_samples=10000):
    alpha = np.array([0.6, -0.5, 0.4, -0.2, 0.3])
    beta = np.array([0.3, -0.2])
    a = 5
    b = 2
    sigma = 0.3

    noises = [0] * b
    arma = [0] * a
    for i in range(n_samples):
        noise = np.random.normal(0, sigma)
        x = np.sum(arma[:-a - 1:-1] * alpha)
        x += np.sum(noises[:-b - 1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    arma = np.array(arma[a:])
    return arma


def gen_dataset2(n_samples):
    # alpha1 = np.array([-0.4, -0.5, 0.4, 0.4, 0.1])
    alpha1 = np.array([0.4, 0.5, -0.4, -0.4, -0.1])
    alpha2 = np.array([0.6, -0.4, 0.4, -0.5, 0.4])
    beta = np.array([0.32, -0.2])
    a = 5
    b = 2

    noises = [0] * b
    arma = [0] * a
    for i in range(n_samples):
        noise = np.random.uniform(-0.5, 0.5)
        alpha = alpha1 * (i / float(n_samples)) + alpha2 * (1 - i / float(n_samples))
        x = np.sum(arma[:-a - 1:-1] * alpha)
        x += np.sum(noises[:-b - 1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    return np.array(arma[a:])


def gen_dataset3(n_samples=10000):
    n = int(n_samples / 2)
    # alpha1 = np.array([0.6, -0.5, 0.4, -0.4, 0.3])
    alpha1 = np.array([0.4, 0.5, -0.4, -0.3, -0.1])
    beta1 = np.array([0.3, -0.2])
    alpha2 = np.array([-0.4, -0.5, 0.4, 0.4, 0.1])
    beta2 = np.array([-0.3, 0.2])

    a = 5
    b = 2
    noises1 = [0] * b
    arma1 = [0] * a
    for i in range(n):
        noise = np.random.uniform(-0.5, 0.5)
        x = np.sum(arma1[:-a - 1:-1] * alpha1)
        x += np.sum(noises1[:-b - 1:-1] * beta1)
        x += noise
        arma1.append(x)
        noises1.append(noise)

    noises2 = [0] * b
    arma2 = [0] * a
    for i in range(n):
        noise = np.random.uniform(-0.5, 0.5)
        x = np.sum(arma2[:-a - 1:-1] * alpha2)
        x += np.sum(noises2[:-b - 1:-1] * beta2)
        x += noise
        arma2.append(x)
        noises2.append(noise)

    arma = arma1[a:] + arma2[a:]
    return np.array(arma)


def gen_dataset4(n_samples=10000):
    alpha = np.array([0.11, -0.5])
    beta = np.array([0.41, -0.39, -0.685, 0.1])
    a = 2
    b = 4
    sigma = 0.3

    noise = 0
    noises = [0] * b
    arma = [0] * a
    for i in range(n_samples):
        noise = np.random.normal(noise, sigma)
        x = np.sum(arma[:-a - 1:-1] * alpha)
        x += np.sum(noises[:-b - 1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    arma = np.array(arma[a:])
    return arma


def gen_temperature(n_samples=10000):
    t = sm.datasets.elnino.load()
    temps = []
    for year in t.data.tolist():
        temps.extend(year[1:])
    data = np.array(temps[0:n_samples])
    data = (data - np.mean(data)) / (np.max(data) - np.min(data))
    return data


def gen_stock(n_samples=10000):
    start = datetime.datetime(2000, 1, 1)
    end = datetime.datetime(2014, 1, 1)
    f = web.DataReader('^GSPC', 'yahoo', start, end)
    data = f['Close'].tolist()
    data = np.array(data)
    data = (data - np.mean(data)) / (np.max(data) - np.min(data))
    return data


def gen_dataset5(n_samples):
    alpha1 = np.array([-0.4, -0.5, 0.4, 0.4, 0.1])
    alpha2 = np.array([0.6, -0.4, 0.4, -0.5, 0.4])
    beta = np.array([0.32, -0.2])
    sigma = 0.3
    a = 5
    b = 2

    noises = [0] * b
    arma = [0] * a
    for i in range(n_samples):
        noise = np.random.normal(-0.5, sigma)
        if i > n_samples / 2:
            alpha = alpha1
        else:
            alpha = alpha2
        x = np.sum(arma[:-a - 1:-1] * alpha)
        x += np.sum(noises[:-b - 1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    return np.array(arma[a:])


def gen_dataset6(n_samples):
    alpha1 = np.array([-0.4, -0.5, 0.4, 0.4, 0.1])
    alpha2 = np.array([5, -0.5, 0.4, 0.4, 0.1])
    beta = np.array([0.32, -0.2])
    sigma = 0.3
    a = 5
    b = 2

    noises = [0] * b
    arma = [0] * a
    for i in range(n_samples):
        noise = np.random.normal(-0.5, sigma)
        if i == n_samples / 2:
            alpha = alpha2
        else:
            alpha = alpha1
        x = np.sum(arma[:-a - 1:-1] * alpha)
        x += np.sum(noises[:-b - 1:-1] * beta)
        x += noise
        arma.append(x)
        noises.append(noise)
    return np.array(arma[a:])


def simulate(datagen, N, arma, n, m, k, q, projection=True):
    # n : number of simulations
    # N : length of the simulations
    # arma : expected a function such as : arma_ons, amra_ogd...
    #############

    list_sim_loss = []
    list_sim_nabla = []
    list_sim_L = []

    for i in range(n):
        # simulate data
        X = datagen(N)
        # get historical parameter and loss

        X_p, loss_hist, nabla_hist, L_hist = arma(X, m, k, q, projection)

        list_sim_loss.append(loss_hist)
        list_sim_nabla.append(nabla_hist)
        list_sim_L.append(L_hist)

    return np.array(list_sim_loss), np.array(list_sim_nabla), np.array(list_sim_L)


def arma_ons(X, m, k, q, projection=True):
    """
    arma online newton step
    ici on prend c = 1 (majore les gamma en valeur absolue)  这里我们取c = 1,增加gamma的绝对值
    """
    # D    : Diamètre de l'espace des paramètres (参数的空间直径)
    # G    : Majorant de ||gradient(loss)||   ()
    # Square Loss -> lambda-exp-concave avec lambda = 1 / (m + k)
    # rate : Eta
    # A    : Proxy de la hessienne
    ##############################
    c = 1
    Xmax = 10
    D = 2 * c * np.sqrt((m + k))    # 决策集的直径
    G = 2 * c * np.sqrt(m + k) * Xmax ** 2  # G表示的是损失函数的上界
    lambda_ = 1.0 / (m + k)  # 当考虑平方损失函数的时候使用的

    rate = 0.5 * min(4 * G * D, lambda_)

    epsilon = 1.0 / (rate ** 2 * D ** 2)

    A = np.matrix(np.diag([1] * (m + k)) * epsilon)

    L = np.matrix(np.random.uniform(-c, c, (m + k, 1)))

    T = X.shape[0]

    X_p = np.zeros(T)

    loss_hist = np.zeros(T)
    nabla_norm_hist = np.zeros(T)    # nabla是一个微分算子 相当于那个倒三角符号
    L_hist = np.zeros(T)

    for t in range(T):
        # ----- Predict ----
        # ------------------
        X_t = 0
        for i in range(m + k):
            if t - i - 1 < 0:
                break
            X_t += L[i] * X[t - i - 1]
        X_p[t] = X_t

        # ----- Loss -------
        # ------------------
        loss_hist[t] = 0.5 * (X[t] - X_t) ** 2

        # ----- Update -----
        # ------------------
        nabla = np.zeros((m + k, 1))
        for i in range(m + k):
            x = X[t - i - 1] if t - i - 1 >= 0 else 0
            nabla[i, 0] = -2 * (X[t] - X_t) * x

        nabla_norm_hist[t] = np.linalg.norm(nabla, 2)  # 求二阶范数

        hess = np.dot(nabla, nabla.T)
        A = A + hess
        L = L - 1 / rate * np.dot(inv(A), nabla)
        if projection:
            L_norm = max(L.max(), -L.min())
            if L_norm > c:
                L /= L_norm
                L *= c

        L_hist[t] = L.max()

        # Sherman–Morrison formula for inverting A :
        # lambda_ = (1 / (1.0 + np.dot(np.dot(nabla.T, A_inv), nabla)[0,0]))
        # A_inv = A_inv - lambda_ * np.dot(A_inv, hess, A_inv)

    return X_p, loss_hist, nabla_norm_hist, L_hist