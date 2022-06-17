import numpy as np
from numpy.linalg import inv
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import data
from statsmodels.tsa.arima_process import arma_generate_sample


def K_min(y, A):
    def f(x):
        tmp = np.matrix(y).reshape(-1, 1) - np.matrix(x).reshape(-1, 1)
        result = np.dot(tmp.T, A)
        result = np.dot(result, tmp)
        return result[0, 0]
    return f


def arma_ons(X, m, k, q):
    """
    arma online newton step
    """
    D = np.sqrt(2*(m+k))  # sqrt返回平方根
    G = 2*np.sqrt(m+k)*D    # 定义 G
    rate = 0.5*min(1./(m+k), 4*G*D)  # min返回最小值，定义一个学习率吧应该是
    epsilon = 1./(rate**2 * D**2)   # 只是简单的除法
    A = np.diag([1]*(m+k)) * epsilon   # diag()函数:构造一个对角矩阵，不在对角线上的元素全为0，或者以向量的形式返回一个矩阵上的对角线元素
    A = np.matrix(A)    # 生成一个矩阵
    T = X.shape[0]  # 输出矩阵的行数 x.shape[1] 输出矩阵的列数
    """
     numpy.random.uniform(low, high, size)
     从一个均匀分布[low, high)中随机采样，定义域是左闭右开
     size: 输出样本的数目，为int或tuple类型，
        例如：size = (m, n, k) 输出m*n*k个样本，缺省时输出1个值
    返回值： ndarray类型
    """
    L = np.random.uniform(-0.5, 0.5, (m+k, 1))
    L = np.matrix(L)    # 生成一个对角线元素为数组L的矩阵

    X_p = np.zeros(T)  # 返回一个用0填充的数组
    loss = np.zeros(T)  # 返回一个用0填充的数组
    for t in range(T): # T是输出的矩阵的行数，对这些行进行操作
        # predict
        X_t = 0  # 定义一个参数
        for i in range(m+k):  # (m+k)维的系数vector
            if t-i-1 < 0:
                break
            X_t += L[i]*X[t-i-1]  #
        X_p[t] = X_t

        # loss
        loss[t] = (X[t]-X_t)**2

        # update
        nabla = np.zeros((m+k, 1))
        for i in range(m+k):
            x = X[t-i-1] if t-i-1 >= 0 else 0
            nabla[i, 0] = -2*(X[t]-X_t)*x
        A = A + np.dot(nabla, nabla.T)  # 两个矩阵相乘 矩阵和自己的转置相乘
        # y = L - 1/rate*np.dot(inv(A), nabla)
        # L = fmin_bfgs(K_min(y, A), L)
        # L = np.matrix(L).reshape(-1, 1)
        L = L - 1/rate*np.dot(inv(A), nabla) # inv函数返回A的逆矩阵，然后再和nabla矩阵点积
    return X_p, loss


def arma_ogd(X, m, k, q):
    """
    ARMA online gradient descent
    """
    D = np.sqrt(2*(m+k))
    G = 2*np.sqrt(m+k)*D
    T = X.shape[0]
    rate = D/(G*np.sqrt(T))

    L = np.random.uniform(-0.5, 0.5, (m+k, 1))
    L = np.matrix(L)

    X_p = np.zeros(T)
    loss = np.zeros(T)
    for t in range(T):
        # predict
        X_t = 0
        for i in range(m+k):
            if t-i-1 < 0:
                break
            X_t += L[i]*X[t-i-1]
        X_p[t] = X_t

        #loss
        loss[t] = (X[t]-X_t)**2

        #update
        nabla = np.zeros((m+k, 1))
        for i in range(m+k):
            x = X[t-i-1] if t-i-1 >= 0 else 0
            nabla[i, 0] = -2*(X[t]-X_t)*x
        L = L - rate*nabla
    return X_p, loss


def gen_errors(loss):
    n = len(loss)
    errors = np.zeros(n)
    for i in range(n):
        errors[i] = np.sum(loss[0:i+1])/(i+1)
    return errors


def average(datagen, N, arma, n):
    avg = np.zeros(N)
    for i in range(n):
        X = datagen(N)
        X_p, loss = arma(X, 5, 5, 0)
        avg += loss
    avg = avg / n
    return avg

if __name__ == '__main__':
    n = 10000
    iters = 2
    t = range(n)
    X = data.gen_dataset1(n)

    plt.subplot(221)
    loss = average(data.gen_dataset1, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-ONS")

    loss = average(data.gen_dataset1, n, arma_ogd, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-OGD")
    plt.legend()
    plt.title("Sanity check")

    plt.subplot(222)
    loss = average(data.gen_dataset2, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-ONS")

    loss = average(data.gen_dataset2, n, arma_ogd, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-OGD")
    plt.legend()
    plt.title("Slowly changing coefficients")

    plt.subplot(223)
    loss = average(data.gen_dataset3, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-ONS")

    loss = average(data.gen_dataset3, n, arma_ogd, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-OGD")
    plt.legend()
    plt.title("Abrupt change")

    plt.subplot(224)
    loss = average(data.gen_dataset4, n, arma_ons, iters)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-ONS")

    # loss = average(data.gen_dataset4, n, arma_ogd, iters)
    # e = gen_errors(loss)
    # plt.plot(t, e, label="ARMA-OGD")
    plt.legend()
    plt.title("Correlated noise")

    plt.show()

    #for real data
    plt.subplot(121)
    X = data.gen_temperature()
    n = X.shape[0]
    t = range(n)

    X_p, loss = arma_ons(X, 5, 5, 0)
    e = gen_errors(loss)
    plt.plot(t, e, label="ARMA-ONS")

    X_p, loss = arma_ogd(X, 5, 5, 0)
    e = gen_errors(loss)
    plt.plot(t, e, label='AMRA-OGD')
    plt.legend()

    plt.subplot(122)
    X = data.gen_stock()
    n = X.shape[0]
    t = range(n)

    X_p, loss = arma_ons(X, 5, 5, 0)
    e = gen_errors(loss)
    plt.plot(t, e, label='ARMA-ONS')

    X_p, loss = arma_ogd(X, 5, 5, 0)
    e = gen_errors(loss)
    plt.plot(t, e, label='ARMA-OGD')
    plt.legend()
    plt.show()
