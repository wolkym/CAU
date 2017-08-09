import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sc
import numpy as np


# Marchenko-Pastur cdf for rho > 1
def MPCdfLargeRho(x, rho):
    y1 = 1 - x + rho
    temp = np.sqrt(4 * rho - y1 * y1)
    y1 /= temp
    y1 = (1 + rho) * np.arctan(y1)
    y2 = x * (1 + rho)
    y2 -= (rho - 1) * (rho - 1)
    y2 /= (temp * (rho - 1))
    y2 = (rho - 1) * np.arctan(y2)
    y = np.pi - temp + y1 + y2
    y /= (2 * np.pi * rho)
    return 1 - y


# Marchenko-Pastur cdf for rho <= 1
def MPCdfSmallRho(x, rho):
    y1 = 1 - x + rho
    temp = np.sqrt(4 * rho - y1 * y1)
    y1 /= temp
    y1 = (1 + rho) * np.arctan(y1)
    if rho == 1:
        y2 = 0
    else:
        y2 = x * (1 + rho)
        y2 -= (rho - 1) * (rho - 1)
        y2 /= (temp * (1 - rho))
        y2 = (rho - 1) * np.arctan(y2)
    y = np.pi * rho + temp - y1 + y2
    y /= (2 * np.pi * rho)
    return y


def MPCdf(x, rho, sigmaSq):
    x0 = x / sigmaSq
    sqrtRho = np.sqrt(rho)
    a = (1 - sqrtRho) * (1 - sqrtRho)
    b = (1 + sqrtRho) * (1 + sqrtRho)
    if x0 >= b:
        return 1
    if x0 < 0:
        return 0
    if rho > 1:
        if x0 <= a:
            return 1 - 1 / rho
        return MPCdfLargeRho(x0, rho)
    if x0 <= a:
        return 0
    return MPCdfSmallRho(x0, rho)


# Marchenko-Pastur pdf
def MPPdf(x, rho, sigmaSq):
    if x == 0:
        if rho > 1:
            return np.inf
        return 0
    sqrtRho = np.sqrt(rho)
    a = sigmaSq * (1 - sqrtRho) * (1 - sqrtRho)
    b = sigmaSq * (1 + sqrtRho) * (1 + sqrtRho)
    if x < a or x > b:
        return 0
    pdf = np.sqrt((b - x) * (x - a))
    pdf /= (rho * x)
    pdf /= (2 * np.pi * sigmaSq)
    return pdf


def get_sorted_eigvals(Sigma):
    return np.sort(np.linalg.eigvals(Sigma))


def ESD(x, sorted_eig_values):
    return len(sorted_eig_values[sorted_eig_values <= x]) / len(sorted_eig_values)


def TVARCV(dX, Sigma_RCV):
    (p, n) = dX.shape
    Sigma = np.zeros((p, p))
    for l in range(n):
        dX_l = np.matrix(dX[:, l]).transpose()
        norm = np.linalg.norm(dX_l, ord=2)
        Sigma += dX_l.dot(np.transpose(dX_l)) / (norm * norm)
    Sigma_RCV -= 1 # remove this to get original TVARCV
    Sigma *= np.trace(Sigma_RCV) / n
    return Sigma


def RCV(dX):
    (p, n) = dX.shape
    Sigma = np.zeros((p, p))
    for l in range(n):
        dX_l = np.matrix(dX[:, l]).transpose()
        Sigma += dX_l.dot(np.transpose(dX_l))
    return Sigma


def get_gamma_piece_const(t):
    if t < 0.25 or t > 0.75:
        return np.sqrt(0.0007), 0.0004
    return np.sqrt(0.0001), 0.0004


def get_gamma_det(t):
    return np.sqrt(0.0009 + 0.0008 * np.cos(2 * np.pi * t)), 0.0009


# dX(t) = β(α - X(t))dt + σ(X(t))^(1/2)dB(t)
# CIR(t | α, β, σ, X0) ~ Y / c, where
# Y ~ Noncentral Chi - Squared(4αβ / σ ^ 2, X0 * c * exp(-βt)))
# and c = 4β / (σ ^ 2 * (1.0 - exp(-βt)))
def get_gamma_CIR(t):
    dt = t[1] - t[0]
    alpha = 1
    beta = 1
    sigma = 1
    exp_mbeta_dt = np.exp(-beta * dt)
    c = 4 * beta / (sigma * sigma * (1.0 - exp_mbeta_dt))
    degree = 4 * alpha * beta / (sigma * sigma)
    gamma = alpha * np.ones(len(t))
    sigma_sq = alpha * alpha * dt
    for i in range(1, len(gamma)):
        noncentrality = c * gamma[i - 1] * exp_mbeta_dt
        gamma[i] = np.random.noncentral_chisquare(degree, noncentrality) / c
        sigma_sq += gamma[i] * gamma[i] * dt
    return gamma, sigma_sq


def get_data_dep(n, p, time):
    alpha = 1
    beta = 1
    sigma_g = 0.0
    sigma_sq = alpha * alpha * (time[1] - time[0])
    X = np.zeros((n, p))
    gamma0 = 1
    volatility = gamma0 * np.ones(len(time) - 1)
    gamma = gamma0
    dX = np.random.normal(0, np.sqrt(time[1] - time[0]), p)
    dX *= gamma
    X[1] = X[0] + dX
    U = np.random.randint(n, size=p)
    for i in range(1, n - 1):
        dt = time[i + 1] - time[i]
        sqrt_dt = np.sqrt(dt)
        dX_l = np.random.normal(0, sqrt_dt, p)
        dW = np.sum(dX_l) / np.sqrt(p)
        # dW = np.random.normal(0, sqrt_dt)
        dg = beta * (alpha - gamma) * dt + sigma_g * np.sqrt(gamma) * dW
        gamma += dg
        sigma_sq += gamma * gamma * dt
        volatility[i] = gamma
        dX_l *= gamma

        for j in range(p):
            if U[j] == i:
                dX_l[j] += 1

        X[i + 1] = X[i] + dX_l
        dX = np.c_[dX, dX_l]
    plt.subplot(2, 2, 1)
    plt.plot(time[0:n - 1], volatility)
    plt.subplot(2, 2, 2)
    plt.plot(time, np.transpose(np.transpose(X)[0:5]))
    return dX, sigma_sq


def get_data(n, p, get_gamma):
    time = np.linspace(0, 1, n)
    sigma = np.sqrt(time[1] - time[0])
    X = np.zeros((n, p))
    gamma, sigma_sq = get_gamma(time)
    dX = np.random.normal(0, sigma, p)
    dX *= gamma[0]
    X[1] = X[0] + dX
    for i in range(1, n - 1):
        dX_l = np.random.normal(0, sigma, p)
        dX_l *= gamma[i]
        X[i + 1] = X[i] + dX_l
        dX = np.c_[dX, dX_l]
    return dX, sigma_sq


def main():
    n = 2000
    p = 500
    dX, sigma_sq = get_data_dep(n, p, np.linspace(0, 1, n))
    print("Data is ready")

    cdf = np.zeros(3000)
    b = 1 + np.sqrt(p / n)
    b *= b
    x = np.linspace(0, sigma_sq * b, 3000)
    for i in range(3000):
        cdf[i] = MPCdf(x[i], p / n, sigma_sq)

    plt.subplot(2, 2, 3)
    Sigma_RCV = RCV(dX)
    sorted_eigvals_RCV = get_sorted_eigvals(Sigma_RCV)
    plt.hist(sorted_eigvals_RCV, p, cumulative=True, normed=True)
    plt.plot(x, cdf, color='C2')
    print("RCV is ready")

    plt.subplot(2, 2, 4)
    sorted_eigvals_TVARCV = get_sorted_eigvals(TVARCV(dX, Sigma_RCV))
    plt.hist(sorted_eigvals_TVARCV, p, cumulative=True, normed=True, color='C1')
    plt.plot(x, cdf, color='C2')
    print("TVARCV is ready")

    plt.show()


main()
