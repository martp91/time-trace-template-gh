import os
import pickle
import numpy as np
from scipy.special import comb
from numba import njit
from time_templates import package_path
from time_templates.templates.universality.S1000_model import (
    S1000_comp_model,
    set_Rmu_df,
)
from time_templates.templates.universality.names import (
    DICT_COMP_SIGNALKEY,
    eMUON,
    eEM_PURE,
    eEM_MU,
    eEM_HAD,
)

data_path = os.path.join(package_path, "data")

XLABELS = ["MCCosTheta", "MCDXstation", "MClgr", "MCSinThetaCosPsi"]

# filepath = os.path.dirname(os.path.abspath(__file__))

RHOPIPEFILE = os.path.join(data_path, "rho_pipes.pl")

try:
    with open(RHOPIPEFILE, "rb") as infile:
        rho_pipes = pickle.load(infile)
except FileNotFoundError:
    print("First run this script to fit rho")


def make_poly(X, degree, include_bias=False):
    n_samples, n_features = X.shape
    n_output_features = comb(n_features + degree, degree, exact=True)

    XP = np.empty((n_samples, n_output_features), dtype=X.dtype, order="C")

    XP[:, 0] = 1
    pos = 1
    n = X.shape[1]
    for d in range(0, degree):
        if d == 0:
            XP[:, pos : pos + n] = X
            index = list(range(pos, pos + n))
            pos += n
            index.append(pos)
        else:
            new_index = []
            end = index[-1]
            for i in range(0, n):
                a = index[i]
                new_index.append(pos)
                new_pos = pos + end - a
                XP[:, pos:new_pos] = np.multiply(XP[:, a:end], X[:, i : i + 1])
                pos = new_pos

            new_index.append(pos)
            index = new_index

    if include_bias:
        Xout = XP
    else:
        Xout = XP[:, 1:]
    return np.ascontiguousarray(Xout)


@njit(fastmath=True)
def predict_exp(Xpoly, intercept, coef):
    return np.exp(intercept + np.dot(Xpoly, coef)) - 1


def predict_custom_Xpoly(Xpoly, pipe):
    regr = pipe["regr"].regressor_
    coef = regr.coef_
    intercept = regr.intercept_
    return predict_exp(Xpoly, intercept, coef)


def predict_custom(X, pipe):
    # twice faster no checks!
    std = pipe["scale"].scale_
    mean = pipe["scale"].mean_
    X = (X - mean) / std
    # Xpoly = pipe["poly"].transform(X)
    Xpoly = make_poly(X, degree=pipe["poly"].degree)
    return predict_custom_Xpoly(Xpoly, pipe)


def get_rho_comp_poly(Xpoly, comp):
    return np.maximum(predict_custom_Xpoly(Xpoly, rho_pipes[comp]), 0)


def get_rho_comp(X, comp):
    return np.maximum(predict_custom(X, rho_pipes[comp]), 0)


# def get_rho_comp(X, comp):
#    return np.maximum(rho_pipes[comp].predict(X), 0)


def make_X_rho(theta, DX, r, psi, n=1):
    if n > 1:
        if np.any(r > 2100) or np.any(r < 490):
            print("WARNING: model not build for r > 2000 or r < 500")
        X = np.array([np.cos(theta), DX, np.log(r), np.sin(theta) * np.cos(psi)]).T
    else:
        if r > 2100 or r < 490:
            print("WARNING: model not build for r > 2000 or r < 500")
        X = np.array([[np.cos(theta), DX, np.log(r), np.sin(theta) * np.cos(psi)]])
    return X


def get_rho_signal_comp(theta, DX, r, psi, comp):
    X = make_X_rho(theta, DX, r, psi)
    return get_rho_comp(X, comp)


def get_rho_signal_all(theta, DX, r, psi, n=1):
    # WARNING: assumes all comps same degree for X
    X = make_X_rho(theta, DX, r, psi, n=1)
    Xpoly = make_poly(X, degree=rho_pipes[eMUON]["poly"].degree)
    return (get_rho_comp(Xpoly, e) for e in [eMUON, eEM_PURE, eEM_MU, eEM_HAD])


def get_total_signal_comp(lgE, DX1000, theta, DX, r, psi, comp, Rmu=1):
    S1000 = S1000_comp_model(lgE, DX1000, theta, Rmu, comp)
    rho = get_rho_signal_comp(theta, DX, r, psi, comp)
    return S1000 * rho


def set_comp_total_signal_df(df, comp, rho_Xlabels=XLABELS, Rmu=None):
    if "Rmu" not in df.keys():
        set_Rmu_df(df)
    if Rmu is None:
        Rmu = df["Rmu"].values
    S1000 = S1000_comp_model(
        df["MClgE"].values, df["MCDX_1000"].values, df["MCTheta"].values, Rmu, comp
    )
    df["S1000_" + comp + "_pred"] = S1000
    X = df[rho_Xlabels].values
    try:
        rho = rho_pipes[comp].predict(X)
    except ValueError as e:
        print("FAILED!")
        print(X.min(axis=0), X.max(axis=0))
        raise e
    df["rho_" + comp + "_pred"] = rho
    df["rho_" + comp + "_MC"] = df[DICT_COMP_SIGNALKEY[comp]] / S1000
    df[DICT_COMP_SIGNALKEY[comp] + "_pred"] = S1000 * rho


def set_total_signal_pred_df(df, Rmu=None):
    df["WCDTotalSignal_pred"] = 0
    for comp, signalkey in DICT_COMP_SIGNALKEY.items():
        set_comp_total_signal_df(df, comp, Rmu=Rmu)
        df["WCDTotalSignal_pred"] += df[DICT_COMP_SIGNALKEY[comp] + "_pred"]
