#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    make_scorer,
)

from time_templates.utilities.plot import plot_hist, plot_profile_1d
from time_templates.datareader.get_data import fetch_MC_data_from_tree

from time_templates.templates.universality.S1000_model import (
    S1000_comp_model,
    set_Rmu_df,
)
from time_templates.templates.universality.rho_model import set_total_signal_pred_df
from time_templates.templates.universality.names import (
    DICT_COMP_SIGNALKEY,
    eMUON,
    eEM_PURE,
    eEM_MU,
    eEM_HAD,
)
from time_templates.templates.universality.rho_model import XLABELS, RHOPIPEFILE
from time_templates import data_path

EPS = 1e-6


def score_func(y, y_pred):
    # return mean_poisson_deviance(np.maximum(y_pred, EPS), np.maximum(y, 0))
    # return mean_gamma_deviance(np.maximum(y_pred, EPS), np.maximum(y, EPS))
    return mean_absolute_percentage_error(np.maximum(y_pred, EPS), np.maximum(y, EPS))


custom_score = make_scorer(score_func, greater_is_better=False)


def fit_pipe(df, Xlabels, ylabel):
    X = df[Xlabels].values
    y = df[ylabel].values
    y[y <= EPS] = EPS

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # regr = LinearRegression()
    regr = Ridge(alpha=1)
    # regr = MLPRegressor(solver="adam", activation="tanh", verbose=True)
    # regr = BayesianRidge()
    # regr = ElasticNet()
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(4, include_bias=False)),
            # ("poly", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            # ("spline", SplineTransformer(5, 3, include_bias=False)),
            (
                "regr",
                TransformedTargetRegressor(regr, func=np.log1p, inverse_func=np.expm1),
            ),
        ]
    )

    alpha_grid = np.logspace(0, 3, 20)

    param_grid = {
        "poly__degree": [4],
        "regr__regressor__alpha": alpha_grid,
    }
    grid = GridSearchCV(
        pipe, param_grid, n_jobs=2, cv=5, scoring="neg_mean_squared_error"
    )
    grid.fit(X, y)
    print("Best parameters via grid search", grid.best_params_)
    pipe = grid.best_estimator_
    print("r2 score", pipe.score(X, y))
    y_pred = pipe.predict(X)
    y_pred[y_pred <= EPS] = EPS
    print("MSE", mean_squared_error(y, y_pred))
    print("MSAE", mean_absolute_error(y, y_pred))
    print("MAPE", mean_absolute_percentage_error(y_pred, y))
    print("MGD", mean_gamma_deviance(y, y_pred))
    print("MPD", mean_poisson_deviance(y, y_pred))

    return pipe


def predict_pipe(df, Xlabels, pipe):
    X = df[Xlabels].values
    return pipe.predict(X)


def fit_component_model_and_pred(df, Xlabels, ylabel):
    pipe = fit_pipe(df.query("primary=='proton'"), Xlabels, ylabel)
    return pipe


def fit_all(df):
    print("Fitting all pipes")

    # only proton is fitted
    dict_pipes = {}
    df["WCDTotalSignal_pred"] = 0
    print("Number of data points", len(df))
    for comp, signalkey in DICT_COMP_SIGNALKEY.items():
        print(comp)
        dict_pipes[comp] = fit_component_model_and_pred(df, XLABELS, "rho_" + comp)

    for comp, signalkey in DICT_COMP_SIGNALKEY.items():
        ylabel = "rho_" + comp
        df[ylabel + "_pred"] = predict_pipe(df, XLABELS, dict_pipes[comp])
        df[signalkey + "_pred"] = df[ylabel + "_pred"] * df[f"S1000_{comp}_pred"]
        df[signalkey + "_res"] = df[signalkey] / df[signalkey + "_pred"]

        df["WCDTotalSignal_pred"] += df[f"S1000_{comp}_pred"] * df[f"rho_{comp}_pred"]

    df["WCDTotalSignal_res"] = df["WCDTotalSignal"] / df["WCDTotalSignal_pred"]

    return df, dict_pipes


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cut_pred", action="store_true")
    parser.add_argument("--use_proton_iron", action="store_true")

    args = parser.parse_args()

    dfs = []
    for primary in ["proton", "iron"]:
        for energy in ["19_19.5", "19.5_20"]:
            df = fetch_MC_data_from_tree(
                primary=primary,
                energy=energy,
                HIM="EPOS_LHC",
                key="new_UUB_SSD_rcola",
                no_traces=True,
                dense=True,
                univ_comp=True,
                force=args.force,
            )
            dfs.append(df)
            # Also non dense?
            df = fetch_MC_data_from_tree(
                primary=primary,
                energy=energy,
                HIM="EPOS_LHC",
                key="new_UUB_SSD_rcola",
                no_traces=True,
                dense=False,
                univ_comp=True,
                force=args.force,
            )
            dfs.append(df)

    df = pd.concat(dfs)

    df = df.query(
        "LowGainSat == 0 & MCCosTheta > 0.6 & MCr < 2100 & MCr > 450 & MCDXstation > 10 & WCDTotalSignal > 0"
    )

    set_Rmu_df(df)
    set_total_signal_pred_df(df, Rmu=df["Rmu"].values)  # only for MC of course

    for comp, signalkey in DICT_COMP_SIGNALKEY.items():
        df[f"S1000_{comp}_pred"] = S1000_comp_model(
            df["MClgE"].values,
            df["MCDX_1000"].values,
            df["MCTheta"].values,
            df["Rmu"].values,
            comp,
        )
        df = df.query(f"S1000_{comp}_pred > 0")
        df[f"rho_{comp}"] = df[signalkey] / df[f"S1000_{comp}_pred"]

    if args.cut_pred:
        df = df.query("WCDTotalSignal_pred > 5")

    df_p = df.query("primary == 'proton'")
    df_Fe = df.query("primary == 'iron'")

    # First fit

    print(f"Saving at {RHOPIPEFILE}")

    if args.use_proton_iron:
        print(len(df_p), len(df_Fe))
        df_p = df_p.sample(len(df_Fe))
        df = pd.concat([df_p, df_Fe])
        df, dict_pipes = fit_all(df)
    else:
        df, dict_pipes = fit_all(df_p)

    with open(RHOPIPEFILE, "wb") as out:
        pickle.dump(dict_pipes, out)

    # ax = plot_hist(df["WCDTotalSignal"], bins=np.logspace(0, 4))
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # plt.show()

    df.to_pickle(data_path + "/df_rho_fitted.pl")
