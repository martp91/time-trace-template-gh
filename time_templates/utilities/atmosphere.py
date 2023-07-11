import os
import numpy as np
import pandas as pd
import sqlite3
from scipy.misc import derivative
from numba import njit
from time_templates.utilities.nb_funcs import interp
from time_templates import package_path, data_path


R_E = 6.371 * 1e6  # radius of Earth
H_MAX = 112829.2  # height above sea level where the mass overburden vanishes

"""
Atmospheric density models as used in CORSIKA.
The parameters are documented in the CORSIKA manual
The parameters for the Auger atmospheres are documented in detail in GAP2011-133
The May and October atmospheres describe the annual average best.
parameters
    a in g/cm^2 --> g/m^2
    b in g/cm^2 --> g/m^2
    c in cm --> m
    h in km --> m
"""
default_model = 17
atm_models = {
    1: {  # US standard after Linsley
        "a": 1e4 * np.array([-186.555305, -94.919, 0.61289, 0.0, 0.01128292]),
        "b": 1e4 * np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0]),
        "c": 1e-2 * np.array([994186.38, 878153.55, 636143.04, 772170.16, 1.0e9]),
        "h": 1e3 * np.array([0.0, 4.0, 10.0, 40.0, 100.0]),
    },
    17: {  # US standard after Keilhauer
        "a": 1e4
        * np.array([-149.801663, -57.932486, 0.63631894, 4.35453690e-4, 0.01128292]),
        "b": 1e4 * np.array([1183.6071, 1143.0425, 1322.9748, 655.67307, 1.0]),
        "c": 1e-2 * np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.0e9]),
        "h": 1e3 * np.array([0.0, 7.0, 11.4, 37.0, 100.0]),
    },
    18: {  # Malargue January
        "a": 1e4
        * np.array(
            [-136.72575606, -31.636643044, 1.8890234035, 3.9201867984e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1174.8298334, 1204.8233453, 1637.7703583, 735.96095023, 1.0]),
        "c": 1e-2
        * np.array([982815.95248, 754029.87759, 594416.83822, 733974.36972, 1e9]),
        "h": 1e3 * np.array([0.0, 9.4, 15.3, 31.6, 100.0]),
    },
    19: {  # Malargue February
        "a": 1e4
        * np.array(
            [-137.25655862, -31.793978896, 2.0616227547, 4.1243062289e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1176.0907565, 1197.8951104, 1646.4616955, 755.18728657, 1.0]),
        "c": 1e-2
        * np.array([981369.6125, 756657.65383, 592969.89671, 731345.88332, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.2, 15.4, 31.0, 100.0]),
    },
    20: {  # Malargue March
        "a": 1e4
        * np.array(
            [-132.36885162, -29.077046629, 2.090501509, 4.3534337925e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1172.6227784, 1215.3964677, 1617.0099282, 769.51991638, 1.0]),
        "c": 1e-2
        * np.array([972654.0563, 742769.2171, 595342.19851, 728921.61954, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.6, 15.2, 30.7, 100.0]),
    },
    21: {  # Malargue April
        "a": 1e4
        * np.array(
            [-129.9930412, -21.847248438, 1.5211136484, 3.9559055121e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1172.3291878, 1250.2922774, 1542.6248413, 713.1008285, 1.0]),
        "c": 1e-2
        * np.array([962396.5521, 711452.06673, 603480.61835, 735460.83741, 1.0e9]),
        "h": 1e3 * np.array([0.0, 10.0, 14.9, 32.6, 100.0]),
    },
    22: {  # Malargue May
        "a": 1e4
        * np.array(
            [-125.11468467, -14.591235621, 0.93641128677, 3.2475590985e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1169.9511302, 1277.6768488, 1493.5303781, 617.9660747, 1.0]),
        "c": 1e-2
        * np.array([947742.88769, 685089.57509, 609640.01932, 747555.95526, 1.0e9]),
        "h": 1e3 * np.array([0.0, 10.2, 15.1, 35.9, 100.0]),
    },
    23: {  # Malargue June
        "a": 1e4
        * np.array(
            [-126.17178851, -7.7289852811, 0.81676828638, 3.1947676891e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1171.0916276, 1295.3516434, 1455.3009344, 595.11713507, 1.0]),
        "c": 1e-2
        * np.array([940102.98842, 661697.57543, 612702.0632, 749976.26832, 1.0e9]),
        "h": 1e3 * np.array([0.0, 10.1, 16.0, 36.7, 100.0]),
    },
    24: {  # Malargue July
        "a": 1e4
        * np.array(
            [-126.17216789, -8.6182537514, 0.74177836911, 2.9350702097e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1172.7340688, 1258.9180079, 1450.0537141, 583.07727715, 1.0]),
        "c": 1e-2
        * np.array([934649.58886, 672975.82513, 614888.52458, 752631.28536, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.6, 16.5, 37.4, 100.0]),
    },
    25: {  # Malargue August
        "a": 1e4
        * np.array(
            [-123.27936204, -10.051493041, 0.84187346153, 3.2422546759e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1169.763036, 1251.0219808, 1436.6499372, 627.42169844, 1.0]),
        "c": 1e-2
        * np.array([931569.97625, 678861.75136, 617363.34491, 746739.16141, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.6, 15.9, 36.3, 100.0]),
    },
    26: {  # Malargue September
        "a": 1e4
        * np.array(
            [-126.94494665, -9.5556536981, 0.74939405052, 2.9823116961e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1174.8676453, 1251.5588529, 1440.8257549, 606.31473165, 1.0]),
        "c": 1e-2
        * np.array([936953.91919, 678906.60516, 618132.60561, 750154.67709, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.5, 15.9, 36.3, 100.0]),
    },
    27: {  # Malargue October
        "a": 1e4
        * np.array(
            [-133.13151125, -13.973209265, 0.8378263431, 3.111742176e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1176.9833473, 1244.234531, 1464.0120855, 622.11207419, 1.0]),
        "c": 1e-2
        * np.array([954151.404, 692708.89816, 615439.43936, 747969.08133, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.5, 15.5, 36.5, 100.0]),
    },
    28: {  # Malargue November
        "a": 1e4
        * np.array(
            [-134.72208165, -18.172382908, 1.1159806845, 3.5217025515e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1175.7737972, 1238.9538504, 1505.1614366, 670.64752105, 1.0]),
        "c": 1e-2
        * np.array([964877.07766, 706199.57502, 610242.24564, 741412.74548, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.6, 15.3, 34.6, 100.0]),
    },
    29: {  # Malargue December
        "a": 1e4
        * np.array(
            [-135.40825209, -22.830409026, 1.4223453493, 3.7512921774e-4, 0.01128292]
        ),
        "b": 1e4
        * np.array([1174.644971, 1227.2753683, 1585.7130562, 691.23389637, 1.0]),
        "c": 1e-2
        * np.array([973884.44361, 723759.74682, 600308.13983, 738390.20525, 1.0e9]),
        "h": 1e3 * np.array([0.0, 9.6, 15.6, 33.3, 100.0]),
    },
}


def height_at_distance(d, zenith=0, observation_level=0):
    """Height above ground for given distance and zenith angle"""
    r = R_E + observation_level
    x = d * np.sin(zenith)
    y = d * np.cos(zenith) + r
    h = (x ** 2 + y ** 2) ** 0.5 - r
    return h


def slant_depth(h, zenith=0, model=default_model):
    """Amount of vertical atmosphere above given height.
    Args:
        h: height above sea level in meter
        model: atmospheric model, default is 17 (US standard after Keilhauer)
    Returns:
        slant depth in g/cm^2
    """
    a = atm_models[model]["a"]
    b = atm_models[model]["b"]
    c = atm_models[model]["c"]
    layers = atm_models[model]["h"]
    h = np.array(h)
    i = layers.searchsorted(h) - 1
    i = np.clip(i, 0, None)  # use layer 0 for negative heights
    x = np.where(i < 4, a[i] + b[i] * np.exp(-h / c[i]), a[4] - b[4] * h / c[4])
    x = np.where(h > H_MAX, 0, x)
    return x * 1e-4 / np.cos(zenith)


def X_z_atmos(height, theta, model=default_model):
    """ Same as slant_depth, but can be faster when not
    vectorized"""

    a = atm_models[model]["a"]
    b = atm_models[model]["b"]
    c = atm_models[model]["c"]
    layers = atm_models[model]["h"]

    section = 0
    for i, layer in enumerate(layers):
        if layer <= height and height <= layers[i + 1]:
            section = i
            break

    if height >= layers[-1]:
        section = -1  # or return 0?

    if section < 4:
        x = a[section] + b[section] * np.exp(-height / c[section])
    else:
        x = a[section] - b[section] * height / c[section]
    return x * 1e-4 / np.cos(theta)


def DX_at_station(r, psi, theta, Xmax, observation_level=1400, model=default_model):
    hproj = r * np.cos(psi) * np.sin(theta)
    iterable = True
    try:
        iter(r)
    except TypeError:
        iterable = False
    if iterable:
        X = slant_depth(observation_level + hproj, theta, model)
    else:
        ## Much faster (for 1 station)
        X = X_z_atmos(observation_level + hproj, theta, model)
    DX = X - Xmax
    return DX


def height_at_slant_depth(x, zenith=0, model=default_model):
    """Height for given slant depth (= traversed atmosphere).
    Args:
        x: traversed atmosphere in g/cm^2
        zenith: zenith angle
        model: atmospheric model, default is 17 (US standard after Keilhauer)
    Returns:
        height above sea level in meter
    """
    a = atm_models[model]["a"]
    b = atm_models[model]["b"]
    c = atm_models[model]["c"]
    layers = atm_models[model]["h"]
    xlayers = slant_depth(layers, model=model)
    x = np.array(x) * np.cos(zenith)
    i = xlayers.size - np.searchsorted(xlayers[::-1], x) - 1
    i = np.clip(i, 0, None)
    h = np.where(
        i < 4, -c[i] * np.log((x * 1e4 - a[i]) / b[i]), -c[4] * (x * 1e4 - a[4]) / b[4]
    )
    h = np.where(x <= 0, H_MAX, h)
    return h


def density(h, model=default_model):
    """Atmospheric density at given height
    Args:
        h: height above sea level in meter
        model: atmospheric model, default is 17 (US standard after Keilhauer)
    Returns:
        atmospheric overburden in g/m^3
    """
    h = np.array(h)

    if model == "barometric":  # barometric formula
        R = 8.31432  # universal gas constant for air: 8.31432 N m/(mol K)
        g0 = 9.80665  # gravitational acceleration (9.80665 m/s2)
        M = 0.0289644  # molar mass of Earth's air (0.0289644 kg/mol)
        rb = [1.2250, 0.36391, 0.08803, 0.01322, 0.00143, 0.00086, 0.000064]
        Tb = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
        Lb = [-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002]
        hb = [0, 11000, 20000, 32000, 47000, 51000, 71000]

        def rho1(_h, _i):  # for Lb == 0
            return rb[_i] * np.exp(-g0 * M * (_h - hb[_i]) / (R * Tb[_i]))

        def rho2(_h, _i):  # for Lb != 0
            return rb[_i] * (Tb[_i] / (Tb[_i] + Lb[_i] * (_h - hb[_i]))) ** (
                1 + (g0 * M) / (R * Lb[_i])
            )

        i = np.searchsorted(hb, h) - 1
        rho = np.where(Lb[i] == 0, rho1(h, i), rho2(h, i))
        rho = np.where(h > 86000, 0, rho)
        return rho * 1e3

    b = atm_models[model]["b"]
    c = atm_models[model]["c"]
    layers = atm_models[model]["h"]
    i = np.searchsorted(layers, h) - 1
    rho = np.where(i < 4, np.exp(-h / c[i]), 1) * b[i] / c[i]
    return rho


def distance(h, zenith=0, observation_level=0):
    """Distance for given height above ground and zenith angle"""
    r = R_E + observation_level
    return (h ** 2 + 2 * r * h + r ** 2 * np.cos(zenith) ** 2) ** 0.5 - r * np.cos(
        zenith
    )


# WARNING TODO: changed to heigh wrt sea level, need to change this in many things
# Anyway refactoring atmosphere fully
# Models above fitted with modified isothermal
# from 100-20000m, works well below 10000m (less than 1% defiation)
# above 20000m this is very bad (something like 200 g/cm2)
# units g/cm2, m, None
# Xg is at sea level
# Xg                    #hs                 #hs2
isothermal_approx_models = {
    17: [1036.0932777872404, 8281.933596010878, -0.08190986539357613],
    18: [1037.2334330264318, 8824.181228561014, -0.09976664274775274],
    19: [1038.1486103719974, 8794.853083683165, -0.09819065393533365],
    20: [1039.3132390353928, 8772.164490128544, -0.0982168970049954],
    21: [1041.504542179216, 8696.830340914117, -0.09721777461522163],
    22: [1044.1601784675947, 8593.31801144592, -0.09368601680076438],
    23: [1044.9069889957204, 8480.46451195711, -0.0901757682985899],
    24: [1047.201275815531, 8391.832778836948, -0.08470119440400878],
    25: [1046.977020453506, 8391.4015852602, -0.08391068045044624],
    26: [1048.6630216066874, 8401.559111546909, -0.08407505760656638],
    27: [1044.3346370161692, 8523.771135006336, -0.08939982558101651],
    28: [1041.0684513385704, 8632.831824515077, -0.09358678240323903],
    29: [1038.7642372478244, 8733.607239395144, -0.09692460037238647],
}


@njit(fastmath=True, cache=True, nogil=True, error_model="numpy")
def isothermal_modified(h, Xg, hs, hs2=0):
    h0 = hs + hs2 * h

    # if height is very small (as can happen from time computation
    # then exp can blow up so cut it somewhere
    h0 = np.where(h0 <= 1e-5, hs, h0)
    return Xg * np.exp(-h / h0)


@njit(fastmath=True, cache=True, nogil=True, error_model="numpy")
def isothermal_modified_derivative(h, Xg, hs, hs2=0):
    h0 = hs + hs2 * h
    h0 = np.where(h0 <= 1e-5, hs, h0)
    prefact = hs / h0 ** 2
    return -prefact * isothermal_modified(h, Xg, hs, hs2)


@njit(fastmath=True, cache=True, nogil=True, error_model="numpy")
def height_from_X_isothermal(X, Xg, hs, hs2=0):
    return hs * np.log(Xg / X) / (1 - hs2 * np.log(Xg / X))


def slant_depth_isothermal(h, theta=0, model=21):
    Xg, hs, hs2 = isothermal_approx_models[model]
    return isothermal_modified(h, Xg, hs, hs2) / np.cos(theta)


def dXdh_isothermal(h, theta, model=21):
    Xg, hs, hs2 = isothermal_approx_models[model]
    return isothermal_modified_derivative(h, Xg, hs, hs2) / np.cos(theta)


def height_at_slant_depth_isothermal(X, theta, model=21):
    Xg, hs, hs2 = isothermal_approx_models[model]
    return height_from_X_isothermal(X * np.cos(theta), Xg, hs, hs2)


def get_X_station_proj_isothermal(r, psi, theta, hstation=0, model=21):
    hproj = r * np.cos(psi) * np.sin(theta) + hstation
    X = slant_depth_isothermal(hproj, theta, model=model)
    return X


def DX_at_station_isothermal(r, psi, theta, Xmax, hstation=0, model=21):
    X = get_X_station_proj_isothermal(r, psi, theta, hstation, model)
    DX = X - Xmax
    return DX


### Database stuff GDAS

# WARING what if database changes, TODO need to update database auto ->fetchSqliteDbs.sh
# TODO: if file is broken it will not give an error here
# TODO: get db path if not default
def setup_molecular_db(fl=os.path.join(data_path, "dbs", "Atm_Molecular_1_A.sqlite")):
    try:
        return sqlite3.connect(fl)
    except sqlite3.OperationalError:
        print(
            f"WARNING: Could not read Atm_Molecular database  at {fl} it will not be available for atmosphere"
        )


connection = setup_molecular_db()


def query_database_molecular_profile_gdas(gpssec, con):
    # Takes the closest entry to gpssec
    # returns dataframe with all
    # Copy from Offline/Framework/Atmosphere/AMolecularSQLManager.cc
    ptype = 1  # GDAS
    query = (
        f"SELECT molecular_id FROM molecular WHERE start_time = (SELECT MAX(start_time) "
        f"FROM molecular WHERE profile_type_id = {ptype} AND start_time <= {gpssec}"
        f" AND end_time >= {gpssec}) AND profile_type_id = {ptype}"
        f" ORDER BY last_modified DESC LIMIT 1"
    )
    df = pd.read_sql_query(
        f"SELECT * FROM molecular_layer WHERE molecular_zone_id = ({query})", con
    )
    if df.empty:
        raise IndexError("GDAS could not be found")
    df["log_height"] = np.log(df["height"])
    df["log_depth"] = np.log(df["depth"])
    df = df.sort_values(by="height")
    return df


def slant_depth_table(height, table, theta=0):
    """interpolate in log
    height above sea level in m
    returns X g/cm2
    """
    return np.exp(
        interp(height, table["height"].values, table["log_depth"].values)
    ) / np.cos(theta)


def height_at_slant_depth_table(depth, table, theta=0):
    """Interpolate in log
    height above sea level in m
    returns X g/cm2
    """
    # df is sorted by height
    # have to reverse the order to get depth sorted and interpolate
    return interp(
        np.log(depth), table["log_depth"].values[::-1], table["height"].values[::-1],
    )


class Atmosphere:
    "Simple atmosphere class for slant depth, can take atmosphere model or take gdas"

    def __init__(self, gps_seconds=None, model=21, isothermal=True):
        self.is_data = False
        self.gps_seconds = gps_seconds
        if gps_seconds is not None:
            self.table = query_database_molecular_profile_gdas(gps_seconds, connection)
            self.is_data = True

        self.model = model
        self.isothermal = isothermal

    def __del__(self):
        try:
            del self.table
        except:
            pass

    def slant_depth_at_height(self, height, theta=0):
        if self.is_data:
            return slant_depth_table(height, self.table, theta)
        else:
            if self.isothermal:
                return slant_depth_isothermal(height, theta, self.model)
            else:
                return slant_depth(height, theta, self.model)

    def height_at_slant_depth(self, depth, theta=0):
        if self.is_data:
            return height_at_slant_depth_table(depth, self.table, theta)
        else:
            if self.isothermal:
                return height_at_slant_depth_isothermal(depth, theta, self.model)
            else:
                return height_at_slant_depth(depth, theta, self.model)

    # dx = 500 gives fairly smooth
    def dXdh(self, height, theta=0, dx=500):
        "Derivative"
        if not self.is_data and self.isothermal:
            return dXdh_isothermal(height, theta, self.model)
        return derivative(lambda h: self.slant_depth_at_height(h, theta), height, dx=dx)

    def dhdX(self, depth, theta=0, dx=500):
        return derivative(lambda X: self.height_at_slant_depth(X, theta), depth, dx=dx)

    def Xg_at_station(self, r, psi, theta, observation_level=1400):
        """
        Slant depth projected for station along shower axis
        observation_level=1400 is standard corsika
        for data make sure this is correct station height
        """
        hproj = r * np.cos(psi) * np.sin(theta)
        return self.slant_depth_at_height(observation_level + hproj, theta)

    def DX_at_station(self, r, psi, theta, Xmax, observation_level=1400):
        """slant depth distance to Xmax along shower axis"""
        return self.Xg_at_station(r, psi, theta, observation_level) - Xmax
