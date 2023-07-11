import uproot
import numpy as np

# From simulations
#PMT1Charge_to_VEM = 1595.45
#PMT1Peak_to_VEM = 215.3
#PMT2Charge_to_VEM = 1575.99
#PMT2Peak_to_VEM = 214.3
#PMT3Charge_to_VEM = 1586.53
#PMT3Peak_to_VEM = 216.76
# PMT5Charge_to_VEM = 175#196.89
#PMT5Peak_to_VEM = 59.8

# From SdSimCalibrationConstants.xml G4StationSimulator
#PMT1Charge_to_VEM = 1582.15
#PMT1Peak_to_VEM = 215.254
#PMT2Charge_to_VEM = 1581.07
#PMT2Peak_to_VEM = 215.334
#PMT3Charge_to_VEM = 1583.52
#PMT3Peak_to_VEM = 215.582
#PMT5Charge_to_VEM = 198.918
#PMT5Peak_to_VEM = 60.078

# These values are from the rec adst when calling station.GetCharge(pmtid)
PMT1Charge_to_VEM = 1606.17
PMT2Charge_to_VEM = 1606.42
PMT3Charge_to_VEM = 1606.81
#PMT5Charge_to_VEM = 187.5
PMT5Charge_to_VEM = 152.37

PMT1Peak_to_VEM = 215.34
PMT2Peak_to_VEM = 215.74
PMT3Peak_to_VEM = 216.71
PMT5Peak_to_VEM = 51.8

# From Offline trunk
OnlineChargeVEMFactor = 1.045
VEMChargeConv = 1.01
MIPChargeConv = 1.16

VEMPeakConv = 0.87

# This happens in SdCalibrator, does this fix some things?
#PMT1Charge_to_VEM *= (OnlineChargeVEMFactor/VEMChargeConv)
#PMT2Charge_to_VEM *= (OnlineChargeVEMFactor/VEMChargeConv)
#PMT3Charge_to_VEM *= (OnlineChargeVEMFactor/VEMChargeConv)
#PMT5Charge_to_VEM *= (OnlineChargeVEMFactor/MIPChargeConv)
#
#PMT1Peak_to_VEM /= VEMPeakConv
#PMT2Peak_to_VEM /= VEMPeakConv
#PMT3Peak_to_VEM /= VEMPeakConv
# PMT5Peak_to_VEM /= VEMPeakConv # This cannot possibly be correct but that is how it is


def load_detector_data(fl, hastraces=False):
    tree = uproot.open(fl)['tree']

    # Fixing weird bug with duplicate keys gives KeyError (new??)
    keys = tree.keys()
    non_trace_keys = []
    for key in keys:
        if not 'Trace' in key and not key in non_trace_keys:
            non_trace_keys.append(key)
    #
    df = tree.arrays(library='pd', expressions=non_trace_keys)
    if hastraces:
        for i in [1, 2, 3, 5]:
            df[f'PMT{i}Trace'] = [
                _ for _ in tree[f'PMT{i}Trace'].array(library='np')]

        df['VEMTrace'] = (df['PMT1Trace'].to_numpy() +
                          df['PMT2Trace'].to_numpy() +
                          df['PMT3Trace'].to_numpy())/3
    df['VEMPeak'] = (df['PMT1Peak']/PMT1Peak_to_VEM +
                     df['PMT2Peak']/PMT2Peak_to_VEM +
                     df['PMT3Peak']/PMT3Peak_to_VEM)/3
    df['VEMCharge'] = (df['PMT1Charge']/PMT1Charge_to_VEM +
                       df['PMT2Charge']/PMT2Charge_to_VEM +
                       df['PMT3Charge']/PMT3Charge_to_VEM)/3
    df['MIPPeak'] = df['PMT5Peak']/PMT5Peak_to_VEM
    df['MIPCharge'] = df['PMT5Charge']/PMT5Charge_to_VEM / MIPChargeConv
    df = df.loc[:, ~df.columns.duplicated()]
    df['cosTheta'] = np.cos(np.pi-df['thetaDir'])
    print("Number of entries: ", len(df))
    return df
