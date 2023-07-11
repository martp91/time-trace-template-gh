"""
Some aliases for latex strings that are commonly used
"""
import os

GCM2 = "\mathrm{{g\,cm^{{-2}}}}"
DXMAX = "DX_{{\\rm max}}"
XMAX = "X_{{\\rm max}}"
XMUMAX = "X^{\\mu}_{{\\rm max}}"


def get_info_from_file_str(fl):
    basename = os.path.splitext(os.path.basename(fl))[0]
    basename_low = basename.lower()
    if "epos" in basename_low:
        HIM = "EPOS-LHC"
    elif "qgs" in basename_low:
        HIM = "QGSJET"
    elif "sib" in basename_low:
        HIM = "SIBYLL"
    else:
        HIM = ""
    if "proton" in basename_low or "_p_" in basename_low:
        primary = "proton"
    elif "iron" in basename_low or "_Fe_" in basename_low:
        primary = "iron"
    else:
        primary = ""

    if "18.5" in basename_low:
        lgE_min = 18.5
        lgE_max = 19
    elif "19-19.5" in basename_low or "19_19.5" in basename_low:
        lgE_min = 19
        lgE_max = 19.5
    elif "19.5-20" in basename_low or "19.5_20" in basename_low:
        lgE_min = 19.5
        lgE_max = 20

    return HIM, primary, lgE_min, lgE_max, basename
