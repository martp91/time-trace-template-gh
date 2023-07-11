import numpy as np


def Xmumax_50_from_Xmax(Xmax, theta=0.87266):
    # EPOS LHC proton
    # determined in bins of cos^theta
    # 19 < lgE < 20
    # Some additional systematics remain (between proton and iron below theta 40 deg for sure
    # up to 10 g/cm2
    # and also in energy above 19.5 ~10 g/cm2
    # sct = 1 / np.cos(theta)
    # a = 547.9
    # b = 0.736 + 0.5 * (sct - 1.556)
    # return a + b * (Xmax - 750)
    return 552 + 0.6 * (Xmax - 750)


# EPOS-LHC, proton/iron mean at lgE:19.5-20 (this is energy dependent!)
def Xmumax_att_factor(sct, a=0, b=58, c=-225, cut=1.556):  # 50 deg
    sct_ = sct - cut
    # flattens above sec theta = 1.3
    return a + b * sct_ + c * sct_ ** 2
