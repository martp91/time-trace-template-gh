import numpy as np


def SdEnergy_resolution(E, rel=False):
    # Piere Auger(2020) https://arxiv.org/pdf/2008.06486.pdf
    SIGMA0 = 0.078
    SIGMA1 = 0.16
    ESIGMA = 6.6e18  # eV
    rel_sigma = SIGMA0 + SIGMA1 * np.exp(-E / ESIGMA)
    if rel:
        return rel_sigma
    return rel_sigma * E


def SdlgE_resolution(lgE):
    return SdEnergy_resolution(10 ** lgE, rel=True) / np.log(10)
