# This is deprecated?


## parameters that determine the signal spread of 1 particle

# filepath = os.path.dirname(os.path.realpath(__file__))
#
# with open(os.path.join(filepath, "UUB_muon_response_pdf.json"), "rb") as infile:
#    signal_response = json.load(infile)
#
# wcd_moy_loc, wcd_moy_scale = signal_response["wcd"]
# wcd_moy_mean, wcd_moy_std = convert_moyal_loc_scale(wcd_moy_loc, wcd_moy_scale)
# wcd_moy_mean = 1

# By taking the first 5 bins in sims (no signal in :20)

UUB_wcd_baseline_HG_sigma = 0.017  # This is probably wrong
UB_wcd_baseline_HG_sigma = 0.0054
UUB_wcd_baseline_LG_sigma = 0.048
UB_wcd_baseline_LG_sigma = 0.17


@njit(fastmath=True)
def sample_muon_wcd_signal(
    mu_Smu, Emu_mean, Emu_sig=None, theta=0, Evem=EVEM, Ech=ECH, gaussian_limit=20
):
    """
    Sample muons with:
    mu_particles = the mean number of muons
    Emu_mean = the mean energy of the muons
    The std of the energy is assumed to be the same as Emu_mean
    """
    # gaussian limit is reached pretty fast, at ~8 particles
    # but doing the error function takes about 22 mus
    # so this is only benefiial for >10-20 particles
    if Emu_sig is None:
        Emu_sig = Emu_mean
    Lmean = f_Lwcd(theta)
    amean = f_average_muon_signal_exact(Emu_mean, Emu_sig, L=Lmean)

    # expected number of muon particles
    mu_Nmu = mu_Smu / amean

    if mu_Nmu > gaussian_limit:
        # Use average track length and muon signal
        # assuming the std on amean is negligible, not true because of L!
        sig = amean * np.sqrt(mu_Nmu + wcd_moy_std ** 2)
        return np.random.normal(mu_Smu, sig)

    n = np.random.poisson(mu_Nmu)

    # using numba is faster here since we are not often in the gauss limit (I think?)
    #     Ei = np.random.lognormal(*convert_lognorm_mean(Emu_mean, Emu_sig), n)
    #     Li = signal_model.sample_track_length(theta, n, signal=True, ninterp=100)
    #     Smu = np.minimum((Ei/Evem)**2, Li)
    #     Si = np.sum(Smu*sample_signal_moyal_v(wcd_moy_loc, wcd_moy_scale, n))
    Si = 0
    lognorm_loc, lognorm_scale = convert_lognorm_mean(Emu_mean, Emu_sig)
    for j in range(n):
        Eij = np.random.lognormal(lognorm_loc, lognorm_scale)
        if Eij > Ech:
            if Eij < Evem:
                Smu = (Eij / Evem) ** 2
            else:
                Smu = sample_track_length(theta, signal=True, ninterp=100)
            npe = np.random.poisson(Smu * WCDCharge_to_VEM)
            Smu = npe / WCDCharge_to_VEM
            Si += Smu * sample_signal_moyal(wcd_moy_loc, wcd_moy_scale)
            # All the uncertainty of the detector response is now into the total charge (or peak
            # There should actually be an uncertainty on the time response
    # note: mean moy is 1.04 for wcd

    ## Depends on going for mode or mean. Let's just now say we want to reproduce what is put in
    return Si / wcd_moy_mean


# # pseudo-code
# def sample_wcd_muon(Smu, Emean, Estd, theta):
#    m, s = convert_lognorm_mean(Emean, Estd)
#    a = average_muon_signal(m, s) # average signal per particle
#    Nmu = Smu/a # expected n particles
#    N = np.random.poisson(Nmu)
#    S = 0
#    for i in range(N):
#        E = np.random.lognormal(m, s)
#        L = sample_track_length(theta)
#        if E > ECH:
#            if E < EVEM:
#                Si = (E/EVEM)**2
#            else:
#                Si = L/L0
#            # n photo-electrons for signal
#            npe = np.random.poisson(Si*WCDCharge_to_VEM)
#            Smu = npe/WCDCharge_to_VEM
#            S += sample_signal_landau(Si)
#    return S


@njit(fastmath=True)
def sample_muon_wcd_signal_v(
    mu_Smu, Emu_mean, Emu_sig=None, theta=0, Evem=EVEM, Ech=ECH, gaussian_limit=20
):
    """
    Vectorized version of sample_muon_wcd_signal
    Note: no detector time response (convolution) #TODO
    """
    if Emu_sig is None:
        Emu_sig = Emu_mean
    if not (len(mu_Smu) == len(Emu_mean)):
        raise ValueError("Emu needs to be same size as mu")
    size = len(mu_Smu)
    out = np.zeros(size)
    for i in range(size):
        if mu_Smu[i] <= 0:
            continue
        out[i] = sample_muon_wcd_signal(
            mu_Smu[i], Emu_mean[i], Emu_sig[i], theta, Evem, Ech, gaussian_limit
        )
    return out


## sampling from random normal is super fast without numba (4 times faster)
##can do this vectorized anyway
@njit(fastmath=True)
def sample_em_wcd_signal(mu_Sem, Eem_mean, Eem_sig=None, Evem=EVEM, gaussian_limit=50):
    """
    mu_Sem is mean EM signal (energy flow)
    """
    if Eem_sig is None:
        Eem_sig = 2 * Eem_mean  # Approx OK
    Nem = mu_Sem / (Eem_mean / Evem)
    Si = 0

    # The limit is reached pretty fast, but I saw some discrepancies so 50 is save
    # Probably because spread can be quite large from energy, so avoid 0
    if Nem > gaussian_limit:
        #        sig = np.sqrt(
        sig = np.sqrt(mu_Sem / (Eem_mean * Evem)) * np.sqrt(
            Eem_sig ** 2 + Eem_mean ** 2 + wcd_moy_std ** 2
        )
        return np.random.normal(mu_Sem, sig)

    n = np.random.poisson(Nem)
    #    Ei = np.random.lognormal(*convert_lognorm_mean(Eem_mean, Eem_sig), n)
    #    Si = np.sum(Ei/Evem * sample_signal_moyal_v(wcd_moy_loc, wcd_moy_scale, n))
    Si = 0
    lognorm_loc, lognorm_scale = convert_lognorm_mean(Eem_mean, Eem_sig)
    for j in range(n):
        Eij = np.random.lognormal(lognorm_loc, lognorm_scale)
        Sem = Eij / Evem

        # Sample pe from poisson extra uncertainty
        # This is not in the gaussian limit so hmmm. WARNING: UB
        npe = np.random.poisson(Sem * WCDCharge_to_VEM)
        Sem = npe / WCDCharge_to_VEM

        Si += Sem * sample_signal_moyal(wcd_moy_loc, wcd_moy_scale)

    # Divide out offset mean mode
    return Si / wcd_moy_mean


@njit(fastmath=True)
def sample_em_wcd_signal_v(
    mu_Sem, Eem_mean, Eem_sig=None, Evem=EVEM, gaussian_limit=50
):
    """
    Vectorized version
    """
    if Eem_sig is None:
        Eem_sig = 2 * Eem_mean  # approx OK
    if not (len(mu_Sem) == len(Eem_mean)):
        raise ValueError("Eem needs to be same size as mu")
    size = len(mu_Sem)
    out = np.zeros(size)
    for i in range(size):
        if mu_Sem[i] <= 0:
            continue
        out[i] = sample_em_wcd_signal(
            mu_Sem[i], Eem_mean[i], Eem_sig[i], Evem, gaussian_limit
        )
    return out
