These files are loaded as a dictionary and supplied to
time_templates/fittemplates/fit_events.py/fit_events().\
Default is in sd_Xmax_fit.json (also shows all possible entries).

From fit_event.py docstring (see there for more info)
```
    reg : dict
        dictionary with regularization parameters for Rmu, lgE, Xmax (Default is no regularization)
    cuts : dict
       cuts to make on total signal, rmin, rmax and for trace optionally on expected number of particles
       or vem charge in bin. tq_cut cuts at the end of the trace that contains tq (time quantile) of the
       total signal. tq_cut=0.95 is default, because this proved to work best
    plot : bool
        plot if True
    no_scale : bool
        do not convert vem into neff particles with poisson factor default=False
    MClgE : bool
        if True, use MC energy and sample from normal distribution with SdlgE resolution to get some fluct.
        default = False
    verbose : bool
        if True print some stuff. default=True
    fix_lgE : bool
        if True fix energy in template fit and also total signal fit, default=True
    fix_Rmu_timefit : bool
        if True fix Rmu in ttt-fit, default=True
    fix_Xmax : bool
        if True fix Xmax in ttt-fit
    useFdlgE : bool
        if True use FdlgE, only works on hybrid, default=False
    useFdXmax : bool
        if True, use FdXmax, only works on hyrbdi (for example if you want to fit Rmu). default=False. see
        also MCXmax
    fit_time_templates : bool
        if True fit ttt (default), but set to false if you only want to fit Rmu from LDF fit for example
    fix_t0s_final : bool
        if True (default), fix t0's in the final fit round (and only fit Xmax). The t0's were allowed to
        fluctuate in an earlier fit, this works best.
    use_Xmumax : bool
        use Xmumax for muon template (experimental!, does not really work yet). default=False
    fix_Xmumax : bool
        if use_Xmumax then still fix_Xmumax, default=True, but has no effect if not use_Xmumax
    MCXmax : bool
        if useFdXmax but it is MC sims, then also set this to True, to sample Xmax from trueMC with resolution
        from FD. default=False
    use_data_pulse_shape : bool
        get MuonPulseShape from average of PMTs and use this as a detector response (tau=40-60ns or
        something). default=True, for MC this has no effect
    fix_Deltat0s : bool
        do not fit nuissance Deltat0s, ttt-fit does not work well with this to True. Default=False
    lgE_shift : bool
        shift the lgE' = lgE_shift * lgE. For example for sys uncertainty lgE_shift=1.14. default=1
```
