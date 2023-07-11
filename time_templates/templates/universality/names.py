# trace names are like: wcd_{comp}_traces, with comp as below
eMUON = "muon"
eEM_PURE = "em"
eEM_MU = "em_mu"
eEM_HAD = "em_had"

# TOTAL SIGNAL, trace names are different. Srry
DICT_COMP_SIGNALKEY = {
    eMUON: "WCDMuonSignal",
    eEM_PURE: "WCDEMSignal",
    eEM_MU: "WCDEMMuSignal",
    eEM_HAD: "WCDEMHadSignal",
}


DICT_COMP_COLORS = {
    "total": "k",
    eMUON: "#1D5287",
    eEM_PURE: "#BC313D",
    eEM_MU: "#4B9444",
    eEM_HAD: "#EAB244",
    "light": "#DAD884",
}


DICT_COMP_LABELS = {
    eMUON: "\\mu",
    eEM_PURE: "e\\gamma",
    eEM_MU: "e\\gamma(\\mu)",
    eEM_HAD: "e\\gamma(\\rm had)",
}

DICT_COMP_MARKERS = {eMUON: "o", eEM_PURE: "s", eEM_MU: "v", eEM_HAD: "X"}
