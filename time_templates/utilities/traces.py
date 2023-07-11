import numpy as np
from numba import njit


def phase_trace(planetimeres, trace, t0=0, offset=20, nt=600, dt=25 / 3, verbose=True):
    """
    Put traces in phase wrt plane-front time PTR (or t0)
    in ns
    supply t0 (in ns) if offset
    """
    new_trace = np.zeros(nt)
    # new_trace[0] should have t0
    # trace[offset] = ptr
    # need to shift trace to the left by ptr-t0
    trstart = 0
    trend_ = len(trace)

    # bin of original trace where plane should be
    # rounded to nearest int
    bin0 = int(round(planetimeres / dt - t0 / dt - offset))

    if bin0 > nt:  # if ptr is too large
        return new_trace

    if bin0 < 0:  # if ptr is before, shift
        trstart = -bin0
        bin0 = 0
        if trstart > trend_:
            return new_trace

    binend = nt
    trend = trstart + binend - bin0

    if trend > trend_:
        trend = trend_
        binend = bin0 + trend - trstart

    new_trace[bin0:binend] = trace[trstart:trend]

    return new_trace


@njit(fastmath=True)
def make_new_bins_by_cutoff(trace, tracesize, cutoff=5):
    """
    Make new bins so that in every bin the sum is larger
    than cutoff
    """
    thesum = 0
    t0 = tracesize
    new_bins = []
    new_trace = []
    for i in range(tracesize):
        ti = tracesize - 1 - i
        thesum += trace[ti]
        if thesum > cutoff:
            new_bins.append(ti)
            dt = t0 - ti
            if dt == 0:
                dt = 1
            new_trace.append(thesum / dt)
            thesum = 0
            t0 = ti

    if new_bins:
        # what is left over add to first bin, but make sure dt is now correct
        new_bins[-1] = 0
        # multiply entry by previous dt, then add thesum then divide by total dt
        new_trace[-1] *= dt
        new_trace[-1] += thesum

        dt += t0 - ti
        if dt == 0:
            dt = 1
        new_trace[-1] /= dt
    else:
        # no previous entries, so just add 0 and the sum
        new_bins.append(0)
        new_trace.append(thesum / tracesize)

    new_bins.insert(0, tracesize - 1)

    return new_trace[::-1], new_bins[::-1]


# @njit(fastmath=True)
def rebin(trace, old_bins, new_bins):
    digitized = np.digitize(old_bins, new_bins)
    return np.array([trace[digitized == i].mean() for i in range(1, len(new_bins))])
