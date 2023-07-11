import os
import argparse
import uproot
import numpy as np
from re import findall
import tables as tb
from math import ceil, floor
from time_templates.utilities.Xmax_MC_moments import Xmax_mean_lgE

def split_key(key):
    "Get info from the tree name for file reading root, regex magic"

    try:
        lgE, ct, Xmax, _ = findall(r"[-+]?\d*\.\d+|\d+", key)
    except ValueError:
        print(key)
    if 'electron' in key:
        ptype = 'electron'
    elif 'photon' in key:
        ptype = 'photon'
    else:
        ptype = 'muon'
    return float(lgE), float(ct), float(Xmax), ptype


parser = argparse.ArgumentParser()
parser.add_argument('input_dir', help='input directory where trees are')
parser.add_argument('output', help='output h5file')

args = parser.parse_args()

directory = args.input_dir
output = args.output
print("Getting files from", directory)
# Get list of files
print("Getting list of files")
rootfiles = []
for subdir, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.root'):
            try:
                hist = uproot.open(os.path.join(subdir, file))
            except OSError:
                continue
            rootfiles.append(os.path.join(subdir, file))

nfiles = len(rootfiles)
print("Number of files = ", nfiles)
#rootfiles = rootfiles[:10]

hists = uproot.open(rootfiles[0])

h, r_bins, psi_bins, lgE_pkin_bins = hists[hists.keys()[0]].to_numpy(flow=True)
nr, npsi, nE = h.shape


h5file = tb.open_file(output, mode="w")
bin_group = h5file.create_group('/', 'bins',
                        'corresponding bins for the histograms. Note that psi, lgEpkin, t \
                        contain -np.inf and np.inf as overflow bins but r does not!')
h5file.create_array(bin_group, 'r_bins', r_bins, "bins in r, note NO overflow bins")
h5file.create_array(bin_group, 'psi_bins', psi_bins, "bins in psi, note overflow bins")
h5file.create_array(bin_group, 'lgE_pkin_bins', lgE_pkin_bins, "bins in lgE of particles kinetic energy, \
note overflow bins")

event_table = h5file.create_table('/', 'event', {'lgE': tb.FloatCol(),
                                       'Xmax': tb.FloatCol(),
                                       'cosTheta': tb.FloatCol(),
                                       'it': tb.IntCol(),
                                       'ID': tb.IntCol(),
                                       'file': tb.StringCol(100)}, expectedrows=nfiles)

muon_spec = h5file.create_carray('/', 'muon_spec', tb.Int32Atom(), shape=(nfiles, nr, npsi, nE),
                                       filters=tb.Filters(1))
electron_spec = h5file.create_carray('/', 'electron_spec', tb.Int32Atom(), shape=(nfiles, nr, npsi, nE),
                                       filters=tb.Filters(1))
photon_spec = h5file.create_carray('/', 'photon_spec', tb.Int32Atom(), shape=(nfiles, nr, npsi, nE),
                                       filters=tb.Filters(1))

row = event_table.row
ifile = 0
for file in rootfiles:
#    print(file)
    try:
        hist = uproot.open(file)
    except OSError:
        continue
    filename = file.split('/')[-1]
    print(filename)
    try:
        ID = int(filename.split('.')[0].split('_')[0][3:])
    except ValueError:
        #does not work for Aab showers
        ID = 0


    for key in hist.iterkeys(filter_name='muon*'):
        try:
            lgE, ct, Xmax, ptype = split_key(key)
            muon_spec[ifile, ...] = hist[key].values(flow=True)
            electron_key = 'electron_' + '_'.join(key.split('_')[1:])
            electron_spec[ifile, ...] = hist[electron_key].values(flow=True)
            photon_key = 'photon_' + '_'.join(key.split('_')[1:])
            photon_spec[ifile, ...] = hist[photon_key].values(flow=True)
        except:
            continue

    row['Xmax'] = Xmax
    row['lgE'] = lgE
    row['cosTheta'] = ct
    row['ID'] = ID
    row['it'] = ifile
    row['file'] = filename
    ifile += 1
    row.append()

event_table.flush()
h5file.close()


