// MP
#include <DetectorGeometry.h>
#include <GenStation.h>
#include <RecEvent.h>
#include <RecEventFile.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <numeric>

#include <TFile.h>
#include <TTree.h>

#include <boost/program_options.hpp>

using namespace std;
using namespace std::chrono;
namespace bpo = boost::program_options;

const double eps = 1e-9; // For log(x+eps)
int TraceOffset = 20;    // take 20 bins before trace

bool UseUniversalityComponents = true; // TODO: this is hardcoded saddd
bool VERBOSE = false;

void MakePMTComponentTrace(vector<double> &trace, double &signal,
                           const SdRecStation station, const ETraceType eTT,
                           const int pmtid, const int signalStartSlot,
                           const int signalEndSlot, const int traceSize,
                           int &nWorkingPmts, const double charge,
                           const double peak) {

  int start = signalStartSlot - TraceOffset;
  if (start < 0) {
    start = 0;
  }

  if (station.HasPMTTraces(eTT, pmtid)) {
    Traces traceObj = station.GetPMTTraces(eTT, pmtid);
    if (traceObj.IsTubeOk()) {
      vector<float> vemtrace = traceObj.GetVEMComponent();
      if (!vemtrace.empty()) {
        nWorkingPmts++; // This appears to be OK like this, not super sure
        int i = 0;
        for (int it = start; it < signalEndSlot; ++it) {
          if (it >= signalStartSlot) {
            // Never get to > signalEndSlot
            signal += vemtrace[it];
          }
          if (i < traceSize) { // if fits in trace, cave seg fault
            trace[i] += vemtrace[it] * peak / charge; // Convert to VEM Charge
          }
          i++;
        }
        signal *= peak / charge;
      }
    }
  }
}

void MakeWCDComponentTrace(vector<double> &trace, double &signal,
                           const SdRecStation station, const ETraceType eTT,
                           const int signalStartSlot, const int signalEndSlot,
                           const int traceSize, const double PMT1Charge,
                           const double PMT2Charge, const double PMT3Charge,
                           const double PMT1Peak, const double PMT2Peak,
                           const double PMT3Peak, int &nWorkingPmts) {

  double PMT1Signal = 0;
  double PMT2Signal = 0;
  double PMT3Signal = 0;

  MakePMTComponentTrace(trace, PMT1Signal, station, eTT, 1, signalStartSlot,
                        signalEndSlot, traceSize, nWorkingPmts, PMT1Charge,
                        PMT1Peak);
  MakePMTComponentTrace(trace, PMT2Signal, station, eTT, 2, signalStartSlot,
                        signalEndSlot, traceSize, nWorkingPmts, PMT2Charge,
                        PMT2Peak);
  MakePMTComponentTrace(trace, PMT3Signal, station, eTT, 3, signalStartSlot,
                        signalEndSlot, traceSize, nWorkingPmts, PMT3Charge,
                        PMT3Peak);

  /*
  if (eTT == eTotalTrace) {
    if ((PMT1Signal <= 0) | (PMT2Signal <= 0) | (PMT3Signal <= 0)) {
      cout << nWorkingPmts << endl;
      cout << PMT1Signal << " " << PMT2Signal << " " << PMT3Signal << endl;
      cout << PMT1Charge << " " << PMT2Charge << " " << PMT3Charge << endl;
    }
  }
  */

  if (nWorkingPmts > 0) {
    for (int i = 0; i < traceSize; i++) {
      trace[i] /= nWorkingPmts;
    }

    signal = (PMT1Signal + PMT2Signal + PMT3Signal) / nWorkingPmts;
  }
}

void ApplySignalComponentCorrection(double &total, double &photon,
                                    double &electron, double &muon,
                                    double &hadron) {
  // Correct component signals
  double totalNoBL = photon + electron + muon + hadron;
  if (totalNoBL > 0) {
    double factor = total / totalNoBL;
    muon *= factor;
    electron *= factor;
    photon *= factor;
    hadron *= factor;
  }
}

void ApplySignalComponentCorrectionUniv(double &total, double &muon,
                                        double &photon_pure,
                                        double &electron_pure,
                                        double &photon_mu, double &electron_mu,
                                        double &photon_had,
                                        double &electron_had, double &hadron) {
  double totalNoBL = muon + photon_pure + electron_pure + photon_mu +
                     electron_mu + photon_had + electron_had + hadron;
  if (totalNoBL > 0) {
    double factor = total / totalNoBL;
    muon *= factor;
    electron_pure *= factor;
    photon_pure *= factor;
    electron_mu *= factor;
    photon_mu *= factor;
    electron_had *= factor;
    photon_had *= factor;
    hadron *= factor;
  }
}

// There is some bug when not getting universality components that the muon
// signal is not correct

void MakeMeanTraces(SdRecStation station, vector<double> &total_trace,
                    vector<double> &muon_trace, double &muon_signal,
                    double PMT1Charge, double PMT2Charge, double PMT3Charge,
                    double PMT1Peak, double PMT2Peak, double PMT3Peak,
                    int &nWorkingPmts) {
  int SignalStartSlot = station.GetSignalStartSlot();
  int SignalEndSlot = station.GetSignalEndSlot();
  int traceSize = SignalEndSlot - SignalStartSlot + TraceOffset;

  vector<double> totalWCDTrace(traceSize, 0);
  vector<double> photonWCDTrace(traceSize, 0);
  vector<double> electronWCDTrace(traceSize, 0);
  vector<double> muonWCDTrace(traceSize, 0);
  vector<double> hadronWCDTrace(traceSize, 0);

  double totalWCDSignal = 0;
  double photonWCDSignal = 0;
  double electronWCDSignal = 0;
  double muonWCDSignal = 0;
  double hadronWCDSignal = 0;

  nWorkingPmts = 0;
  MakeWCDComponentTrace(photonWCDTrace, photonWCDSignal, station, ePhotonTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(electronWCDTrace, electronWCDSignal, station,
                        eElectronTrace, SignalStartSlot, SignalEndSlot,
                        traceSize, PMT1Charge, PMT2Charge, PMT3Charge, PMT1Peak,
                        PMT2Peak, PMT3Peak, nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(muonWCDTrace, muonWCDSignal, station, eMuonTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(hadronWCDTrace, hadronWCDSignal, station, eHadronTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(totalWCDTrace, totalWCDSignal, station, eTotalTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);

  // factor to correct for sum of comps is not total signal due to baseline
  // subtraction etc. Just save total signal and muon signal, then EM is the
  // rest
  ApplySignalComponentCorrection(totalWCDSignal, photonWCDSignal,
                                 electronWCDSignal, muonWCDSignal,
                                 hadronWCDSignal);

  for (int i = 0; i < traceSize; ++i) {
    ApplySignalComponentCorrection(totalWCDTrace[i], photonWCDTrace[i],
                                   electronWCDTrace[i], muonWCDTrace[i],
                                   hadronWCDTrace[i]);
  }

  muon_trace = muonWCDTrace;
  total_trace = totalWCDTrace;
  muon_signal = muonWCDSignal;
  if (VERBOSE) {
    cout << "----------------------" << endl;
    cout << "Total : " << totalWCDSignal << " = "
         << "Muon : " << muon_signal << " + "
         << "photon : " << photonWCDSignal << " + "
         << "em : " << electronWCDSignal << " + "
         << "had : " << hadronWCDSignal << endl;
  }
}

void MakeMeanTracesUniv(SdRecStation station, vector<double> &total_trace,
                        vector<double> &muon_trace,
                        vector<double> &em_pure_trace,
                        vector<double> &em_mu_trace,
                        vector<double> &em_had_trace, double &muon_signal,
                        double &em_pure_signal, double &em_mu_signal,
                        double &em_had_signal, double PMT1Charge,
                        double PMT2Charge, double PMT3Charge, double PMT1Peak,
                        double PMT2Peak, double PMT3Peak, int &nWorkingPmts) {
  if (VERBOSE)
    cout << "Getting WCD Universality comp traces" << endl;
  int SignalStartSlot = station.GetSignalStartSlot();
  int SignalEndSlot = station.GetSignalEndSlot();
  int traceSize = SignalEndSlot - SignalStartSlot + TraceOffset;

  vector<double> totalWCDTrace(traceSize, 0);
  vector<double> photonWCDTrace(traceSize, 0);
  vector<double> electronWCDTrace(traceSize, 0);
  vector<double> muonWCDTrace(traceSize, 0);
  vector<double> hadronWCDTrace(traceSize, 0);

  vector<double> photonMuWCDTrace(traceSize, 0);
  vector<double> electronMuWCDTrace(traceSize, 0);
  vector<double> photonHadWCDTrace(traceSize, 0);
  vector<double> electronHadWCDTrace(traceSize, 0);

  double totalWCDSignal = 0;
  double photonWCDSignal = 0;
  double electronWCDSignal = 0;
  double muonWCDSignal = 0;
  double hadronWCDSignal = 0;
  double photonMuWCDSignal = 0;
  double electronMuWCDSignal = 0;
  double photonHadWCDSignal = 0;
  double electronHadWCDSignal = 0;

  MakeWCDComponentTrace(photonWCDTrace, photonWCDSignal, station, ePhotonTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(electronWCDTrace, electronWCDSignal, station,
                        eElectronTrace, SignalStartSlot, SignalEndSlot,
                        traceSize, PMT1Charge, PMT2Charge, PMT3Charge, PMT1Peak,
                        PMT2Peak, PMT3Peak, nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(muonWCDTrace, muonWCDSignal, station, eMuonTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(hadronWCDTrace, hadronWCDSignal, station, eHadronTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);

  nWorkingPmts = 0;
  MakeWCDComponentTrace(photonMuWCDTrace, photonMuWCDSignal, station,
                        ePhotonFromMuonTrace, SignalStartSlot, SignalEndSlot,
                        traceSize, PMT1Charge, PMT2Charge, PMT3Charge, PMT1Peak,
                        PMT2Peak, PMT3Peak, nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(electronMuWCDTrace, electronMuWCDSignal, station,
                        eElectronFromMuonTrace, SignalStartSlot, SignalEndSlot,
                        traceSize, PMT1Charge, PMT2Charge, PMT3Charge, PMT1Peak,
                        PMT2Peak, PMT3Peak, nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(photonHadWCDTrace, photonHadWCDSignal, station,
                        eShowerLocalHadronPhotonTrace, SignalStartSlot,
                        SignalEndSlot, traceSize, PMT1Charge, PMT2Charge,
                        PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak, nWorkingPmts);
  nWorkingPmts = 0;
  MakeWCDComponentTrace(electronHadWCDTrace, electronHadWCDSignal, station,
                        eShowerLocalHadronElectronTrace, SignalStartSlot,
                        SignalEndSlot, traceSize, PMT1Charge, PMT2Charge,
                        PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak, nWorkingPmts);

  // Has to be last for nWorkingPMTs
  nWorkingPmts = 0;
  MakeWCDComponentTrace(totalWCDTrace, totalWCDSignal, station, eTotalTrace,
                        SignalStartSlot, SignalEndSlot, traceSize, PMT1Charge,
                        PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak, PMT3Peak,
                        nWorkingPmts);

  /*
  if (VERBOSE) {
    cout << "Total : " << totalWCDSignal << " = "
         << "Muon : " << muonWCDSignal << " + "
         << "electron(pure) " << electronWCDSignal << " + "
         << "photon(pure) " << photonWCDSignal << " + electron(mu) "
         << electronMuWCDSignal << " + photon(mu) " << photonMuWCDSignal
         << " + electron(had) " << electronHadWCDSignal << " + photon(had) "
         << photonHadWCDSignal << " + hadron " << hadronWCDSignal << endl;
  }
  */

  // factor to correct for sum of comps is not total signal due to baseline
  // subtraction etc. Just save total signal and muon signal, then EM is the
  // rest
  ApplySignalComponentCorrectionUniv(
      totalWCDSignal, muonWCDSignal, photonWCDSignal, electronWCDSignal,
      photonMuWCDSignal, electronMuWCDSignal, photonHadWCDSignal,
      electronHadWCDSignal, hadronWCDSignal);
  /*
  if (VERBOSE) {
    cout << "Total : " << totalWCDSignal << " = "
         << "Muon : " << muonWCDSignal << " + "
         << "electron(pure) " << electronWCDSignal << " + "
         << "photon(pure) " << photonWCDSignal << " + electron(mu) "
         << electronMuWCDSignal << " + photon(mu) " << photonMuWCDSignal
         << " + electron(had) " << electronHadWCDSignal << " + photon(had) "
         << photonHadWCDSignal << " + hadron " << hadronWCDSignal << endl;
  }
  */

  vector<double> WCDEMPureTrace(traceSize, 0);
  vector<double> WCDEMMuTrace(traceSize, 0);
  vector<double> WCDEMHadTrace(traceSize, 0);

  for (int i = 0; i < traceSize; ++i) {
    ApplySignalComponentCorrectionUniv(
        totalWCDTrace[i], muonWCDTrace[i], photonWCDTrace[i],
        electronWCDTrace[i], photonMuWCDTrace[i], electronMuWCDTrace[i],
        photonHadWCDTrace[i], electronHadWCDTrace[i], hadronWCDTrace[i]);
    WCDEMPureTrace[i] = photonWCDTrace[i] + electronWCDTrace[i];

    WCDEMMuTrace[i] = photonMuWCDTrace[i] + electronMuWCDTrace[i];
    WCDEMHadTrace[i] =
        photonHadWCDTrace[i] + electronHadWCDTrace[i] + hadronWCDTrace[i];
  }

  muon_trace = muonWCDTrace;
  em_pure_trace = WCDEMPureTrace;
  em_mu_trace = WCDEMMuTrace;
  em_had_trace = WCDEMHadTrace;
  total_trace = totalWCDTrace;

  muon_signal = muonWCDSignal;
  em_pure_signal = photonWCDSignal + electronWCDSignal;
  em_mu_signal = photonMuWCDSignal + electronMuWCDSignal;
  em_had_signal = photonHadWCDSignal + electronHadWCDSignal + hadronWCDSignal;
  if (VERBOSE) {
    cout << "----------------------" << endl;
    cout << "Total : " << totalWCDSignal << " = "
         << "Muon : " << muon_signal << " + "
         << "EM(pure) : " << em_pure_signal << " + "
         << "EM(mu) : " << em_mu_signal << " + "
         << "EM(had) : " << em_had_signal << endl;
  }
}

void MakeMeanTracesSSD(SdRecStation station, vector<double> &ssd_total_trace,
                       vector<double> &ssd_muon_trace, double &ssd_muon_signal,
                       double PMT5Charge, double PMT5Peak,
                       double &SSDTotalSignal, double &SSDRiseTime) {

  if (station.HasScintillator()) {
    if (VERBOSE)
      cout << "Getting SSD" << endl;
    SdRecScintillator scintillator = station.GetScintillator();

    SSDRiseTime = scintillator.GetRiseTime();
    SSDTotalSignal = scintillator.GetTotalSignal();

    // Assume ssd start is also wcd start (with offset of course)
    // but if ssdend is later take this
    // caveat is that you might take some ssd signal (in the early part)
    // that is noise, but for now simpler to take just the same start

    int SignalStartSlot = station.GetSignalStartSlot();
    int SignalEndSlot = station.GetSignalEndSlot();
    int SignalSSDStartSlot = scintillator.GetSignalStartSlot();
    if (SignalSSDStartSlot < SignalStartSlot) {
      SignalStartSlot = SignalSSDStartSlot;
    }
    int SignalSSDEndSlot = scintillator.GetSignalEndSlot();
    if (SignalSSDEndSlot > SignalEndSlot) {
      SignalEndSlot = SignalSSDEndSlot;
    }
    SignalStartSlot = SignalSSDStartSlot;
    SignalEndSlot = SignalSSDEndSlot;
    int traceSize = SignalEndSlot - SignalStartSlot + TraceOffset;

    vector<double> totalSSDTrace(traceSize, 0);
    vector<double> photonSSDTrace(traceSize, 0);
    vector<double> electronSSDTrace(traceSize, 0);
    vector<double> hadronSSDTrace(traceSize, 0);
    vector<double> muonSSDTrace(traceSize, 0);

    double totalSSDSignal = 0;
    double photonSSDSignal = 0;
    double electronSSDSignal = 0;
    double hadronSSDSignal = 0;
    double muonSSDSignal = 0;
    int dummy;

    MakePMTComponentTrace(totalSSDTrace, totalSSDSignal, station, eTotalTrace,
                          5, SignalStartSlot, SignalEndSlot, traceSize, dummy,
                          PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(photonSSDTrace, photonSSDSignal, station,
                          ePhotonTrace, 5, SignalStartSlot, SignalEndSlot,
                          traceSize, dummy, PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(electronSSDTrace, electronSSDSignal, station,
                          eElectronTrace, 5, SignalStartSlot, SignalEndSlot,
                          traceSize, dummy, PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(muonSSDTrace, muonSSDSignal, station, eMuonTrace, 5,
                          SignalStartSlot, SignalEndSlot, traceSize, dummy,
                          PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(hadronSSDTrace, hadronSSDSignal, station,
                          eHadronTrace, 5, SignalStartSlot, SignalEndSlot,
                          traceSize, dummy, PMT5Charge, PMT5Peak);

    /*
    cout << "SSD Total " << totalSSDSignal << endl;
    cout << "SSD Muon " << muonSSDSignal << endl;
    cout << "SSD electron " << electronSSDSignal << endl;
    cout << "SSD photon " << photonSSDSignal << endl;
    cout << "SSD hadron  " << hadronSSDSignal << endl;
    */
    // factor to correct for sum of comps is not total signal due to
    // baseline subtraction etc.
    ApplySignalComponentCorrection(totalSSDSignal, photonSSDSignal,
                                   electronSSDSignal, muonSSDSignal,
                                   hadronSSDSignal);

    // Here is bug:
    for (int i = 0; i < traceSize; ++i) {
      ApplySignalComponentCorrection(totalSSDTrace[i], photonSSDTrace[i],
                                     electronSSDTrace[i], muonSSDTrace[i],
                                     hadronSSDTrace[i]);
    }

    ssd_muon_signal = muonSSDSignal;
    ssd_total_trace = totalSSDTrace;
    ssd_muon_trace = muonSSDTrace;
  }
}

void MakeMeanTracesSSDUniv(SdRecStation station, vector<double> &total_trace,
                           vector<double> &muon_trace,
                           vector<double> &em_pure_trace,
                           vector<double> &em_mu_trace,
                           vector<double> &em_had_trace, double &muon_signal,
                           double &em_pure_signal, double &em_mu_signal,
                           double &em_had_signal, double PMT5Charge,
                           double PMT5Peak, double &SSDTotalSignal,
                           double &SSDRiseTime) {
  if (station.HasScintillator()) {
    // cout << "Getting SSD" << endl;
    SdRecScintillator scintillator = station.GetScintillator();

    SSDRiseTime = scintillator.GetRiseTime();
    SSDTotalSignal = scintillator.GetTotalSignal();

    // Assume ssd start is also wcd start (with offset of course)
    // but if ssdend is later take this
    // caveat is that you might take some ssd signal (in the early part)
    // that is noise, but for now simpler to take just the same start

    int SignalStartSlot = station.GetSignalStartSlot();
    int SignalEndSlot = station.GetSignalEndSlot();
    int SignalSSDStartSlot = scintillator.GetSignalStartSlot();
    if (SignalSSDStartSlot < SignalStartSlot) {
      SignalStartSlot = SignalSSDStartSlot;
    }
    int SignalSSDEndSlot = scintillator.GetSignalEndSlot();
    if (SignalSSDEndSlot > SignalEndSlot) {
      SignalEndSlot = SignalSSDEndSlot;
    }
    SignalStartSlot = SignalSSDStartSlot;
    SignalEndSlot = SignalSSDEndSlot;
    int traceSize = SignalEndSlot - SignalStartSlot + TraceOffset;

    vector<double> totalSSDTrace(traceSize, 0);
    vector<double> photonSSDTrace(traceSize, 0);
    vector<double> electronSSDTrace(traceSize, 0);
    vector<double> muonSSDTrace(traceSize, 0);
    vector<double> hadronSSDTrace(traceSize, 0);

    vector<double> photonMuSSDTrace(traceSize, 0);
    vector<double> electronMuSSDTrace(traceSize, 0);
    vector<double> photonHadSSDTrace(traceSize, 0);
    vector<double> electronHadSSDTrace(traceSize, 0);

    double totalSSDSignal = 0;
    double photonSSDSignal = 0;
    double electronSSDSignal = 0;
    double muonSSDSignal = 0;
    double hadronSSDSignal = 0;
    double photonMuSSDSignal = 0;
    double electronMuSSDSignal = 0;
    double photonHadSSDSignal = 0;
    double electronHadSSDSignal = 0;
    int dummy;

    MakePMTComponentTrace(totalSSDTrace, totalSSDSignal, station, eTotalTrace,
                          5, SignalStartSlot, SignalEndSlot, traceSize, dummy,
                          PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(photonSSDTrace, photonSSDSignal, station,
                          ePhotonTrace, 5, SignalStartSlot, SignalEndSlot,
                          traceSize, dummy, PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(electronSSDTrace, electronSSDSignal, station,
                          eElectronTrace, 5, SignalStartSlot, SignalEndSlot,
                          traceSize, dummy, PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(
        photonMuSSDTrace, photonMuSSDSignal, station, ePhotonFromMuonTrace, 5,
        SignalStartSlot, SignalEndSlot, traceSize, dummy, PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(electronMuSSDTrace, electronMuSSDSignal, station,
                          eElectronFromMuonTrace, 5, SignalStartSlot,
                          SignalEndSlot, traceSize, dummy, PMT5Charge,
                          PMT5Peak);
    MakePMTComponentTrace(photonHadSSDTrace, photonHadSSDSignal, station,
                          eShowerLocalHadronPhotonTrace, 5, SignalStartSlot,
                          SignalEndSlot, traceSize, dummy, PMT5Charge,
                          PMT5Peak);
    MakePMTComponentTrace(electronHadSSDTrace, electronHadSSDSignal, station,
                          eShowerLocalHadronElectronTrace, 5, SignalStartSlot,
                          SignalEndSlot, traceSize, dummy, PMT5Charge,
                          PMT5Peak);

    MakePMTComponentTrace(muonSSDTrace, muonSSDSignal, station, eMuonTrace, 5,
                          SignalStartSlot, SignalEndSlot, traceSize, dummy,
                          PMT5Charge, PMT5Peak);
    MakePMTComponentTrace(hadronSSDTrace, hadronSSDSignal, station,
                          eHadronTrace, 5, SignalStartSlot, SignalEndSlot,
                          traceSize, dummy, PMT5Charge, PMT5Peak);

    // factor to correct for sum of comps is not total signal due to
    // baseline subtraction etc.
    ApplySignalComponentCorrectionUniv(
        totalSSDSignal, muonSSDSignal, photonSSDSignal, electronSSDSignal,
        photonMuSSDSignal, electronMuSSDSignal, photonHadSSDSignal,
        electronHadSSDSignal, hadronSSDSignal);

    vector<double> SSDEMPureTrace(traceSize, 0);
    vector<double> SSDEMMuTrace(traceSize, 0);
    vector<double> SSDEMHadTrace(traceSize, 0);

    for (int i = 0; i < traceSize; ++i) {
      ApplySignalComponentCorrectionUniv(
          totalSSDTrace[i], muonSSDTrace[i], photonSSDTrace[i],
          electronSSDTrace[i], photonMuSSDTrace[i], electronMuSSDTrace[i],
          photonHadSSDTrace[i], electronHadSSDTrace[i], hadronSSDTrace[i]);
      SSDEMPureTrace[i] = photonSSDTrace[i] + electronSSDTrace[i];

      SSDEMMuTrace[i] = photonMuSSDTrace[i] + electronMuSSDTrace[i];
      SSDEMHadTrace[i] =
          photonHadSSDTrace[i] + electronHadSSDTrace[i] + hadronSSDTrace[i];
    }
    muon_trace = muonSSDTrace;
    em_pure_trace = SSDEMPureTrace;
    em_mu_trace = SSDEMMuTrace;
    em_had_trace = SSDEMHadTrace;
    muon_signal = muonSSDSignal;
    em_pure_signal = photonSSDSignal + electronSSDSignal;
    em_mu_signal = photonMuSSDSignal + electronMuSSDSignal;
    em_had_signal = photonHadSSDSignal + electronHadSSDSignal + hadronSSDSignal;
    total_trace = totalSSDTrace;
  }
}

int main(int argc, char *argv[]) {
  vector<string> inputFiles;
  string outputFilename = "tree.root";
  bpo::options_description options("Usage");
  options.add_options()("help,h", "Output this help.")(
      "input,i", bpo::value<vector<string>>(&inputFiles), "Input ADST file.")(
      "output,o", bpo::value<string>(&outputFilename), "Output file,");
  bpo::positional_options_description positional;
  positional.add("input", -1);
  bpo::variables_map vm;
  try {
    bpo::store(bpo::command_line_parser(argc, argv)
                   .options(options)
                   .positional(positional)
                   .run(),
               vm);
    bpo::notify(vm);
  } catch (bpo::error &err) {
    cerr << "Command line error : " << err.what() << '\n' << options << endl;
    exit(EXIT_FAILURE);
  }

  if (vm.count("help") || inputFiles.empty()) {
    cerr << options << endl;
    return EXIT_SUCCESS;
  }

  TTree::SetMaxTreeSize(1e12);

  TFile file(outputFilename.c_str(), "recreate");
  TTree tree("tree", "tree with event info");

  long long int EventId;
  vector<int> EventIdsDone;

  string adstFilename;

  int YYMMDD;
  int HHMMSS;
  int GPSSecond;
  int GPSNanoSecond;
  double Pressure;
  double Temperature;
  int HasGDAS;

  int Primary = 0;

  double MClgE;
  double MCXmax;
  double MCXmaxGH;
  double MCXmumax;
  double MCCosTheta;
  double MCCoreTimeS;
  double MCCoreTimeNS;

  double MCStationPlaneFrontTimeS;
  double MCStationPlaneFrontTimeNS;

  TVector3 MCCore;
  TVector3 MCAxis;
  TVector3 SdCore;
  TVector3 SdAxis;

  TVector3 StationPos;

  double FdlgE;
  double FdlgE_err;
  double FdXmax;
  double FdXmax_err;

  double SdlgE;
  double SdCosTheta;
  double SdAzimuth;
  double SdS1000;
  double SdS1000_err;
  double Sdbeta;
  double SdCoreTimeS;
  double SdCoreTimeNS;

  int Is6T5;

  int StationId;
  int LowGainSat;
  int HighGainSat;
  int IsCandidate;
  int IsDense;
  int IsUUB;
  double TimeS;
  double TimeNS;
  double PlaneTimeRes;
  double CurveTimeRes;
  string TriggerName;
  double PMT1Peak;
  double PMT2Peak;
  double PMT3Peak;
  double PMT5Peak;
  double PMT1Charge;
  double PMT2Charge;
  double PMT3Charge;
  double PMT5Charge;
  double PMT1DAratio;
  double PMT2DAratio;
  double PMT3DAratio;
  double PMT5DAratio;
  double PMT1PulseShape;
  double PMT2PulseShape;
  double PMT3PulseShape;

  int nWorkingPmts;

  double Sdr;
  double Sdr_err;
  double Sdpsi;

  double WCDRiseTime;
  double SSDRiseTime;

  double MCr = 0;
  double MCpsi = 0;

  double WCDTotalSignal = 0;
  double SSDTotalSignal = 0;
  double WCDTotalSignal_err;
  double WCDMuonSignal = 0;
  double WCDEMSignal = 0;
  double WCDEMMuSignal = 0;
  double WCDEMHadSignal = 0;
  double SSDMuonSignal = 0;
  double SSDEMSignal = 0;
  double SSDEMMuSignal = 0;
  double SSDEMHadSignal = 0;

  vector<double> wcd_total_trace;
  vector<double> ssd_total_trace;

  vector<double> wcd_muon_trace;
  vector<double> wcd_em_pure_trace;
  vector<double> wcd_em_mu_trace;
  vector<double> wcd_em_had_trace;
  vector<double> ssd_muon_trace;
  vector<double> ssd_em_pure_trace;
  vector<double> ssd_em_mu_trace;
  vector<double> ssd_em_had_trace;

  tree.Branch("adstFilename", &adstFilename);
  tree.Branch("EventId", &EventId);
  tree.Branch("primary", &Primary);
  tree.Branch("MCXmax", &MCXmax);
  tree.Branch("MCXmaxGH", &MCXmaxGH);
  tree.Branch("MCXmumax", &MCXmumax);
  tree.Branch("MClgE", &MClgE);
  tree.Branch("MCCosTheta", &MCCosTheta);
  tree.Branch("FdXmax", &FdXmax);
  tree.Branch("FdXmax_err", &FdXmax_err);
  tree.Branch("FdlgE", &FdlgE);
  tree.Branch("FdlgE_err", &FdlgE_err);
  tree.Branch("SdlgE", &SdlgE);
  tree.Branch("SdCosTheta", &SdCosTheta);
  tree.Branch("SdAzimuth", &SdAzimuth);
  tree.Branch("SdS1000", &SdS1000);
  tree.Branch("SdS1000_err", &SdS1000_err);
  tree.Branch("Sdbeta", &Sdbeta);
  tree.Branch("Is6T5", &Is6T5);
  tree.Branch("SdCoreTimeS", &SdCoreTimeS);
  tree.Branch("SdCoreTimeNS", &SdCoreTimeNS);
  tree.Branch("MCCoreTimeS", &MCCoreTimeS);
  tree.Branch("MCCoreTimeNS", &MCCoreTimeNS);
  tree.Branch("YYMMDD", &YYMMDD);
  tree.Branch("HHMMSS", &HHMMSS);
  tree.Branch("GPSSecond", &GPSSecond);
  tree.Branch("GPSNanoSecond", &GPSNanoSecond);
  tree.Branch("Pressure", &Pressure);
  tree.Branch("Temperature", &Temperature);
  tree.Branch("HasGDAS", &HasGDAS);

  tree.Branch("TraceOffset", &TraceOffset);

  // Adding . behind branch name should give the fX, fY, fZ
  // to bread like MCCore.fX in uproot
  tree.Branch("MCCore.", &MCCore);
  tree.Branch("MCAxis.", &MCAxis);
  tree.Branch("SdCore.", &SdCore);
  tree.Branch("SdAxis.", &SdAxis);
  tree.Branch("StationPos.", &StationPos);

  tree.Branch("IsUUB", &IsUUB);
  tree.Branch("LowGainSat", &LowGainSat);
  tree.Branch("HighGainSat", &HighGainSat);
  tree.Branch("StationId", &StationId);
  tree.Branch("IsCandidate", &IsCandidate);
  tree.Branch("Sdr", &Sdr);
  tree.Branch("Sdr_err", &Sdr_err);
  tree.Branch("Sdpsi", &Sdpsi);
  tree.Branch("TimeS", &TimeS);
  tree.Branch("TimeNS", &TimeNS);
  tree.Branch("PlaneTimeRes", &PlaneTimeRes);
  tree.Branch("CurveTimeRes", &CurveTimeRes);
  tree.Branch("WCDRiseTime", &WCDRiseTime);
  tree.Branch("SSDRiseTime", &SSDRiseTime);
  tree.Branch("TriggerName", &TriggerName);
  tree.Branch("PMT1Charge", &PMT1Charge);
  tree.Branch("PMT2Charge", &PMT2Charge);
  tree.Branch("PMT3Charge", &PMT3Charge);
  tree.Branch("PMT5Charge", &PMT5Charge);
  tree.Branch("PMT1Peak", &PMT1Peak);
  tree.Branch("PMT2Peak", &PMT2Peak);
  tree.Branch("PMT3Peak", &PMT3Peak);
  tree.Branch("PMT5Peak", &PMT5Peak);
  tree.Branch("PMT1DAratio", &PMT1DAratio);
  tree.Branch("PMT2DAratio", &PMT2DAratio);
  tree.Branch("PMT3DAratio", &PMT3DAratio);
  tree.Branch("PMT5DAratio", &PMT5DAratio);
  tree.Branch("PMT1PulseShape", &PMT1PulseShape);
  tree.Branch("PMT2PulseShape", &PMT2PulseShape);
  tree.Branch("PMT3PulseShape", &PMT3PulseShape);
  tree.Branch("nWorkingPmts", &nWorkingPmts);

  tree.Branch("MCr", &MCr);
  tree.Branch("MCpsi", &MCpsi);
  tree.Branch("MCStationPlaneFrontTimeS", &MCStationPlaneFrontTimeS);
  tree.Branch("MCStationPlaneFrontTimeNS", &MCStationPlaneFrontTimeNS);
  tree.Branch("IsDense", &IsDense);

  tree.Branch("WCDTotalSignal", &WCDTotalSignal);
  tree.Branch("WCDTotalSignal_err", &WCDTotalSignal_err);
  tree.Branch("SSDTotalSignal", &SSDTotalSignal);

  tree.Branch("WCDMuonSignal", &WCDMuonSignal);
  tree.Branch("WCDEMSignal", &WCDEMSignal);
  tree.Branch("WCDEMMuSignal", &WCDEMMuSignal);
  tree.Branch("WCDEMHadSignal", &WCDEMHadSignal);
  tree.Branch("SSDMuonSignal", &SSDMuonSignal);
  tree.Branch("SSDEMSignal", &SSDEMSignal);
  tree.Branch("SSDEMMuSignal", &SSDEMMuSignal);
  tree.Branch("SSDEMHadSignal", &SSDEMHadSignal);

  tree.Branch("wcd_total_trace", &wcd_total_trace);
  tree.Branch("ssd_total_trace", &ssd_total_trace);

  tree.Branch("wcd_muon_trace", &wcd_muon_trace);
  tree.Branch("wcd_em_pure_trace", &wcd_em_pure_trace);
  tree.Branch("wcd_em_mu_trace", &wcd_em_mu_trace);
  tree.Branch("wcd_em_had_trace", &wcd_em_had_trace);
  tree.Branch("ssd_muon_trace", &ssd_muon_trace);
  tree.Branch("ssd_em_pure_trace", &ssd_em_pure_trace);
  tree.Branch("ssd_em_mu_trace", &ssd_em_mu_trace);
  tree.Branch("ssd_em_had_trace", &ssd_em_had_trace);

  auto start = high_resolution_clock::now();

  RecEventFile recEventFile(inputFiles);
  RecEvent *recEvent = 0; // will be assigned by root
  recEventFile.SetBuffers(&recEvent);
  DetectorGeometry detectorGeometry;
  recEventFile.ReadDetectorGeometry(detectorGeometry);

  unsigned int nEvents = recEventFile.GetNEvents();
  cout << "Looping events" << endl;

  for (unsigned int i_event = 0; i_event < nEvents; ++i_event) {
    cout << "\r" << i_event << "/" << nEvents;
    cout.flush();
    if (recEventFile.ReadEvent(i_event) != RecEventFile::eSuccess)
      continue;

    // HACK: only works if 1 event per file. Such as for UUB_sims_rec
    if (i_event < inputFiles.size()) {
      adstFilename = inputFiles[i_event]; // Assume same order??
    }

    double FdEnergy = 0;
    double FdEnergyErr = 0;
    FdXmax = 0;
    FdXmax_err = 0;
    double chi2;
    int ndof;

    recEvent->CalculateAverage(RecEvent::eFDs, RecEvent::eEnergy, FdEnergy,
                               FdEnergyErr, chi2, ndof);
    recEvent->CalculateAverage(RecEvent::eFDs, RecEvent::eXmax, FdXmax,
                               FdXmax_err, chi2, ndof);
    if (FdEnergy <= 0) {
      FdEnergy = 1;
    }
    FdlgE = log10(FdEnergy);
    FdlgE_err = FdEnergyErr / (log(10) * FdEnergy);
    //

    const Detector &detector = recEvent->GetDetector();
    Pressure = detector.GetPressure();
    Temperature = detector.GetTemperature();
    HasGDAS = detector.HasGDASDatabase();
    const SDEvent &sdEvent = recEvent->GetSDEvent();
    EventId = sdEvent.GetEventId();
    if (VERBOSE)
      cout << "At event " << EventId << endl;

    if (EventId < 0) {
      cout << "EventId overflow?? " << EventId << endl;
      EventId = -EventId;
    }
    // If EventId is already done than multiply eventid * 10 (should be OK)
    /*
    if (find(EventIdsDone.begin(), EventIdsDone.end(), EventId) !=
    EventIdsDone.end()){ EventId *= 10;
    }

    EventIdsDone.push_back(EventId);
    */
    YYMMDD = sdEvent.GetYYMMDD();
    HHMMSS = sdEvent.GetHHMMSS();
    GPSSecond = sdEvent.GetGPSSecond();
    GPSNanoSecond = sdEvent.GetGPSNanoSecond();

    Is6T5 = sdEvent.Is6T5();
    const SdRecShower sdRecShower = sdEvent.GetSdRecShower();

    const double sdEnergy = sdRecShower.GetEnergy();
    SdS1000 = sdRecShower.GetS1000();
    SdS1000_err = sdRecShower.GetS1000TotalError(); // include sys beta
    Sdbeta = sdRecShower.GetBeta();
    SdlgE = log10(sdEnergy + eps);
    SdCosTheta = sdRecShower.GetCosZenith();
    SdAzimuth = sdRecShower.GetAzimuth();
    SdCoreTimeS = sdRecShower.GetCoreTimeSecond();
    SdCoreTimeNS = sdRecShower.GetCoreTimeNanoSecond();

    SdAxis = sdRecShower.GetAxisSiteCS();
    SdCore = sdRecShower.GetCoreSiteCS();

    const GenShower genShower = recEvent->GetGenShower();

    Primary = genShower.GetPrimary();
    MClgE = log10(genShower.GetEnergy() + eps);
    MCXmax = genShower.GetXmaxInterpolated();
    MCXmaxGH = genShower.GetXmaxGaisserHillas();
    MCCosTheta = genShower.GetCosZenith();
    MCXmumax = genShower.GetXmaxMu();
    /*
    cout << "--------------------" << endl;
    cout << MCXmumax << endl;
    cout << endl;
    */

    MCCore = genShower.GetCoreSiteCS();
    MCAxis = genShower.GetAxisSiteCS();
    MCCoreTimeS = genShower.GetCoreTimeSecond();
    MCCoreTimeNS = genShower.GetCoreTimeNanoSecond();

    const vector<SdRecStation> sdStations = sdEvent.GetStationVector();
    if (VERBOSE)
      cout << "Looping sd stations" << endl;

    for (auto const &station : sdStations) {
      /*
      if (!station.IsCandidate()) {continue;}
      if (station.IsLowGainSaturated()) {continue; }
      */

      StationId = station.GetId();
      if (VERBOSE)
        cout << "At Station " << StationId << endl;
      IsCandidate = station.IsCandidate();

      if (detectorGeometry.IsKnown(StationId)) {
        StationPos = detectorGeometry.GetStationPosition(StationId);
      } else {
        StationPos = detector.GetStationPosition(StationId);
      }

      MCr = detectorGeometry.GetStationAxisDistance(StationPos, MCAxis, MCCore);
      MCpsi = detectorGeometry.GetStationShowerPolarAngle(
          StationPos, MCAxis, MCCore); // no more cos

      HighGainSat = station.IsHighGainSaturated();
      LowGainSat = station.IsLowGainSaturated();
      IsDense = station.IsDense();
      IsUUB = station.IsUUB();
      TimeS = station.GetTimeSecond();
      TimeNS = station.GetTimeNSecond();
      PlaneTimeRes = station.GetPlaneTimeResidual();
      CurveTimeRes = station.GetCurvatureTimeResidual();

      Sdr = station.GetSPDistance();
      Sdr_err = station.GetSPDistanceError();
      Sdpsi = station.GetAzimuthSP();

      WCDRiseTime = station.GetRiseTime();
      TriggerName = station.GetStationTriggerName();

      if (sdEvent.HasGenStations()) {
        // if not simulations, will give seg fault
        const GenStation *genStation = sdEvent.GetSimStationById(StationId);
        MCStationPlaneFrontTimeS = genStation->GetPlaneFrontTimeSecond();
        MCStationPlaneFrontTimeNS = genStation->GetPlaneFrontTimeNSecond();
      }
      PMT1Charge = station.GetCharge(1);
      PMT2Charge = station.GetCharge(2);
      PMT3Charge = station.GetCharge(3);
      PMT5Charge = station.GetCharge(5);
      PMT1Peak = station.GetPeak(1);
      PMT2Peak = station.GetPeak(2);
      PMT3Peak = station.GetPeak(3);
      PMT5Peak = station.GetPeak(5);
      PMT1DAratio = station.GetDynodeAnodeRatio(1);
      PMT2DAratio = station.GetDynodeAnodeRatio(2);
      PMT3DAratio = station.GetDynodeAnodeRatio(3);
      PMT5DAratio = station.GetDynodeAnodeRatio(5);

      if (station.HasPMTTraces(eTotalTrace, 1)) {
        PMT1PulseShape =
            station.GetPMTTraces(eTotalTrace, 1).GetMuonPulseDecayTime();
      }
      if (station.HasPMTTraces(eTotalTrace, 2)) {
        PMT2PulseShape =
            station.GetPMTTraces(eTotalTrace, 2).GetMuonPulseDecayTime();
      }
      if (station.HasPMTTraces(eTotalTrace, 3)) {
        PMT3PulseShape =
            station.GetPMTTraces(eTotalTrace, 3).GetMuonPulseDecayTime();
      }

      if (VERBOSE)
        cout << "Getting traces" << endl;

      // Sets traces and signal
      if (UseUniversalityComponents) {
        MakeMeanTracesUniv(station, wcd_total_trace, wcd_muon_trace,
                           wcd_em_pure_trace, wcd_em_mu_trace, wcd_em_had_trace,
                           WCDMuonSignal, WCDEMSignal, WCDEMMuSignal,
                           WCDEMHadSignal, PMT1Charge, PMT2Charge, PMT3Charge,
                           PMT1Peak, PMT2Peak, PMT3Peak, nWorkingPmts);

        MakeMeanTracesSSDUniv(station, ssd_total_trace, ssd_muon_trace,
                              ssd_em_pure_trace, ssd_em_mu_trace,
                              ssd_em_had_trace, SSDMuonSignal, SSDEMSignal,
                              SSDEMMuSignal, SSDEMHadSignal, PMT5Charge,
                              PMT5Peak, SSDTotalSignal, SSDRiseTime);

      } else {
        MakeMeanTraces(station, wcd_total_trace, wcd_muon_trace, WCDMuonSignal,
                       PMT1Charge, PMT2Charge, PMT3Charge, PMT1Peak, PMT2Peak,
                       PMT3Peak, nWorkingPmts);
        // SSD
        MakeMeanTracesSSD(station, ssd_total_trace, ssd_muon_trace,
                          SSDMuonSignal, PMT5Charge, PMT5Peak, SSDTotalSignal,
                          SSDRiseTime);
      }

      WCDTotalSignal = station.GetTotalSignal();
      WCDTotalSignal_err = station.GetTotalSignalError();

      // fill for every station
      if (VERBOSE)
        cout << "Filling tree" << endl;
      tree.Fill();
    } // end station loop
    if (VERBOSE)
      cout << "End event" << endl;
  }
  cout << endl;
  cout << "Writing and closing " << outputFilename << endl;

  tree.Write();

  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);

  cout << "Time spend: " << duration.count() / 1e6 << endl;

  return EXIT_SUCCESS;
}
