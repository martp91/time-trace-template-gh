#!/usr/bin/env sh

nevents=1000
key='rcola_Offlinev3r3p4'
for HIM in EPOSLHC QGSII04 SIB23; do
        for energy in 19_19.5 19.5_20; do
                for primary in proton iron; do
                        python fit_events.py $primary $energy -HIM=$HIM -key=$key \
                                --MClgE --fix_lgE --fix_Rmu_timefit \
                                -nevents=$nevents &
                done
                wait
        done
done
