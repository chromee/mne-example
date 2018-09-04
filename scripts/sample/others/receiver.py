CH_INDEX = list(range(1,17))  # zero-baesd
TIME_INDEX = None # integer or None. None = average of raw values of the current window
SHOW_PSD = False

import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from pycnbi.stream_receiver.stream_receiver import StreamReceiver
import pycnbi.utils.pycnbi_utils as pu
import pycnbi.utils.q_common as qc
mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

amp_name, amp_serial = pu.search_lsl()
sr = StreamReceiver(window_size=1, buffer_size=1, amp_serial=amp_serial, eeg_only=False, amp_name=amp_name)
sfreq = sr.get_sample_rate()
watchdog = qc.Timer()
tm = qc.Timer(autoreset=True)
trg_ch = sr.get_trigger_channel()
last_ts = 0
qc.print_c('Trigger channel: %d' % trg_ch, 'G')

if SHOW_PSD:
    psde = mne.decoding.PSDEstimator(sfreq=sfreq, fmin=1, fmax=50, bandwidth=None, \
        adaptive=False, low_bias=True, n_jobs=1, normalization='length', verbose=None)

sfreq = 512
interval = 1. / sfreq
fig, ax = plt.subplots(1, 1)
time_range = 1
x = np.arange(0, time_range, interval)
y = [0]*(time_range*sfreq)
lines, = ax.plot(x, y)

while True:
    sr.acquire()
    window, tslist = sr.get_window() # window = [samples x channels]
    window = window.T # chanel x samples
    # print(window)

    # # print event values
    # tsnew = np.where(np.array(tslist) > last_ts)[0][0]
    # trigger = np.unique(window[trg_ch, tsnew:])

    # for Biosemi
    # if sr.amp_name=='BioSemi':
    #    trigger= set( [255 & int(x-1) for x in trigger ] )

    # if len(trigger) > 0:
    #     qc.print_c('Triggers: %s' % np.array(trigger), 'G')

    # print('[%.1f] Receiving data...' % watchdog.sec())

    # if TIME_INDEX is None:
    #     datatxt = qc.list2string(np.mean(window[CH_INDEX, :], axis=1), '%-15.6f')
    #     print('[%.3f : %.3f]' % (tslist[0], tslist[1]) + ' data: %s' % datatxt)
    # else:
    #     datatxt = qc.list2string(window[CH_INDEX, TIME_INDEX], '%-15.6f')
    #     print('[%.3f]' % tslist[TIME_INDEX] + ' data: %s' % datatxt)
    # print(window[1])

    x += interval
    y.pop(0)
    y.append(np.mean(window[1]))
    lines.set_data(x, y)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((min(y), max(y)))
    plt.pause(interval)

    # # show PSD
    # if SHOW_PSD:
    #     psd = psde.transform(window.reshape((1, window.shape[0], window.shape[1])))
    #     psd = psd.reshape((psd.shape[1], psd.shape[2]))
    #     psdmean = np.mean(psd, axis=1)
    #     for p in psdmean:
    #         print('%.1f' % p, end=' ')

    last_ts = tslist[-1]
    tm.sleep_atleast(0.05)