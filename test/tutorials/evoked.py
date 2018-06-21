import os.path as op
import mne

mne.set_log_level('WARNING')

# Evokedデータ構造は，主に試行を通して平均化されたデータを格納するために使用される．

data_path = mne.datasets.sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')

# evokeds = mne.read_evokeds(fname, baseline=(None, 0), proj=True)
# print(evokeds)

evoked = mne.read_evokeds(fname, condition='Left Auditory')
evoked.pick_types(meg=False, eeg=True, eog=False)
evoked.apply_baseline((None, 0)).apply_proj()
# print(evoked)


# print(evoked.info)
# print(evoked.times)

# print(evoked.nave)  # Number of averaged epochs.
# print(evoked.first)  # First time sample.
# print(evoked.last)  # Last time sample.
# print(evoked.comment)  # Comment on dataset. Usually the condition.
# print(evoked.kind)  # Type of data, either average or standard_error.

# data = evoked.data
# print(data.shape)

# print('Data from channel {0}:'.format(evoked.ch_names[315]))
# print(data[315])

# 呼び出されたデータを他のシステムからインポートしたい場合は、
# その配列をnumpy配列にして、mne.EvokedArrayを使用することができる
# evoked = mne.EvokedArray(data, evoked.info, tmin=evoked.times[0])
# evoked.plot(time_unit='s')
