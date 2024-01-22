# -*- coding: utf-8 -*-
"""
A script to perform pre-processing, forward modelling and beamforming on
Nottingham mTBI-predict data, or any CTF data. The script assumes that the 
data has already been converted to BIDS format. The script also assumes that
you have a FreeSurfer anatomical reconstruction for the subject.

@author: Sebastian C. Coleman, ppysc6@nottingham.ac.uk
"""

import os
import os.path as op

import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, inspect_dataset

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% loading the dataset

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'

# scanning session info
subject = '17359'
session = '01N'
task = 'CRT'  # name of the task
run = '01'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=bids_root)
data = read_raw_bids(bids_path=bids_path, verbose=False)
print(data.info)

#data.crop(0, 60)    # uncomment for quick testing

#events = mne.find_events(data, stim_channel="UPPT002")
events = mne.find_events(data, stim_channel="UPPT001")   # for button press
mne.viz.plot_events(events)

#%% basic preprocessing

data_3rd = data.copy().apply_gradient_compensation(grade=3)
orig_freq = 600
sfreq = 250
data_ds = data_3rd.resample(sfreq=sfreq)
events[:,0] = np.round(events[:,0] * (sfreq/orig_freq))
data_filt = data_ds.copy().filter(l_freq=1, h_freq=48)

#%% remove bad channels

inspect_dataset(bids_path)

#%% ica to remove artifacts

data_ica = data_filt.copy()
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(data_ica.pick_types(meg=True))
ica.plot_components()
ica.plot_sources(data_ica)

#%% apply ica

ica.exclude = [0, 10]   # CHANGE THESE TO ECG/CARDIAC COMPONENTS
ica.apply(data_ica)

#%% epochs

event_id = 32     # trigger of interest
fband = [13, 30]
tmin, tmax = -0.5, 1.5
epochs = mne.Epochs(
    data_ica,
    events,
    event_id,
    tmin,
    tmax,
    preload=True,
    reject=dict(mag=4e-12))

epochs_filt = epochs.filter(fband[0], fband[1]).copy()

#%% compute covariance

active_cov = mne.compute_covariance(epochs_filt, tmin=0.3, tmax=0.8, method="shrunk")
control_cov= mne.compute_covariance(epochs_filt, tmin=-0.3, tmax=0.2, method="shrunk")
all_cov = mne.compute_covariance(epochs_filt, method="shrunk")
all_cov.plot(epochs_filt.info)

#%% forward model, view BEM

subjects_dir = r"R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS"
fs_subject="standard" # CHANGE THIS TO YOUR SUBJECT IF YOU HAVE THE FS RECON

plot_bem_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200])

mne.viz.plot_bem(**plot_bem_kwargs)

#%% coreg

#mne.coreg.Coregistration(data_filt.info, fs_subject, subjects_dir=subjects_dir)
mne.gui.coregistration(subjects_dir=subjects_dir, subject=fs_subject, scale_by_distance=False)

#%% visualise coreg

trans = r"C:\Users\ppysc6\OneDrive - The University of Nottingham\Documents\python\mTBI_predict\test_files\jess_standard-trans.fif"
info = epochs_filt.info

mne.viz.plot_alignment(
    info,
    trans,
    subject=fs_subject,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=subjects_dir,
    surfaces="head-dense")

#%% compute source space

surf_file = op.join(subjects_dir, fs_subject, "bem", "brain.surf")

# can change oct5 to other surface source space
src = mne.setup_source_space(
    fs_subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir)
src.plot(subjects_dir=subjects_dir)

#%% forward solution

conductivity = (0.3,)
model = mne.make_bem_model(
    subject=fs_subject, ico=4,
    conductivity=conductivity,
    subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(
    info,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=0,
    )
print(fwd)

#%% spatial filter

filters = mne.beamformer.make_lcmv(
    info,
    fwd,
    all_cov,
    reg=0.05,
    noise_cov=None,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None)

#%% apply filter to covariance for static sourcemap
 
stc_active = mne.beamformer.apply_lcmv_cov(active_cov, filters)
stc_base = mne.beamformer.apply_lcmv_cov(control_cov, filters)
 
stc_change = (stc_active - stc_base) / (stc_active + stc_base)
 
#%% visualise static sourcemap
 
stc_change.plot(src=src, subject=fs_subject,
                subjects_dir=subjects_dir, surface="pial", hemi="both")

#%% apply beamformer to filtered raw data for timecourse extraction
 
stc_raw = mne.beamformer.apply_lcmv_raw(data_ica, filters)
 
#%% extract absolute max voxel TFS/timecourse
 
peak = stc_change.get_peak(mode="abs", vert_as_index=True)[0]
 
stc_peak = stc_raw.data[peak]

# make fake raw object from source time course
 
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=info["sfreq"],
                              ch_types=ch_types)
source_raw = mne.io.RawArray([stc_peak], source_info)
 
source_epochs = mne.Epochs(
    source_raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True)
 
 
# TFR
 
freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs/2
power = mne.time_frequency.tfr_morlet(source_epochs, freqs=freqs, n_cycles=n_cycles,
                                           use_fft=True, picks="all"
                                           )
power[0].plot(picks="all", baseline=(1, 1.5))

### timecourse

source_epochs_filt = source_epochs.filter(fband[0], fband[1], picks="all").copy()
source_epochs_filt.apply_hilbert(envelope=True, picks="all")
stc_epochs_filt = source_epochs_filt.average(picks="all")
stc_epochs_filt.plot()

#%% extract maximum from within a parcel

# get label names
parc = "aparc"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

# get induced peak within label
label = 44   # left motor
stc_inlabel = stc_change.in_label(labels[label])
label_peak = stc_inlabel.get_peak(mode="abs", vert_as_index=True)[0]

# extract timecourse of peak
stc_label_all = mne.extract_label_time_course(stc_raw, labels[label], src, mode=None)
stc_label_peak = stc_label_all[0][label_peak,:]

# make fake raw object from source time course
ch_names = ["peak"]
ch_types = ["misc"]
source_info = mne.create_info(ch_names=ch_names, sfreq=info["sfreq"],
                              ch_types=ch_types)
source_label_raw = mne.io.RawArray([stc_label_peak], source_info)
 
source_label_epochs = mne.Epochs(
    source_label_raw,
    events=events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=None,
    preload=True)
 
 
# TFR
freqs = np.logspace(*np.log10([6, 35]), num=20)
n_cycles = freqs/2
power = mne.time_frequency.tfr_morlet(source_label_epochs, freqs=freqs, n_cycles=n_cycles,
                                           use_fft=True, picks="all")
power[0].plot(picks="all", baseline=(1, 1.5))

### timecourse
source_label_filt = source_label_epochs.filter(fband[0], fband[1], picks="all").copy()
source_label_filt.apply_hilbert(envelope=True, picks="all")
stc_label_filt = source_label_filt.average(picks="all")
stc_label_filt.plot()
