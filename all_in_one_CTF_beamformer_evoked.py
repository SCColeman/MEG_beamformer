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
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids, inspect_dataset

### need the following line so 3d plotting works, for some reason
mne.viz.set_3d_options(depth_peeling=False, antialias=False)

#%% loading the dataset

bids_root = r'R:\DRS-mTBI\Seb\mTBI_predict\BIDS'

subject = '17359'
session = '01N'
task = 'EmoFace'
run = '01'
suffix = 'meg'

bids_path = BIDSPath(subject=subject, session=session,
task=task, run=run, suffix=suffix, root=bids_root)
data = read_raw_bids(bids_path=bids_path, verbose=False)
print(data.info)

#data.crop(0, 60)   # use this for testing purposes, much quicker

# set stim channel to UPPT002 unless you want button presses for CRT task,
# in which case set stim channel to UPPT001.
events = mne.find_events(data, stim_channel="UPPT002")
mne.viz.plot_events(events)

#%% basic preprocessing

orig_freq = 600
sfreq = 250   # downsampling improves speed in later steps
data_ds = data.copy().resample(sfreq=sfreq)
events[:,0] = np.round(events[:,0] * (sfreq/orig_freq))
data_filt = data_ds.copy().filter(l_freq=1, h_freq=48)

#%% remove bad channels by eye

inspect_dataset(bids_path)

#%% ica to remove artifacts

data_ica = data_filt.copy()
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(data_ica.pick_types(meg=True))
ica.plot_components()
ica.plot_sources(data_ica)

#%% apply ica MAKE SURE YOU MANUALLY CHECK AND CHANGE THE NUMBERS

ica.exclude = [0, 1, 4]
ica.apply(data_ica)

#%% epochs

event_id = [101, 102, 103]
tmin, tmax = -0.4, 0.5
epochs = mne.Epochs(
    data_ica,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(None,-0.1),
    preload=True,
    reject=dict(mag=4e-12))

#%% compute covariance

data_cov = mne.compute_covariance(epochs, tmin=0.1, tmax=0.4, method="empirical")
noise_cov= mne.compute_covariance(epochs, tmin=tmin, tmax=-0.1, method="empirical")
all_cov = mne.compute_covariance(epochs, method="empirical")
data_cov.plot(epochs.info)

#%% forward model, view BEM

subjects_dir = r"R:\DRS-mTBI\Seb\mTBI_predict\FreeSurfer_SUBJECTS"
fs_subject="standard"   # set to "standard" if you haven't done the FS recon

plot_bem_kwargs = dict(
    subject=fs_subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200])

mne.viz.plot_bem(**plot_bem_kwargs)

#%%coreg   SAVE THIS OUT SOMEWHERE WITH -trans.fif at the end

mne.coreg.Coregistration(data_filt.info, fs_subject, subjects_dir=subjects_dir)
mne.gui.coregistration(subjects_dir=subjects_dir, subject=fs_subject)

#%% visualise coreg

# trans is path to -trans.fif file saved out in previous section
trans = r"C:\Users\ppysc6\OneDrive - The University of Nottingham\Documents\python\mTBI_predict\test_files\jess_standard-trans.fif"
info = data_filt.info

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

# pos argument is how many mm side length the voxels are in source space
# can change oct5 to different surface source space
src_vol = mne.setup_volume_source_space(
    fs_subject, pos=8, subjects_dir=subjects_dir, surface=surf_file)
src = mne.setup_source_space(
    fs_subject, spacing="oct5", add_dist=False, subjects_dir=subjects_dir)
src += src_vol
mne.viz.plot_bem(src=src, **plot_bem_kwargs)
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
    mindist=1,
    )
print(fwd)

#%% spatial filter

filters = mne.beamformer.make_lcmv(
    info,
    fwd,
    all_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None)

#%% apply filter

evoked = epochs.average()
stc = mne.beamformer.apply_lcmv(evoked, filters)

#%% visualise surface

brain = stc.surface().plot(
    surface="pial",
    hemi="both",
    src=src,
    views="coronal",
    subjects_dir=subjects_dir,
    brain_kwargs=dict(silhouette=True),
    smoothing_steps=7,
)

#%% visualise volume

stc.volume().plot(src=src, subjects_dir=subjects_dir)

#%% Extract parcellation timecourse

# get label names
parc = "aparc"
labels = mne.read_labels_from_annot(fs_subject, parc=parc, subjects_dir=subjects_dir)

label_ts = mne.extract_label_time_course(
    [stc], labels, src, mode="mean", allow_empty=True
)

# plot the times series of a label
label = 59
fig, axes = plt.subplots(1, layout="constrained")
axes.plot(1e3 * stc.times, label_ts[0][label, :], "k", label=labels[label].name)
axes.set(xlabel="Time (ms)", ylabel="MNE current (nAm)")
axes.legend()



