import mne, glob
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as pl
from pathlib import Path

base_path = Path('/Users/mehdi/work/ghent/side_projects/adhd/data/')
data_path = base_path / 'raw'
save_dir  = base_path / 'clean'

montage = mne.channels.read_custom_montage(base_path / 'chanlocs_64elecs3.loc')


###################################

# one of the missing: PP55_REST_Artifact_Rejection_noise.mat

conds = np.array(['stil', 'noise', 'toon'])

list_mrks = glob.glob((data_path / 'PP*.MRK').as_posix())
pps_mrks = np.array([int(file_n.split('PP')[1][:2]) for file_n in list_mrks])
file_conds_mrks = np.array([file_n.split('.MRK')[0].split('_')[-1] for file_n in list_mrks])

list_mats = glob.glob((data_path / 'PP*.mat').as_posix())
pps_mat = np.array([int(file_n.split('PP')[1][:2]) for file_n in list_mats])
file_conds_mat = np.array([file_n.split('.mat')[0].split('_')[-1] for file_n in list_mats])

uniq_pp = np.unique(pps_mrks)
n_pp = len(uniq_pp)
file_pres = np.zeros(shape=(n_pp, len(conds), 2)).astype(np.bool)
for ind_pp, pp in enumerate(uniq_pp):
	for ind_cond, cond in enumerate(conds):
		file_mrk = glob.glob((data_path / ('PP%02i*%s.MRK' % (pp, cond))).as_posix())
		file_pres[ind_pp, ind_cond, 0] = len(file_mrk)>0

		file_mat = glob.glob((data_path / ('PP%02i*%s.mat' % (pp, cond))).as_posix())
		file_pres[ind_pp, ind_cond, 1] = len(file_mat)>0


fig, axs = pl.subplots(2, 1)
for i in range(2):
	axs[i].imshow(file_pres[:,:,i].T, origin='lower', aspect=2, cmap='Greens', vmin=0, vmax=1)
	axs[i].set_xticks(np.arange(n_pp)-.5)
	axs[i].set_xticklabels(uniq_pp)
	axs[i].set_yticks(np.arange(3)-.5)
	axs[i].set_yticklabels(conds)
	axs[i].set_title('%s files' % ['Marker', 'Mat'][i])
	axs[i].grid()


for ind_pp, pp in enumerate(uniq_pp):
	print('pp %i'%pp)
	for ind_cond, cond in enumerate(conds):
		if file_pres[ind_pp, ind_cond, 1]:
			file_mat_n = glob.glob((data_path / ('PP%02i*%s.mat' % (pp, cond))).as_posix())
			data_rej = loadmat(file_mat_n[0])
			chlist = [ch_i[-1][0] for ch_i in data_rej['Channels'][0]]
			n_elec = len(chlist)
			eegdata_rej = np.array([data_rej[elec_n].squeeze() for elec_n in chlist])

			eeg_info = mne.create_info(ch_names=chlist, sfreq = int(data_rej['SampleRate'].squeeze()), ch_types='eeg')
			eeg_epochs = mne.EpochsArray(np.rollaxis(eegdata_rej,1), eeg_info, tmin=0, baseline = None)
			
			if ~((pp == 55) & (cond=='toon')):
				# read MRK files
				file_mrk_n = glob.glob((data_path / ('PP%02i*%s.MRK' % (pp, cond))).as_posix())
				mrk_info = pd.read_csv(file_mrk_n[0], delimiter='\t')
				trials_torej = mrk_info.MarkerType=='Bad_Interval'
				eeg_epochs = eeg_epochs.drop(trials_torej)

			eeg_epochs = eeg_epochs.set_montage(montage)

			eeg_epochs.save(save_dir / ('pp%02i_%s_clean-epo.fif.gz' % (pp, cond)), overwrite = True)





