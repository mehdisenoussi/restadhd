import mne, os
import numpy as np
from scipy import signal as sig
from scipy.io import loadmat
from fooof import FOOOFGroup

base_path = '/Users/mehdi/work/ghent/side_projects/adhd/data/'
data_path = base_path + 'clean/'
save_path  = base_path + 'fooof/'
if not os.path.exists(save_dir): os.mkdir(save_dir)

conds = np.array(['stil', 'noise', 'toon'])
# obs_all = np.array([53])
pps = np.array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
				22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
				39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
				57, 58])

file_pres = np.load(base_path+'file_present_20-01-22.npy', allow_pickle=True)
uniq_pp = np.load(base_path+'uniq_pp_list_20-01-22.npy', allow_pickle=True)

# for obs_ind, obs_i in enumerate(pps):
	
# 	for cond_i in conds:

for ind_pp, pp in enumerate(pps):
	print('pp %i'%pp)
	for ind_cond, cond in enumerate(conds):
		if file_pres[ind_pp, ind_cond, 1]:
			epochs = mne.read_epochs(data_path + 'pp%02i_%s_clean-epo.fif.gz' % (pp, cond),
								verbose = False, preload = True)
			eegdata = epochs.get_data()

			chlist = np.array(epochs.ch_names)
			n_elec = len(chlist)

			# data = loadmat(data_path + 'pp%i/raw/PP53_REST_Segmentation_%s.mat' % (obs_i, cond_i))
			# chlist = [ch_i[-1][0] for ch_i in data['Channels'][0]]
			# n_elec = len(chlist)
			# eegdata = np.array([data[elec_n].squeeze() for elec_n in chlist])

			freqs_welch, amp_welch = sig.welch(x = eegdata, fs = 500, window='hann', average = 'mean', nperseg=1000, axis=-1)
			amp_welch_mean = amp_welch.mean(axis=0)

			fg = FOOOFGroup(verbose = False, max_n_peaks = 6, peak_width_limits = [.5, 2])
			fg.fit(freqs_welch, amp_welch_mean, [1, 35], n_jobs=-1)

			peaks_elec_num = np.array([])
			aperiodic_params_all = np.zeros(shape=[2, n_elec])
			fit_params_all = np.zeros(shape=[2, n_elec])
			for elec_n in np.arange(n_elec):
				if len(fg.get_fooof(elec_n).peak_params_) > 0:
					peaks = np.concatenate([fg.get_fooof(elec_n).peak_params_,
						fg.get_fooof(elec_n).gaussian_params_], axis=1)
					if len(peaks_elec_num) == 0:
						peak_params = peaks
						peaks_elec_num = np.repeat(elec_n, peaks.shape[0])
					else:
						peak_params = np.concatenate([peak_params, peaks], axis=0)
						peaks_elec_num = np.concatenate([peaks_elec_num, np.repeat(elec_n, peaks.shape[0])])
				
				# returns [offset, exponent]
				aperiodic_params_all[:, elec_n] = fg.get_fooof(elec_n).get_results().aperiodic_params
				fit_params_all[:, elec_n] = fg.get_fooof(elec_n).get_results().r_squared, fg.get_fooof(elec_n).get_results().error

			# peak stuff
			# peak_params_all = np.concatenate([peaks_elec_num[:, np.newaxis],
			# 		np.repeat(elec_n, len(peak_params))[:, np.newaxis], peak_params], axis=1)
			
			peak_params_all = np.concatenate([peaks_elec_num[:, np.newaxis], peak_params], axis=1)

			
			np.savez(save_dir + 'pp_%i_avg-spectra_all_fooof_params_%s.npz' % (pp, cond),
				{'peak_params_all':peak_params_all, 'aperiodic_params_all':aperiodic_params_all,
				'fit_params_all':fit_params_all})







