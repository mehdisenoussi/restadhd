import numpy as np
from matplotlib import pyplot as pl
import mne

base_path = '/Users/mehdi/work/ghent/side_projects/adhd/data/'
data_path = base_path + 'fooof/'

# loading one MNE epochs object to have channel list and locations
epochs = mne.read_epochs(base_path + 'clean/pp04_stil_clean-epo.fif.gz',verbose = False, preload = False)
chlist = np.array(epochs.ch_names)

conds = np.array(['stil', 'noise', 'toon'])
n_cond = len(conds)
pps = np.array([ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21,
				22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
				39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
				57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72])
n_pp = len(pps)
n_elec = 64

res_all = np.zeros(shape=[3, n_pp, n_elec, n_cond, 2])

for param_set in np.arange(3):
	data_path = base_path + 'fooof/param_set%i/'%(param_set+1)

	for ind_pp, pp in enumerate(pps):
		print('pp %i'%pp)
		for ind_cond, cond in enumerate(conds):
			if file_pres[ind_pp, ind_cond, 1]:
				# {'peak_params_all':peak_params_all, 'aperiodic_params_all':aperiodic_params_all, 'fit_params_all':fit_params_all})
				z = np.load(data_path + 'pp_%i_avg-spectra_all_fooof_params_%s.npz' % (pp, cond), allow_pickle=True)['arr_0'][..., np.newaxis][0]
				res_all[param_set, ind_pp, :, ind_cond, :] = z['fit_params_all'].T



# To get this list of participants I simply sampled 10 values from the array pps
# using this command : np.random.choice(pps, 10, replace=False)
pp_mask = np.array([20, 41, 27, 16,  6, 50, 36, 43, 11, 56])

labels = ['Senoussi et al (2022)', 'Ostlund et al (2022)', 'Lendner et al. (2020)']
elec_name = 'Cz'
elec_ind = int(np.argwhere(chlist==elec_name))
xlims=[[0, .5], [0, 2.5]]
fig, axs = pl.subplots(2, 2)
for param_set in np.arange(3):
	for i in range(2):
		axs[0, i].errorbar(x=np.arange(3)+(param_set/10), y=np.nanmean(res_all[param_set, pp_mask, elec_ind, :, i], axis=0),
			yerr=np.nanstd(res_all[param_set, pp_mask, elec_ind, :, i], axis=0)/np.sqrt(n_pp), fmt='o', label=labels[param_set])
		axs[0, i].set_xticks(range(3))
		axs[0, i].set_xticklabels(conds)
		axs[0, i].set_xlabel('Conditions')
		axs[0, i].set_ylabel('%s'% (['R-squared', 'error'][i]))

		axs[1, i].errorbar(x=np.arange(3)+(param_set/10), y=np.nanmean(res_all[param_set, pp_mask, :, :, i].mean(axis=1), axis=0),
			yerr=np.nanstd(res_all[param_set, pp_mask, :, :, i].mean(axis=1), axis=0)/np.sqrt(n_pp), fmt='o', label=labels[param_set])
		axs[1, i].set_xticks(range(3))
		axs[1, i].set_xticklabels(conds)
		axs[1, i].set_xlabel('Conditions')
		axs[1, i].set_ylabel('%s'% (['R-squared', 'error'][i]))
pl.suptitle('Goodness of fit measures at %s' % elec_name)
pl.legend()






fig, axs = pl.subplots(1, 3)
for ind_cond, cond in enumerate(conds):
	im1, cn = mne.viz.plot_topomap(np.nanmean(res_all[:, :, ind_cond, 0],axis=0), epochs.info, cmap='Greens_r',
		sensors=True, outlines='head',# extrapolate ='local',
		sphere=(0, 0, .038, .145),
		# head_pos = {'scale':[1.3, 1.7]},
		axes=axs[ind_cond], vmin=.93, vmax=.97,
		contours=0)
	axs[ind_cond].set_title(cond, fontsize=10)
	fig.colorbar(im1, ax = axs[ind_cond])
pl.suptitle('R-squared')


fig, axs = pl.subplots(1, 3)
for ind_cond, cond in enumerate(conds):
	im1, cn = mne.viz.plot_topomap(np.nanmean(res_all[:, :, ind_cond, 1],axis=0), epochs.info, cmap='Greens',
		sensors=True, outlines='head',# extrapolate ='local',
		sphere=(0, 0, .038, .145),
		# head_pos = {'scale':[1.3, 1.7]},
		axes=axs[ind_cond], vmin=.05, vmax=.1,
		contours=0)
	axs[ind_cond].set_title(cond, fontsize=10)
	fig.colorbar(im1, ax = axs[ind_cond])

pl.suptitle('error')


