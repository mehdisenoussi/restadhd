import numpy as np
from matplotlib import pyplot as pl

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
				57, 58])
n_pp = len(pps)
n_elec = 64

file_pres = np.load(base_path+'file_present_05-01-22.npy', allow_pickle=True)
uniq_pp = np.load(base_path+'uniq_pp_list_05-01-22.npy', allow_pickle=True)

res_all = np.zeros(shape=[n_pp, n_elec, n_cond, 2])-1

for ind_pp, pp in enumerate(pps):
	print('pp %i'%pp)
	for ind_cond, cond in enumerate(conds):
		if file_pres[ind_pp, ind_cond, 1]:
			# {'peak_params_all':peak_params_all, 'aperiodic_params_all':aperiodic_params_all, 'fit_params_all':fit_params_all})
			z = np.load(data_path + 'pp_%i_avg-spectra_all_fooof_params_%s.npz' % (pp, cond), allow_pickle=True)['arr_0'][..., np.newaxis][0]
			res_all[ind_pp, :, ind_cond, :] = z['aperiodic_params_all'].T

res_all[res_all==-1] = np.nan


elec_name = 'Fz'
elec_ind = int(np.argwhere(chlist==elec_name))
xlims=[[0, .5], [0, 2.5]]
fig, axs = pl.subplots(1, 2)
for i in range(2):
	axs[i].errorbar(x=[0, 1, 2], y=np.nanmean(res_all[:, elec_ind, :, i], axis=0),
		yerr=np.nanstd(res_all[:, elec_ind, :, i], axis=0)/np.sqrt(n_pp), fmt='o')
	axs[i].set_xticks(range(3))
	axs[i].set_xticklabels(conds)
	axs[i].set_xlabel('Conditions')
	axs[i].set_ylabel('Aperiodic %s'% (['offset', 'exponent (slope)'][i]))
pl.suptitle('Aperiodic parameters at %s' % elec_name)






fig, axs = pl.subplots(1, 3)
for ind_cond, cond in enumerate(conds):
	im1, cn = mne.viz.plot_topomap(np.nanmean(res_all[:, :, ind_cond, 0],axis=0), epochs.info, cmap='Greens_r',
		sensors=True, outlines='head',# extrapolate ='local',
		sphere=(0, 0, .038, .145),
		# head_pos = {'scale':[1.3, 1.7]},
		axes=axs[ind_cond], vmin=.5, vmax=1.5,
		contours=0)
	axs[ind_cond].set_title(cond, fontsize=10)
	fig.colorbar(im1, ax = axs[ind_cond])
