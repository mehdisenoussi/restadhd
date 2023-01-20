import mne, os, glob
import numpy as np
from scipy import signal as sig
from scipy.io import loadmat
from fooof import FOOOFGroup
from matplotlib import pyplot as pl
# from bs4 import BeautifulSoup


base_path = '/Users/mehdi/work/ghent/side_projects/adhd/data/'
data_path = base_path + 'raw/'
save_dir  = base_path + 'clean/'

montage = mne.channels.read_custom_montage(base_path + 'chanlocs_64elecs3.loc')

# data_rej = loadmat(data_path + 'newsept22/PP35_REST_Artifact_Rejection_stil.mat')
# chlist = [ch_i[-1][0] for ch_i in data_rej['Channels'][0]]
# n_elec = len(chlist)
# eegdata_rej = np.array([data_rej[elec_n].squeeze() for elec_n in chlist])


# data_raw = loadmat(data_path + 'newsept22/PP35_REST_Segmentation_in_epochs_stil')
# eegdata_raw = np.array([data_raw[elec_n].squeeze() for elec_n in chlist])



# # Reading the data inside the xml
# # file to a variable under the name
# # data
# with open(data_path + 'newsept22/PP35_REST_Artifact_Rejection_stil.Markers', 'r') as f:
# 	data = f.read()

# # Passing the stored data inside
# # the beautifulsoup parser, storing
# # the returned object
# Bs_data = BeautifulSoup(data, "xml")

# # get the position tags (not sure if 'tag' is correct)
# b_pos = Bs_data.find_all('Position')
# positions = np.array([int(bb.get_text()) for bb in b_pos])



###################################

# one of the missing: PP55_REST_Artifact_Rejection_noise.mat

conds = np.array(['stil', 'noise', 'toon'])

list_mrks = glob.glob(data_path + 'PP*.MRK')
pps_mrks = np.array([int(file_n.split('PP')[1][:2]) for file_n in list_mrks])
file_conds_mrks = np.array([file_n.split('.MRK')[0].split('_')[-1] for file_n in list_mrks])

list_mats = glob.glob(data_path + 'PP*.mat')
pps_mat = np.array([int(file_n.split('PP')[1][:2]) for file_n in list_mats])
file_conds_mat = np.array([file_n.split('.mat')[0].split('_')[-1] for file_n in list_mats])

uniq_pp = np.unique(pps_mrks)
n_pp = len(uniq_pp)
file_pres = np.zeros(shape=(n_pp, len(conds), 2)).astype(np.bool)
for ind_pp, pp in enumerate(uniq_pp):
	for ind_cond, cond in enumerate(conds):
		file_mrk = glob.glob(data_path + 'PP%02i*%s.MRK' % (pp, cond))
		file_pres[ind_pp, ind_cond, 0] = len(file_mrk)>0

		file_mat = glob.glob(data_path + 'PP%02i*%s.mat' % (pp, cond))
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


# # folder = 'newsept22-2/'
# # obs = [5, 6, 7][2]
# for obs in [5, 6, 7]:
# 	print('obs %i'%obs)
# 	for ind_cond, cond in enumerate(conds):


for ind_pp, pp in enumerate(uniq_pp):
	print('pp %i'%pp)
	for ind_cond, cond in enumerate(conds):
		if file_pres[ind_pp, ind_cond, 1]:
			file_mat_n = glob.glob(data_path + 'PP%02i*%s.mat' % (pp, cond))
			data_rej = loadmat(file_mat_n[0])
			chlist = [ch_i[-1][0] for ch_i in data_rej['Channels'][0]]
			n_elec = len(chlist)
			eegdata_rej = np.array([data_rej[elec_n].squeeze() for elec_n in chlist])

			eeg_info = mne.create_info(ch_names=chlist, sfreq = int(data_rej['SampleRate'].squeeze()), ch_types='eeg')
			eeg_epochs = mne.EpochsArray(np.rollaxis(eegdata_rej,1), eeg_info, tmin=0, baseline = None)
			# eeg_epochs.plot(n_epochs = 5, n_channels = 64, scalings = dict(eeg=60), picks='all')

			# data_raw = loadmat(data_path + folder + 'PP%0i_REST_Segmentation_in_epochs_stil')
			# eegdata_raw = np.array([data_raw[elec_n].squeeze() for elec_n in chlist])
			
			if ~((pp == 55) & (cond=='toon')):
				# read MRK files
				file_mrk_n = glob.glob(data_path + 'PP%02i*%s.MRK' % (pp, cond))
				mrk_info = pd.read_csv(file_mrk_n[0], delimiter='\t')
				trials_torej = mrk_info.MarkerType=='Bad_Interval'
				eeg_epochs = eeg_epochs.drop(trials_torej)

			eeg_epochs = eeg_epochs.set_montage(montage)

			eeg_epochs.save(save_dir + 'pp%02i_%s_clean-epo.fif.gz' % (pp, cond), overwrite = True)
			


			# # visual check of rejected segments
			# pl.figure()
			# for i in eegdata_rej[:,trials_torej,:]:
			# 	for j in i:
			# 		pl.plot(j)
			

			########################################################
			# OLD WAY TO GET THE REJECTED TRIALS
			########################################################
			# # Reading the data inside the xml
			# # file to a variable under the name
			# # data
			# file_mrk_n = glob.glob(data_path + 'PP%02i*%s.MRK' % (obs, cond))
			# with open(file_mrk_n[0], 'r') as f:
			# 	data = f.read()

			# # Passing the stored data inside
			# # the beautifulsoup parser, storing
			# # the returned object
			# Bs_data = BeautifulSoup(data, "xml")

			# # get the position tags (not sure if 'tag' is correct)
			# b_pos = Bs_data.find_all('Position')
			# positions = np.array([int(bb.get_text()) for bb in b_pos])

			# newpos = np.array([int(str(i)[:-3])-1 for i in positions])
			# # newpos =[]
			# # for i in positions:
			# # 	ii = str(i)[]
			# print('\tcond %s: %s'% (cond, str(newpos)))
			# print('\t\t%s'%str(positions))




# checking electrodes
montage = mne.channels.read_custom_montage(data_path + 'chanlocs_64elecs.loc')

for ch in montage.ch_names: print('%s is %sin EEG data'%(ch, ['NOT ', ''][int(ch in np.array(chlist))]))

for ch in chlist: print('%s is %sin elec loc file'%(ch, ['NOT ', ''][int(ch in np.array(montage.ch_names))]))












