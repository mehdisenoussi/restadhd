# get one elec
elec_name = 'Cz'
elec_ind = int(np.argwhere(chlist==elec_name))


# make csv files
col_names = ['participant', 'condition', 'aper_slope', 'aper_offset']
# make pandas dtaframe
df_arr = np.zeros(shape=[n_pp*n_cond, 4], dtype='<U18')
k = 0
for ind, pp in enumerate(pps):
	for j in np.arange(n_cond):
		# one elec
		# df_arr[k, :] = pp, conds[j], '%.4f'%res_all[ind, elec_ind, j, 1], '%.4f'%res_all[ind, elec_ind, j, 0]
		
		# avg across elecs
		df_arr[k, :] = pp, conds[j], '%.4f'%res_all[ind, :, j, 1].mean(), '%.4f'%res_all[ind, :, j, 0].mean()
		k+=1

df = pd.DataFrame(data = df_arr, columns=col_names)
# df.to_csv('aperiodic_data_pp4_to_72_elec%s.csv' % elec_name)
df.to_csv('aperiodic_data_pp4_to_72_elecAllAvg.csv')
