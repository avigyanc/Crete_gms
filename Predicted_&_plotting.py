#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 09:25:52 2021

@author: chris
"""

import matplotlib.pyplot as plt
import rupture_functions as rup


import pandas as pd
import numpy as np

slip_model_file = '/Users/chris/Documents/Valerie_work/Crete/Crete_slip_model.txt'
slip_data = pd.read_csv(slip_model_file, delimiter="\s+", skiprows=14, header = None)
slip_data.columns = ["Latitude", "Longitude", "Depth", "Slip(cm)", "Rake", "Strike", "Dip"]
pga_file = ('/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/Strong_motion/mainshockPGA.txt')
station_flatfile_path = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/collected_flatfile_main_event.csv'
station_flatfile = pd.read_csv(station_flatfile_path, header='infer')
Rhypo = np.array([])
Rrup = np.array([])

hypo_location = [25.71, 34.182, 10]


#%%
Rrup, Rhypo = rup.compute_openquake_distances(slip_data, pga_file, hypo_location)
vs30_file_path = '/Users/chris/Documents/Valerie_work/global_vs30.grd'
xytmpfile_path = '/Users/chris/Documents/Valerie_work/Crete/xytmpfile'
xy_vs30_tmpfile_path = '/Users/chris/Documents/Valerie_work/Crete/xy_vs30_tmpfile'

vs30 = rup.extract_vs30_forpandas(pga_file, xytmpfile_path, xy_vs30_tmpfile_path, vs30_file_path) 
#%%
kk = pd.read_csv('/Users/chris/Documents/Valerie_work/Crete/Observed_IMTS/obs_IMs_P2_#1.csv', header = 'infer' )
pga_values = np.genfromtxt('/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/Strong_motion/mainshockPGA.txt', usecols=8)
#Rrup_1 = np.array([np.log10(750), np.log10(1000), np.log10(5000), np.log10(10000),np.log10(20000),np.log10(30000),np.log10(40000) ,np.log10(50000),np.log10(60000),np.log10(700000)])
Rrup_1 = np.logspace(1.5, 3, 30, endpoint=True)
Rrup_model = np.asarray([100, 200])
#Rrup = np.sort(kk['hypdist'].values)
#Rrup = np.sort(Rrup)
mag = np.linspace(3.5,6.5, 30)
lmean_ask14, sd_ask14 = rup.oq_ask2014(6.6, Rrup, vs30=vs30)
lmean_ask14 =  9.8*(np.exp(lmean_ask14))
lmean_b14, sd_b14 = rup.oq_boore2014(6.6,Rrup, vs30=vs30)
lmean_b14 = 9.8*(np.exp(lmean_b14))
lmean_b14_5, sd_b14_5 = rup.oq_boore2014(5.3, Rrup)
lmean_b14_5 = 9.8*(np.exp(lmean_b14_5))
lmean_cy14, sd_cy14 = rup.oq_Chioy2014(6.6, Rrup, vs30=vs30)
lmean_cy14 =  9.8*(np.exp(lmean_cy14))
lmean_sk13, sd_sk13 = rup.oq_Skarlatoudis2013(6.6, Rrup, vs30=vs30)
lmean_sk13 =  9.8*(np.exp(lmean_sk13))
lmean_sk13_5, sd_sk13_5 = rup.oq_Skarlatoudis2013(5.8, Rrup)
lmean_sk13_5 =  9.8*(np.exp(lmean_sk13_5))
lmean_z16, sd_z16 = rup.oq_Zhao_2006(6.6,Rrup, vs30=vs30)
lmean_z16 =  9.8*(np.exp(lmean_z16))
#%%
## Prediction calculation and plotting


plt.figure()
plt.scatter(Rrup, pga_values)
# for i in range(len(Rrup_model)):
#     b = [el[i] for el in lmean_ask14]
#     plt.plot(mag, b, '--', color='black') 
# for i in range(len(Rrup_model)):    
#     c = [el[i] for el in lmean_b14]
#     plt.plot(mag, c, '--', color='black') 
    
plt.scatter(Rrup, lmean_ask14, color='red', label='Abrahamson2014')
plt.scatter(Rrup, lmean_b14, color='green', label= 'Boore2014')
plt.scatter(Rrup, lmean_cy14, color='black', label= 'ChioyYoung2014')
plt.scatter(Rrup, lmean_sk13, color='blue', label= 'Skarlatoudis2013')
plt.scatter(Rrup, lmean_z16, color='orange', label= 'Zhao2006')
plt.scatter(Rrup, lmean_sk13_5, color='blue', marker='+', label= 'Skarlatoudis2013_5.8')
plt.scatter(Rrup, lmean_b14_5, color='green', marker='+', label= 'Boore2014_5.3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rrup (km)')
plt.ylabel('PGA (m/s/s/)')
plt.text(5.5, 0.8E-3, 'ASK14 = 100km', rotation = 20)
plt.text(5.5, 2.8E-5, 'ASK14 = 200km', rotation = 20)
plt.xlim(0.5E2, 8E2)
plt.ylim(1E-3, 1E0)
plt.legend()
# plt.savefig('/Users/chris/Documents/Valerie_work/Crete/figures' +'figure_p2_apr_12.png')
# clb = plt.colorbar()
# clb.ax.set_title('Distance(km)')
plt.show()

#%%
## Scatter plot for all polygons


kkk = kk.loc[kk['mw'] > 5.]
kkk = kkk[kkk['pga'].notna()]
plt.figure()
plt.scatter( kkk['hypdist'], kkk['pga'], c=kkk['mw'],cmap='viridis')
plt.xscale('log')
plt.yscale('log')    
plt.xlabel('Rrup(km)')
plt.ylabel('PGA (m/s/s)')
clb = plt.colorbar()
clb.ax.set_title('Mw')
#%%
## Residual plot
from scipy.stats import binned_statistic
bin_r_start = np.log10(55)
bin_r_end = np.log10(1000)
bin_r_num = 9
r_bins = np.logspace(bin_r_start,bin_r_end,bin_r_num)
ask14_color = '#40235e'
sk13_color = '#5aad6a'
b14_color = '#eded55'
zhao_color = '#3b918d'

i_total_residual_ask14 = np.log(pga_values) - np.log(lmean_ask14)
ask14_means,ask14_binedges,ask14_binnumbers = binned_statistic(Rrup,i_total_residual_ask14, bins=r_bins,statistic='mean')
ask14_std,ask14_std_binedges,ask14_std_binnumbers = binned_statistic(Rrup,i_total_residual_ask14,bins=r_bins,statistic='std') 

i_total_residual_sk13 = np.log(pga_values) - np.log(lmean_sk13)
sk13_means,sk13_binedges,sk13_binnumbers = binned_statistic(Rrup,i_total_residual_sk13, bins=r_bins,statistic='mean')
sk13_std,sk13_std_binedges,sk13_std_binnumbers = binned_statistic(Rrup,i_total_residual_sk13,bins=r_bins,statistic='std') 

i_total_residual_b14 = np.log(pga_values) - np.log(lmean_b14)
b14_means,b14_binedges,b14_binnumbers = binned_statistic(Rrup,i_total_residual_b14, bins=r_bins,statistic='mean')
b14_std,b14_std_binedges,b14_std_binnumbers = binned_statistic(Rrup,i_total_residual_b14,bins=r_bins,statistic='std') 

i_total_residual_z16 = np.log(pga_values) - np.log(lmean_z16)
z16_means,z16_binedges,z16_binnumbers = binned_statistic(Rrup,i_total_residual_z16, bins=r_bins,statistic='mean')
z16_std,z16_std_binedges,z16_std_binnumbers = binned_statistic(Rrup,i_total_residual_z16,bins=r_bins,statistic='std') 

bin_x_ask14 = (ask14_binedges+(bin_r_num/2))[0:-1]
bin_x_sk13 = (sk13_binedges+(bin_r_num/2))[0:-1]
bin_x_b14 = (b14_binedges+(bin_r_num/2))[0:-1]
bin_x_z16 = (z16_binedges+(bin_r_num/2))[0:-1]

# Scatter garcia:
plt.figure()    
plt.scatter(Rrup,i_total_residual_ask14,marker='^',s=10,facecolor=ask14_color,edgecolor='k',label='ASK14')
plt.scatter(Rrup,i_total_residual_sk13,marker='o',s=10,facecolor=sk13_color,edgecolor='k',label='SK13')
plt.scatter(Rrup,i_total_residual_b14,marker='h',s=10,facecolor=b14_color,edgecolor='k',label='B14')
plt.scatter(Rrup,i_total_residual_z16,marker='s',s=10,facecolor=zhao_color,edgecolor='k',label='Zhao06')

plt.xscale('log')
plt.errorbar(bin_x_ask14,ask14_means,yerr=ask14_std,marker='^',linewidth=2,markersize=10,color=ask14_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='ASK14 binned')
plt.errorbar(bin_x_sk13,sk13_means,yerr=sk13_std,marker='o',linewidth=2,markersize=10,color=sk13_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='SK13 binned')
plt.errorbar(bin_x_b14,b14_means,yerr=b14_std,marker='h',linewidth=2,markersize=10,color=b14_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='B14 binned')
plt.errorbar(bin_x_z16,z16_means,yerr=z16_std,marker='s',linewidth=2,markersize=10,color=zhao_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='Zhao binned')
plt.axhline(y=0,linestyle='--',color='#606060')
plt.xlabel('Distance (km)',labelpad=0)
plt.ylabel('ln residual',labelpad=0)

plt.legend()


#%%

#SA plotting

from matplotlib import colors

SA_period = [3,2,1.5,1,0.5,0.3]
Rrup_1 = np.logspace(1.5, 3, 30, endpoint=True)
SA_file = ('/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/Strong_motion/mainshockPGA.txt')
number = 9
SA_legend = ['Period = 3s', 'Period = 2s', 'Period = 1.5s', 'Period = 1s', 'Period = 0.5s', 'Period = 0.3s', 'Period = 0.1s' ]
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1,7 ):
    lmean_ask14, sd_ask14 = rup.oq_ask2014(6.6, Rrup_1, predictive_parameter=SA_period[i-1])
    lmean_ask14 =  9.8*(np.exp(lmean_ask14))
    lmean_sk13, sd_sk13 = rup.oq_Skarlatoudis2013(6.6, Rrup_1, predictive_parameter=SA_period[i-1])
    lmean_sk13 =  9.8*(np.exp(lmean_sk13))
    ax = fig.add_subplot(2, 3, i)
    ax.scatter(Rrup, np.genfromtxt('/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/Strong_motion/mainshockPGA.txt', usecols=number+i), label=SA_legend[i-1])
    ax.plot(Rrup_1, lmean_ask14, color='red', label='Ask14')
    ax.plot(Rrup_1, lmean_sk13, color='green', label='SK13')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Rrup(km)')
    ax.set_ylabel('SA')    
    ax.legend()       


# SA_period = [3,2,1.5,1,0.5,0.3]
# Rrup_1 = np.logspace(1.5, 3, 30, endpoint=True)
# SA_file = ('/Users/chris/Documents/Valerie_work/Crete/Observed_IMTS/SA_main_P4_#1.txt')
# number = 1
# SA_legend = ['Period = 3s', 'Period = 2s', 'Period = 1.5s', 'Period = 1s', 'Period = 0.5s', 'Period = 0.3s', 'Period = 0.1s' ]
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.4)
# for i in range(1,7 ):
#     gg = fig.add_subplot(2, 3, i)
#     axx = gg.scatter(np.genfromtxt(SA_file, usecols=7, skip_header = 1), np.genfromtxt(SA_file, usecols=i+7, skip_header = 1),c=dataset_dict['mw'],cmap='viridis', label=SA_legend[i-1], vmin=3, vmax=6)
#     gg.set_xscale('log')
#     gg.set_yscale('log')
#     gg.set_xlabel('Repi(km)')
#     gg.set_ylabel('SA')    
#     gg.legend()
# fig.subplots_adjust(right=0.75)    
# cbar_ax = fig.add_axes([0.80, 0.15, 0.02, 0.6])    
# clb = fig.colorbar(axx, cax=cbar_ax, label= 'Mw')


#%%

# Residual plot
from scipy.stats import binned_statistic
bin_r_start = np.log10(55)
bin_r_end = np.log10(1000)
bin_r_num = 9
r_bins = np.logspace(bin_r_start,bin_r_end,bin_r_num)
ask14_color = '#40235e'
sk13_color = '#5aad6a'
b14_color = '#eded55'
zhao_color = '#3b918d'

pred_file_1 = '/Users/chris/Documents/Valerie_work/Crete/Predicted_IMTS/pred_IMs_P1_#1.csv'
df_1 = pd.read_csv(pred_file_1, header='infer')
pred_file_2 = '/Users/chris/Documents/Valerie_work/Crete/Predicted_IMTS/pred_IMs_P2_#1.csv'
df_2 = pd.read_csv(pred_file_2, header='infer')
pred_file_3 = '/Users/chris/Documents/Valerie_work/Crete/Predicted_IMTS/pred_IMs_P3_#1.csv'
df_3 = pd.read_csv(pred_file_3, header='infer')
pred_file_4 = '/Users/chris/Documents/Valerie_work/Crete/Predicted_IMTS/pred_IMs_P4_#1.csv'
df_4 = pd.read_csv(pred_file_4, header='infer')

#i_total_residual_ask14 = np.log(pga_values) - np.log(lmean_ask14)
i_total_residual_ask14 = df_1['sk13_0.3']
Rrup_1 = df_1['hypdist']
ask14_means,ask14_binedges,ask14_binnumbers = binned_statistic(Rrup_1,i_total_residual_ask14, bins=r_bins,statistic='mean')
ask14_std,ask14_std_binedges,ask14_std_binnumbers = binned_statistic(Rrup_1,i_total_residual_ask14,bins=r_bins,statistic='std') 

#i_total_residual_sk13 = np.log(pga_values) - np.log(lmean_sk13)
i_total_residual_sk13 = df_2['sk13_0.3']
Rrup_2 = df_2['hypdist']
sk13_means,sk13_binedges,sk13_binnumbers = binned_statistic(Rrup_2,i_total_residual_sk13, bins=r_bins,statistic='mean')
sk13_std,sk13_std_binedges,sk13_std_binnumbers = binned_statistic(Rrup_2,i_total_residual_sk13,bins=r_bins,statistic='std') 

#i_total_residual_b14 = np.log(pga_values) - np.log(lmean_b14)
i_total_residual_b14 = df_3['sk13_0.3']
Rrup_3 = df_3['hypdist']
b14_means,b14_binedges,b14_binnumbers = binned_statistic(Rrup_3,i_total_residual_b14, bins=r_bins,statistic='mean')
b14_std,b14_std_binedges,b14_std_binnumbers = binned_statistic(Rrup_3,i_total_residual_b14,bins=r_bins,statistic='std') 

#i_total_residual_z16 = np.log(pga_values) - np.log(lmean_z16)
i_total_residual_z16 = df_4['sk13_0.3']
Rrup_4 = df_4['hypdist']
z16_means,z16_binedges,z16_binnumbers = binned_statistic(Rrup_4,i_total_residual_z16, bins=r_bins,statistic='mean')
z16_std,z16_std_binedges,z16_std_binnumbers = binned_statistic(Rrup_4,i_total_residual_z16,bins=r_bins,statistic='std') 

bin_x_ask14 = (ask14_binedges+(bin_r_num/2))[0:-1]
bin_x_sk13 = (sk13_binedges+(bin_r_num/2))[0:-1]
bin_x_b14 = (b14_binedges+(bin_r_num/2))[0:-1]
bin_x_z16 = (z16_binedges+(bin_r_num/2))[0:-1]

# Scatter garcia:
plt.figure()    
plt.scatter(Rrup_1,i_total_residual_ask14,marker='^',s=10,facecolor=ask14_color,edgecolor='k',label='Polygon 1')
plt.scatter(Rrup_2,i_total_residual_sk13,marker='o',s=10,facecolor=sk13_color,edgecolor='k',label='Polygon 2')
plt.scatter(Rrup_3,i_total_residual_b14,marker='h',s=10,facecolor=b14_color,edgecolor='k',label='Polygon 3')
plt.scatter(Rrup_4,i_total_residual_z16,marker='s',s=10,facecolor=zhao_color,edgecolor='k',label='Polygon 4')

plt.xscale('log')
plt.errorbar(bin_x_ask14,ask14_means,yerr=ask14_std,marker='^',linewidth=2,markersize=10,color=ask14_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='1 binned')
plt.errorbar(bin_x_sk13,sk13_means,yerr=sk13_std,marker='o',linewidth=2,markersize=10,color=sk13_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='2 binned')
plt.errorbar(bin_x_b14,b14_means,yerr=b14_std,marker='h',linewidth=2,markersize=10,color=b14_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='3 binned')
plt.errorbar(bin_x_z16,z16_means,yerr=z16_std,marker='s',linewidth=2,markersize=10,color=zhao_color,ecolor='k',elinewidth=2,capsize=5,capthick=2,label='4 binned')
plt.axhline(y=0,linestyle='--',color='#606060')
plt.xlabel('Distance (km)',labelpad=0)
plt.ylabel('ln residual(SA)',labelpad=0)
plt.title('SK13 Period = 0.3s')

plt.legend()


































