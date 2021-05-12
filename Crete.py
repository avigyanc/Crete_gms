#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:00:46 2021

@author: chris
"""
#%%
import numpy as np
import pandas as pd
import obspy
from glob import glob
import matplotlib.pyplot as plt

# Local Imports
import tsueqs_main_fns as tmf
import signal_average_fns as avg
import rupture_functions as rup
#%%
######################### Functions ##########################################

def calc_time_to_peak(pgm, trace, IMarray, origintime, hypdist):
    """
    Calculates time to peak intensity measure (IM) from origin time and from
    estimated p-arrival.
    
    Inputs:
        pgm(float): Peak ground motion.
        trace: Trace object with times for this station. 
        IMarray: Array pgd was calculated on. 
        origintime(datetime): Origin time of event.
        hypdist(float): Hypocentral distance (km). 
        
    """
    
    from obspy.core import UTCDateTime
    import datetime
    
    # Calculate time from origin 
    pgm_index = np.where(IMarray==pgm)
    tPGM_orig = trace.times(reftime=UTCDateTime(origintime))[pgm_index]
    
    # Calculate time from p-arrival
    p_time = hypdist/6.5
    dp = datetime.timedelta(seconds=p_time)
    p_arrival = origintime+dp
    tPGM_parriv = trace.times(reftime=UTCDateTime(p_arrival))[pgm_index]
    
    return(tPGM_orig, tPGM_parriv)



#%%


################################ Parameters ###################################

# Used for directory paths


# Data types to loop through.  I have a folder for displacement ('disp') and a 
    # folder for acceleration ('accel'), so those are my data types. 
data_types = ['accel']

# Project directory 
proj_dir = '/Users/chris/Documents/Valerie_work/Crete' 
xml_file_path = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/SNR_10/raw/XML_files/'


# Table of earthquake data
eq_table_path = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/P2/flatfile_p2.csv'   
eq_table = pd.read_csv(eq_table_path, header='infer')
station_flatfile_path= '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/P2/collected_flatfile_p2.csv'
station_flatfile = pd.read_csv(station_flatfile_path, header='infer')
# Data directories- one for displacement and one for strong motion (acc)     
data_dir = proj_dir + '/Data_download' 
vel_dir = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/P2/SNR_5'

sm_dir = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/SNR_10/raw'
sm_flatfile = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/SNR_10/raw/XML_files/sm_collected_flatfile_main_event.csv'
sm_flatfile = pd.read_csv(sm_flatfile, header='infer')
figures = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/P1/figures/'
# Path to send flatfiles of intensity measures
flatfile_path = proj_dir + '/obs_IMs_P2_#1.csv'   
outfile = proj_dir + '/SA_main_P2_#1.txt'  

# Parameters for integration to velocity and filtering 
fcorner = 1/15.                          # Frequency at which to high pass filter
order = 2                                # Number of poles for filter  
fmin = 0.01
fmax = 1
# Gather displacement and strong motion files
vel_files = np.array(sorted(glob(vel_dir + '/*.sac')))
sm_files = np.array(sorted(glob(vel_dir + '/*.sac')))
#sm_files = np.array(sorted(glob(sm_dir + '/*.mseed')))

#%%
################################ Event Data ###################################




#origin = pd.to_datetime('2020-5-2T12:51:05')  

eventname = eq_table['Event_num']

origintime = eq_table['Origin_time']
hyplon = eq_table['LONGITUDE']
hyplat = eq_table['LATITUDE']
hypdepth = eq_table['DEPTH']
mw = eq_table['MAG']
m0 = 10**(mw*(3/2.) + 9.1)



########## Initialize lists for the event and station info for the df #########

eventnames = np.array([])
origintimes = np.array([])
hyplons = np.array([])
hyplats = np.array([])
hypdepths = np.array([])
mws = np.array([])
m0s = np.array([])
networks = np.array([])
stations = np.array([])
stn_type_list = np.array([])
stlons = np.array([])
stlats = np.array([])
stelevs = np.array([])
hypdists = np.array([])
instrument_codes = np.array([])
nostations = np.array([])
E_Td_list = np.array([])
N_Td_list = np.array([])
Z_Td_list = np.array([])
horiz_Td_list = np.array([])
comp3_Td_list = np.array([])
pga_list = np.array([])
pgv_list = np.array([])
pgd_list = np.array([])
tPGV_orig_list = np.array([])
tPGV_parriv_list = np.array([])
tPGA_orig_list = np.array([])
tPGA_parriv_list = np.array([])
tPGA_list = np.array([])
evlon = np.array([])
evlat = np.array([])
evdp = np.array([])
ot = np.array([])
ntw = np.array([])
stnname = np.array([])
stnlon = np.array([])
stnlat = np.array([])
stnelv = np.array([])
eventid = np.array([])
SA = []
SAfreqs=[1/3,1/2,1/1.5,1,1/0.5,1/0.3,1/0.1]
#%%
###################### Data Processing and Calculations #######################
from itertools import groupby
from obspy.core import UTCDateTime
# Threshold- used to calculate duration 
threshold = 0.0
E_files = []
N_files = []
# Loop through data types
for data in data_types:
    
    ###################### Set parameters for data type #######################
    
    if data == 'vel':
        
        # Get metadata file
        metadata_file = station_flatfile
        
        # Get mseed files
        files = vel_files
        
        # Types of IMs associated with this data type
        IMs = ['pgv']
        
        # Sampling rate
        nsamples = 100
        
        # Channel code prefix
        code = 'HH'
        
        # Filtering
            # Displacement data don't need to be highpass filtered 
        filtering = False


    elif data == 'accel':
        
        # Get metadata file
        metadata_file = station_flatfile
        
        # Get mseed files
        files =  vel_files
        
    
        
        # Types of IMs associated with this data type
        IMs = ['pga']
  
        # Sampling rate
        nsamples = 100
        
        # Channel code prefix
        code = 'HN'
        
        # Filtering
            # Acceleration data need to be highpass fitlered 
        filtering = True


    ############################# Get metadata ################################
   
    # Read in metadata file
    metadata = station_flatfile 
    #metadata = sm_flatfile
    
    
    
    
    ######################## Get station data and files #######################
    
    # Create lists to add station names, channels, and miniseed files to 
    stn_name_list = []
    channel_list = []
    mseed_list = []
    
    # Group all files by station since there should be 3 components for each 
        # station
    N = 2
    stn_files = [files[n:n+N] for n in range(0, len(files), N)]
    # print(stn_files)
    # print("********************")
 
    # Loop over files to get the list of station names, channels, and mseed files 
    for station in stn_files:
        #print(station)
        
 
        # Initialize lists for components and mseed files for this station
        components = []
        mseeds = []
    
        # Get station name and append to station name list
        stn_name = station[0].split('_')[11]
        
        #stn_name = (station[0].split('/')[11]).split('.')[1]
        stn_name_list.append(stn_name)
        
    
            
        # Loop through station mseed files
        for count, mseed_file in enumerate(station):
            
            #print(mseed_file)
            
            # Get channel code and append to components list
            channel_code = mseed_file.split('_')[12]
            station_name = mseed_file.split('_')[11]
            
            #channel_code = ((station[count].split('/')[11]).split('.')[3]).split('_')[0]
            
            components.append(channel_code)
            
            # Append mseed file to mseed files list
            mseeds.append(mseed_file)
        
        # Append station's channel code list to channel list for all stations
        channel_list.append(components)
        # Append station's mseed files list to mseed files list for all stations
        mseed_list.append(mseeds)
        mseed_list = [item for sub_list in mseed_list for item in sub_list]
        mseed_list = sorted(mseed_list, key = lambda x: str(x.split('_')[11]))
        mseed_list = [list(i) for j, i in groupby(mseed_list, lambda a: a.split('_')[11])]
        

    #################### Begin Processing and Calculations ####################
    ll = list(set(stn_name_list))
    ll.sort()
    # Loop over the stations for this earthquake, and start to run the computations:
    for i, station in enumerate(ll):
        #print(i)
        # Get the components for this station (E, N, and Z):
        components = []
        for channel in channel_list[i]:
            components.append(channel[2])
           
        # Get the metadata for this station from the chan file - put it into
            # a new dataframe and reset the index so it starts at 0
       
            station_metadata = metadata[metadata.Name == station].reset_index(drop=True)

        # Pull out the data. Take the first row of the subset dataframe, 
            # assuming that the gain, etc. is always the same:
        stnetwork = station_metadata.loc[0].Network
        stlon = station_metadata.loc[0].Slon
        stlat = station_metadata.loc[0].Slat
        stelev = station_metadata.loc[0].Selv
        stname = station_metadata.loc[0].Name
        #print(stname)


        ######################### Start computations ##########################       

        # Compute the hypocentral distance
        for jj in range(len(hyplon)):
            #print(jj)
            hypdist = tmf.compute_rhyp(stlon,stlat,stelev,hyplon[jj],hyplat[jj],hypdepth[jj])
            #print(hypdist)
          
            # Append the earthquake and station info for this station to their lists
            eventnames = np.append(eventnames,eventname)
            networks = np.append(networks,stnetwork)
            stations = np.append(stations,station)
            stlons = np.append(stlons,stlon)
            stlats = np.append(stlats,stlat)
            stelevs = np.append(stelevs,stelev)
            origintimes = np.append(origintimes,origintime[jj])
            hyplons = np.append(hyplons,hyplon[jj])
            hyplats = np.append(hyplats,hyplat[jj])
            hypdepths = np.append(hypdepths,hypdepth[jj])
            #mws = np.append(mws,mw[jj])
            #m0s = np.append(m0s,m0[jj])
            

            
            
           
            if data == 'vel':
                stn_type_list = np.append(stn_type_list, 'TS')
            elif data == 'accel':
                stn_type_list = np.append(stn_type_list, 'SM')
            
            # Initialize list for all spectra at this station
            station_spec = []
     
            # Turn the components list into an array 
            components = np.asarray(components)
            
            # Get the values for the E component 
            if 'E' in components:
              
                # Get index for E component 
                E_index = np.where(components=='E')[0][0]
                b= [4,5,6,7,8,9]
                waveform =UTCDateTime( *(np.array(mseed_list[i][E_index].split('_'))[b]).astype(int))
                # Read file into a stream object
                #print(origintimes[jj])
                for kk in range(len(mseed_list[i])):
                    if(kk%2==0):
                        #print(mseed_list[i][kk])
                        waveform =UTCDateTime( *(np.array(mseed_list[i][kk].split('_'))[b]).astype(int))
                        if(abs(UTCDateTime(origintimes[jj]) - waveform) < 1):
                            #print('##TRUE##')
                            E_raw = obspy.read(mseed_list[i][kk])
                            E_files.append(E_raw)
                            hypdists = np.append(hypdists,hypdist)
                            mws = np.append(mws,mw[jj])
                            evlon = np.append(evlon,hyplon[jj])
                            evlat = np.append(evlat,hyplat[jj])
                            evdp = np.append(evdp,hypdepth[jj])
                            ot = np.append(ot,origintimes[jj])
                            eventid = np.append(eventid,eventnames[jj])
                            ntw = np.append(ntw,stnetwork)
                            stnname = np.append(stnname,station)
                            stnlon = np.append(stnlon,stlon)
                            stnlat = np.append(stnlat,stlat)
                            stnelv = np.append(stnelv,stelev)
                            
        
                
            if 'N' in components:
               
                # Get index for N component 
                N_index = np.where(components=='N')[0][0]   
                b= [4,5,6,7,8,9]
                waveform =UTCDateTime( *(np.array(mseed_list[i][N_index].split('_'))[b]).astype(int))
                # Read file into a stream object
                #print(origintimes[jj])
                for kk in range(len(mseed_list[i])):
                    if(kk%2!=0):
                        #print(mseed_list[i][kk])
                        waveform =UTCDateTime( *(np.array(mseed_list[i][kk].split('_'))[b]).astype(int))
                        if(abs(UTCDateTime(origintimes[jj]) - waveform) < 1):
                            #print('##TRUE##')
                            N_raw = obspy.read(mseed_list[i][kk])
                            N_files.append(N_raw)
                            
                


for i in range(len(E_files)):
    E_record = E_files[i]
    N_record = N_files[i]
    E_Td, E_start, E_end = tmf.determine_Td(threshold,E_record)      
    E_Td_list = np.append(E_Td_list,E_Td)  
    N_Td, N_start, N_end = tmf.determine_Td(threshold,N_record)  
    N_Td_list = np.append(N_Td_list,N_Td)  

# Take the min time of E and N start times to be the start
    EN_start = np.min([E_start,N_start])
    
    # Take the max time of E and N end times to be the end
    EN_end = np.max([E_end,N_end])
    
    # Get the duration to be the time between these
    EN_Td = EN_end - EN_start
    horiz_Td_list = np.append(horiz_Td_list,EN_Td)      
    
    
    print('************************************')
    E_record_acc =  tmf.vel_to_acc(E_record, fmin, fmax, order)
    N_record_acc =  tmf.vel_to_acc(N_record, fmin, fmax, order)
# if(E_record_acc[0].stats.station =='STIA') and (N_record_acc[0].stats.station == 'STIA'):
#     fig, (ax1, ax2) = plt.subplots(2)
#     ax1.plot(E_record_acc[0].times(), E_record_acc[0].data)
#     ax2.plot(N_record_acc[0].times(), N_record_acc[0].data)
#     ax1.set_ylabel(E_record_acc[0].stats.channel)
#     ax2.set_ylabel(N_record_acc[0].stats.channel)
#     fig.savefig(figures + E_record_acc[0].stats.station+'.png')
#     fig = plt.close()

## PGA         
# Get euclidean norm of acceleration components     
    if(len(E_record_acc[0].data) != len(N_record_acc[0].data)):
        
        st = E_record_acc.append(N_record_acc[0])
        gaps = st.get_gaps()
        
        if gaps==[]:
            print(f'.....Shortening record for {E_record[0].stats.station}')
            npts = np.min([len(N_record_acc[0].data), len(E_record_acc[0].data)])
            N_record_acc[0].data = N_record_acc[0].data[:npts]
            E_record_acc[0].data = E_record_acc[0].data[:npts]
            E_record_acc[0].data-=np.mean(E_record_acc[0].data)
            N_record_acc[0].data-=np.mean(N_record_acc[0].data)
            acc_euc_norm = avg.get_eucl_norm_2comp(E_record_acc[0].data,
                                               N_record_acc[0].data)
            
            # Calculate PGA
            pga = np.max(np.abs(acc_euc_norm))
            print(E_record_acc[0].stats.station)
            print('pga')
            print(pga)
            pga_list = np.append(pga_list,pga)
            # Calcualte tPGD from origin and p-arrival
            tPGA_orig, tPGA_parriv = calc_time_to_peak(pga, E_record[0],
                                                            np.abs(acc_euc_norm),
                                                            pd.to_datetime(origintime[jj]), hypdist)
            tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)
            tPGA_parriv_list = np.append(tPGA_parriv_list,tPGA_parriv)
            SA.append(rup.rotatedResponseSpectrum(0.01, (E_record_acc[0].data), (N_record_acc[0].data), SAfreqs, oscDamping=0.05))
            m0s = np.append(mws,mw[jj])
        else:
            continue
    else:
        E_record_acc[0].data-= np.mean(E_record_acc[0].data)
        N_record_acc[0].data-= np.mean(N_record_acc[0].data)
        acc_euc_norm = avg.get_eucl_norm_2comp(E_record_acc[0].data,
                                           N_record_acc[0].data)
        
        # Calculate PGA
        pga = np.max(np.abs(acc_euc_norm))
        print(E_record_acc[0].stats.station)
        print('pga')
        print(pga)
        pga_list = np.append(pga_list,pga)
        # Calcualte tPGD from origin and p-arrival
        tPGA_orig, tPGA_parriv = calc_time_to_peak(pga, E_record[0],
                                                        np.abs(acc_euc_norm),
                                                        pd.to_datetime(origintime[jj]), hypdist)
        tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)
        tPGA_parriv_list = np.append(tPGA_parriv_list,tPGA_parriv)
        SA.append(rup.rotatedResponseSpectrum(0.01, (E_record_acc[0].data), (N_record_acc[0].data), SAfreqs, oscDamping=0.05))
        
    # else:
    #     pga_list = np.append(pga_list,np.nan)
    #     tPGA_orig_list = np.append(tPGA_orig_list,np.nan)
    #     tPGA_parriv_list = np.append(tPGA_parriv_list,np.nan)         
    
        
        # fig, (ax1, ax2) = plt.subplots(2)
        # ax1.plot(E_record[0].times(), E_record[0].data)
        # ax2.plot(N_record[0].times(), N_record[0].data)
        # ax1.set_ylabel(E_record[0].stats.channel)
        # ax2.set_ylabel(N_record[0].stats.channel)
        # fig.savefig(figures + E_record[0].stats.station+'.png')
        # fig = plt.close()
        ## PGD
        # Get euclidean norm of displacement components
    if(len(E_record[0].data) != len(N_record[0].data)):
        
        st = E_record.append(N_record[0])
        gaps = st.get_gaps()
        
        if gaps==[]:
            print(f'.....Shortening record for {E_record[0].stats.station}')
            npts = np.min([len(N_record[0].data), len(E_record[0].data)])
            N_record[0].data = N_record[0].data[:npts]
            E_record[0].data = E_record[0].data[:npts]
            euc_norm = avg.get_eucl_norm_2comp(E_record[0].data, N_record[0].data)
            pgv = np.max(np.abs(euc_norm))
            print(pgv)
            pgv_list = np.append(pgv_list,pgv)
            # Calcualte tPGV from origin and p-arrival
            tPGV_orig, tPGV_parriv = calc_time_to_peak(pgv, E_record[0],
                                                            np.abs(euc_norm),
                                                            pd.to_datetime(origintime[jj]), hypdist)
            print(tPGV_orig)
            tPGV_orig_list = np.append(tPGV_orig_list,tPGV_orig)
            tPGV_parriv_list = np.append(tPGV_parriv_list,tPGV_parriv)
        else:
            continue
    else:
        euc_norm = avg.get_eucl_norm_2comp(E_record[0].data, N_record[0].data)
        pgv = np.max(np.abs(euc_norm))
        print(pgv)
        pgv_list = np.append(pgv_list,pgv)
        # Calcualte tPGV from origin and p-arrival
        tPGV_orig, tPGV_parriv = calc_time_to_peak(pgv, E_record[0],
                                                        np.abs(euc_norm),
                                                        pd.to_datetime(origintime[jj]), hypdist)
        print(tPGV_orig)
        tPGV_orig_list = np.append(tPGV_orig_list,tPGV_orig)
        tPGV_parriv_list = np.append(tPGV_parriv_list,tPGV_parriv)
        
               
                # Calculate PGV
                
    
        
            # If data type is not displacement, append 'nans'
    # else:
    #     pgv_list = np.append(pgv_list,np.nan)
    #     tPGV_orig_list = np.append(tPGV_orig_list,np.nan)
    #     tPGV_parriv_list = np.append(tPGV_parriv_list,np.nan)    

'''                
                # Get the duration, stream file time of start, and time of stop of shaking
                if(len(E_record) > 0):
                    E_Td, E_start, E_end = tmf.determine_Td(threshold,E_record)      
                    E_Td_list = np.append(E_Td_list,E_Td)
    
              
    
                
            # Get the values for the N component
            if 'N' in components:
               
                # Get index for N component 
                N_index = np.where(components=='N')[0][0]
                
                # Read file into a stream object
                waveform =UTCDateTime( *(np.array(mseed_list[i][N_index].split('_'))[b]).astype(int))
                # Read file into a stream object
                print(origintimes[jj])
                #print(abs(UTCDateTime(origintimes[jj]) - waveform))
                k = len(mseed_list[i])
                print(k)
                while k!=0:
                    ii = 0
                    waveform =UTCDateTime( *(np.array(mseed_list[i][N_index+ii].split('_'))[b]).astype(int))
                    print(mseed_list[i][N_index+ii])
                    if(abs(UTCDateTime(origintimes[jj]) - waveform) < 1) :
                       
                        N_raw = obspy.read(mseed_list[i][N_index + ii])
                    else:
                        
                        N_raw = obspy.Stream().clear()
                       
                    k -=2
                    ii +=2
                    # waveform =UTCDateTime( *(np.array(mseed_list[i][N_index].split('_'))[b]).astype(int))
                    # try:
                    #     if(abs(UTCDateTime(origintimes[jj]) - waveform) < 1):
                    #         print('True')
                    #         N_raw = obspy.read(mseed_list[i][N_index])
                    #     else:
                    #         waveform =UTCDateTime( *(np.array(mseed_list[i][N_index +2].split('_'))[b]).astype(int))                                                            
                    #         print('*******FALSE**********')
                    #         N_raw = obspy.Stream().clear()
                    #         jj+=1
                    # except IndexError:
                    #      continue
                print(N_raw)
                
                N_record = N_raw 
             
                    
                # Get the duration, stream file time of start, and time of stop of shaking
                if(len(N_record) > 0):
                    N_Td, N_start, N_end = tmf.determine_Td(threshold,N_record)  
                    N_Td_list = np.append(N_Td_list,N_Td)
    
    


    
            # Get the values for the horizontal components 
            if ('E' in components) and ('N' in components):
                
                # Take the min time of E and N start times to be the start
                EN_start = np.min([E_start,N_start])
                
                # Take the max time of E and N end times to be the end
                EN_end = np.max([E_end,N_end])
                
                # Get the duration to be the time between these
                EN_Td = EN_end - EN_start
                horiz_Td_list = np.append(horiz_Td_list,EN_Td)
    
            else:
                # Append nan to the overall arrays if horizontals don't exist:
                horizon_Td_list = np.append(horiz_Td_list,np.nan)
                
            
    
           ########################### Intensity Measures ########################
            
            # Calculate displacement intensity measures
            if data == 'vel':
                fig, (ax1, ax2) = plt.subplots(2)
                ax1.plot(E_record[0].times(), E_record[0].data)
                ax2.plot(N_record[0].times(), N_record[0].data)
                ax1.set_ylabel(E_record[0].stats.channel)
                ax2.set_ylabel(N_record[0].stats.channel)
                fig.savefig(figures + E_record[0].stats.station+'.png')
                fig = plt.close()
                ## PGD
                # Get euclidean norm of displacement components
                if(len(E_record[0].data) != len(N_record[0].data)):
                    
                    st = E_record.append(N_record[0])
                    gaps = st.get_gaps()
                    
                    if gaps==[]:
                        print(f'.....Shortening record for {E_record[0].stats.station}')
                        npts = np.min([len(N_record[0].data), len(E_record[0].data)])
                        N_record[0].data = N_record[0].data[:npts]
                        E_record[0].data = E_record[0].data[:npts]
                        euc_norm = avg.get_eucl_norm_2comp(E_record[0].data, N_record[0].data)
                        pgv = np.max(np.abs(euc_norm))
                        print(pgv)
                        pgv_list = np.append(pgv_list,pgv)
                        # Calcualte tPGV from origin and p-arrival
                        tPGV_orig, tPGV_parriv = calc_time_to_peak(pgv, E_record[0],
                                                                        np.abs(euc_norm),
                                                                        pd.to_datetime(origintime[jj]), hypdist)
                        print(tPGV_orig)
                        tPGV_orig_list = np.append(tPGV_orig_list,tPGV_orig)
                        tPGV_parriv_list = np.append(tPGV_parriv_list,tPGV_parriv)
                    else:
                        continue
                else:
                    euc_norm = avg.get_eucl_norm_2comp(E_record[0].data, N_record[0].data)
                    pgv = np.max(np.abs(euc_norm))
                    print(pgv)
                    pgv_list = np.append(pgv_list,pgv)
                    # Calcualte tPGV from origin and p-arrival
                    tPGV_orig, tPGV_parriv = calc_time_to_peak(pgv, E_record[0],
                                                                    np.abs(euc_norm),
                                                                    pd.to_datetime(origintime[jj]), hypdist)
                    print(tPGV_orig)
                    tPGV_orig_list = np.append(tPGV_orig_list,tPGV_orig)
                    tPGV_parriv_list = np.append(tPGV_parriv_list,tPGV_parriv)
                    
               
                # Calculate PGV
                
    
        
            # If data type is not displacement, append 'nans'
            else:
                pgv_list = np.append(pgv_list,np.nan)
                tPGV_orig_list = np.append(tPGV_orig_list,np.nan)
                tPGV_parriv_list = np.append(tPGV_parriv_list,np.nan)
            
            # Calculate acceleration and velocity intensity measures
            if data == 'accel':
                if(len(E_record)>0) and len(N_record) >0 :
                    E_record_acc =  tmf.vel_to_acc(E_record, fmin, fmax, order)
                    N_record_acc =  tmf.vel_to_acc(N_record, fmin, fmax, order)
                # if(E_record_acc[0].stats.station =='STIA') and (N_record_acc[0].stats.station == 'STIA'):
                #     fig, (ax1, ax2) = plt.subplots(2)
                #     ax1.plot(E_record_acc[0].times(), E_record_acc[0].data)
                #     ax2.plot(N_record_acc[0].times(), N_record_acc[0].data)
                #     ax1.set_ylabel(E_record_acc[0].stats.channel)
                #     ax2.set_ylabel(N_record_acc[0].stats.channel)
                #     fig.savefig(figures + E_record_acc[0].stats.station+'.png')
                #     fig = plt.close()
                
                ## PGA         
                # Get euclidean norm of acceleration components     
                    if(len(E_record_acc[0].data) != len(N_record_acc[0].data)):
                        
                        st = E_record_acc.append(N_record_acc[0])
                        gaps = st.get_gaps()
                        
                        if gaps==[]:
                            print(f'.....Shortening record for {E_record[0].stats.station}')
                            npts = np.min([len(N_record_acc[0].data), len(E_record_acc[0].data)])
                            N_record_acc[0].data = N_record_acc[0].data[:npts]
                            E_record_acc[0].data = E_record_acc[0].data[:npts]
                            E_record_acc[0].data-=np.mean(E_record_acc[0].data)
                            N_record_acc[0].data-=np.mean(N_record_acc[0].data)
                            acc_euc_norm = avg.get_eucl_norm_2comp(E_record_acc[0].data,
                                                               N_record_acc[0].data)
                            
                            # Calculate PGA
                            pga = np.max(np.abs(acc_euc_norm))
                            print(E_record_acc[0].stats.station)
                            print(pga)
                            pga_list = np.append(pga_list,pga)
                            # Calcualte tPGD from origin and p-arrival
                            tPGA_orig, tPGA_parriv = calc_time_to_peak(pga, E_record[0],
                                                                            np.abs(acc_euc_norm),
                                                                            pd.to_datetime(origintime[jj]), hypdist)
                            tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)
                            tPGA_parriv_list = np.append(tPGA_parriv_list,tPGA_parriv)
                            SA.append(rup.rotatedResponseSpectrum(0.01, (E_record_acc[0].data), (N_record_acc[0].data), SAfreqs, oscDamping=0.05))
                        else:
                            continue
                    else:
                        E_record_acc[0].data-= np.mean(E_record_acc[0].data)
                        N_record_acc[0].data-= np.mean(N_record_acc[0].data)
                        acc_euc_norm = avg.get_eucl_norm_2comp(E_record_acc[0].data,
                                                           N_record_acc[0].data)
                        
                        # Calculate PGA
                        pga = np.max(np.abs(acc_euc_norm))
                        print(E_record_acc[0].stats.station)
                        print(pga)
                        pga_list = np.append(pga_list,pga)
                        # Calcualte tPGD from origin and p-arrival
                        tPGA_orig, tPGA_parriv = calc_time_to_peak(pga, E_record[0],
                                                                        np.abs(acc_euc_norm),
                                                                        pd.to_datetime(origintime[jj]), hypdist)
                        tPGA_orig_list = np.append(tPGA_orig_list,tPGA_orig)
                        tPGA_parriv_list = np.append(tPGA_parriv_list,tPGA_parriv)
                        SA.append(rup.rotatedResponseSpectrum(0.01, (E_record_acc[0].data), (N_record_acc[0].data), SAfreqs, oscDamping=0.05))
            else:
                pga_list = np.append(pga_list,np.nan)
                tPGA_orig_list = np.append(tPGA_orig_list,np.nan)
                tPGA_parriv_list = np.append(tPGA_parriv_list,np.nan)
    
    
                # ## PGV      
                # # Get euclidean norm of velocity components 
                # vel_euc_norm = avg.get_eucl_norm_2comp(E_vel[0].data,
                #                                     N_vel[0].data)
                # # Calculate PGV
                # pgv = np.max(np.abs(vel_euc_norm))
                # pgv_list = np.append(pgv_list,pgv)
'''     

########################### Put together dataframe ############################

# First, make a dictionary for main part of dataframe:
dataset_dict = {'eventname':eventid,'origintime':ot,
                    'hyplon':evlon,'hyplat':evlat,'hypdepth (km)':evdp,
                    'mw':mws,'network':ntw,'station':stnname,
                    'stlon':stnlon,'stlat':stnlat,
                    'stelev':stnelv,'hypdist':hypdists,
                    'duration_e':E_Td_list,'duration_n':N_Td_list,
                    'duration_horiz':horiz_Td_list,
                    'pga':pga_list, 'pgv':pgv_list,'tPGV_origin':tPGV_orig_list,
                    'tPGV_parriv':tPGV_parriv_list, 'tPGA_origin':tPGA_orig_list,
                    'tPGA_parriv':tPGA_parriv_list}

f = open(outfile,'w')
f.write('# station,network,stlon,stlat, evlon, evlat, evdp, Repi(km),SA (m/s/s) at f= [1/3,1/2,1/1.5,1,1/0.5,1/0.3,1/0.1] Hz\n')
for i in range(len(SA)):
    saout=''
    for ksa in range(len(SAfreqs)):
        saout+='%.12f\t' % SA[i][ksa]
    f.write('%6s\t%12s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%s\n' % (dataset_dict['station'][i],dataset_dict['network'][i],dataset_dict['stlon'][i],dataset_dict['stlat'][i],dataset_dict['hyplon'][i], dataset_dict['hyplat'][i], dataset_dict['hypdepth (km)'][i], dataset_dict['hypdist'][i],saout))


# Make main dataframe
main_df = pd.DataFrame(data=dataset_dict)
# Save df to file:
main_df.to_csv(flatfile_path,index=False)














