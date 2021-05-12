#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 14:47:20 2021

@author: chris
"""

def compute_openquake_distances(slip_model,gm_data,hypo_location,slip_rake=None):
    '''
    Compute distances for openquake rupture class, from a pandas dataframe
    Input:
        slip_model:             Pandas dataframe with slip model, formatted like D. Melgar's slip models
        gm_data:                Pandas dataframe with ground motion data, formatted like Mexico 2017 files from D. Melgar
        hypo_location:          Hypocenter location: [hypo_lon, hypo_lat, hypo_depth]
        slip_rake:              Average slip rake.  If slip_rake==None, compute this.
        
    '''
    
    import numpy as np
    from pyproj import Geod
    
    
    # Get strike/dip:
    tehuantepec_strike = slip_model['Strike']
    tehuantepec_dip = slip_model['Dip']
    slip_rake = np.full_like(tehuantepec_strike,slip_rake)
    
    # Now get min distance for each statioN:
    rrup = []
    rhypo = []
    
    for stationi in range(len(np.genfromtxt(gm_data, usecols=1))):  #len(np.genfromtxt(gm_data, usecols=1))
        # Get station location:
        # i_stlon = gm_data['Slon'][stationi]
        # i_stlat = gm_data['Slat'][stationi]
        i_stlon = np.genfromtxt(gm_data, usecols = 1)[stationi]
        i_stlat = np.genfromtxt(gm_data, usecols = 2)[stationi]
        
        # Get projection:
        g = Geod(ellps='WGS84')
        
        ###############    
        # Get distances:
        az,backaz,horizdist = g.inv(i_stlon,i_stlat,hypo_location[0],hypo_location[1])
        
        # Get overall:
        i_rhypo = np.sqrt(horizdist**2 + (hypo_location[2]*1000)**2)
        
        # Append to list:
        rhypo.append(i_rhypo)
        
        #####################################
        # For Rrup:
        
        # Turn into arrays length of slip model subfautls:
        i_stlon = np.full_like(slip_model['Longitude'],i_stlon)
        i_stlat = np.full_like(slip_model['Latitude'],i_stlat)
        
        # Get slip model lat/lon:
        slip_lon = slip_model['Longitude'].to_numpy()
        slip_lat = slip_model['Latitude'].to_numpy()
        # Get slip depth in m:
        slip_depth = (slip_model['Depth'].to_numpy())*1000
        
        # Get horizontal distances:
        i_az,i_backaz,i_horizontaldist = g.inv(i_stlon,i_stlat,slip_lon,slip_lat)
    
        # Get distances:
        i_dist = np.sqrt(i_horizontaldist**2 + slip_depth**2)
        
        # Find minimum distance:
        i_mindist = np.min(i_dist)
    
        # Append:
        rrup.append(i_mindist)
    
        
    # Turn minimum distance into an array, convert to km:
    rrup = np.array(rrup)/1000
    rhypo = np.array(rhypo)/1000

    # Return:
    return rrup, rhypo

#%%
def extract_vs30_forpandas(dataframe,xytmpfile,xy_vs30tmpfile,vs30_grdfile):
    '''
    Extract Vs30 from a proxy-based grd file for a list of station lon/lats, from a pandas dataframe
    (formatted as in D.Melgar's Mexico 2017 ground motion files)
    Input:
        dataframe:              Pandas dataframe, must have station lon/lat named as 'stlat'/'stlon'
        xytmpfile:              Path to tmp file for the xy station lon/lat
        xy_vs30tmpfile:         Path to the tmp file with xy and vs30 values
        vs30_grdfile:           Path to Vs30 grd file
    Output:
        vs30:                   Array with Vs30 for each row in dataframe
    '''
    
    import pandas as pd
    import numpy as np
    import subprocess
    from shlex import split
    

    # Write out station lat lon to tmp file:
    
    #xy_out = np.c_[dataframe['Slon'],dataframe['Slat']]
    xy_out = np.c_[np.genfromtxt(dataframe, usecols = 1),np.genfromtxt(dataframe, usecols = 2)]
    
    np.savetxt(xytmpfile,xy_out,fmt='%.8f\t%.8f')
    
    # Make command:
    command = split('gmt grdtrack ' + xytmpfile + ' -G' + vs30_grdfile + ' > ' + xy_vs30tmpfile)
    
    # Run subprocess:
    p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    out,err=p.communicate()
    
    # Read back in:
    vs30 = np.genfromtxt(xy_vs30tmpfile,usecols=[2])

    # Return
    return vs30


#%%


import pandas as pd
import numpy as np

slip_model_file = '/Users/chris/Documents/Valerie_work/Crete/Crete_slip_model.txt'
slip_data = pd.read_csv(slip_model_file, delimiter="\s+", skiprows=14, header = None)
slip_data.columns = ["Latitude", "Longitude", "Depth", "Slip(cm)", "Rake", "Strike", "Dip"]

station_flatfile_path = '/Users/chris/Documents/Valerie_work/Crete/Data_download/events/Main_event/collected_flatfile_main_event.csv'
station_flatfile = pd.read_csv(station_flatfile_path, header='infer')

Rhypo = np.array([])
Rrup = np.array([])

hypo_location = [25.71, 34.182, 10]


#%%

# Rrup, Rhypo = compute_openquake_distances(slip_data, station_flatfile, hypo_location)
# vs30_file_path = '/Users/chris/Documents/Valerie_work/global_vs30.grd'
# xytmpfile_path = '/Users/chris/Documents/Valerie_work/Crete/xytmpfile'
# xy_vs30_tmpfile_path = '/Users/chris/Documents/Valerie_work/Crete/xy_vs30_tmpfile'

# vs30 = extract_vs30_forpandas(station_flatfile, xytmpfile_path, xy_vs30_tmpfile_path, vs30_file_path) 

#%%

#%%

def oq_ask2014(M,Rrup,predictive_parameter='pga',vs30=760,ztor=7.13,rake=0.0,dip=90.0,width=10.0,z1pt0 = 0.05):
    '''
    Compute the predicted ground motions with Abrahamson, Silva, and Kamai 2014 model
        from OpenQuake engine.  Assuming all events are a point source.
    Input:
        M:                      Float or array with magnitudes to compute
        Rrup:                   Float or array with rrups - if it's an array, it should be np.logspace(log10(start),log10(stop),num)
        predictive_parameter:   Predictive parameter to compute: 'pga','pgv', or float with SA period (i.e., 1.0).  Default: 'pga'
        vs30:                   Value or array with Vs30 to use.  Default: 760. 
        ztor:                   Depth to top of rupture. Default: 7.13 from Annemarie. ASSUMPTION IS RRUP > ZTOR!!!!
        rake:                   Rake.  Default: 0.0 degrees.
        dip:                    Dip.  Default: 90.0 degrees.
        width:                  Fault width.  Default: 10.0
        z1pt0:                  Soil depth to Vs = 1.0km/s, in km.  Default: 0.05.
    Output:
        lmean_ask14:            Mean ground motion. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        sd_ask14:               Standard deviation. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        
    '''
    from openquake.hazardlib.gsim.abrahamson_2014 import AbrahamsonEtAl2014
    from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
    from openquake.hazardlib import imt, const
    from openquake.hazardlib.gsim.base import RuptureContext
    from openquake.hazardlib.gsim.base import DistancesContext
    from openquake.hazardlib.gsim.base import SitesContext
    import numpy as np
    
    # Initiate which model:
    ASK14 = AbrahamsonEtAl2014()
    B14  = BooreEtAl2014()

    # Predictive parameter:
    if predictive_parameter=='pga':
        IMT = imt.PGA()
    elif predictive_parameter=='pgv':
        IMT = imt.PGV()
    else:
        IMT = imt.SA(predictive_parameter)
        
    # Initiate the rupture, distances, and sites objects:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    sctx_rock = SitesContext()

    # Fill the rupture context...assuming rake is 0, dip is 90,
    rctx.rake = rake
    rctx.dip = dip
    rctx.ztor = ztor
    rctx.width = width   
    
    # Scenario I: If M and Rrup are both single values:
    #   Then set the magnitude as a float, and the rrup/distance as an array
    #   of one value
    if isinstance(M,float) & isinstance(Rrup,float):
        rctx.mag = M
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)

        # Then compute everything else...
	#    Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        #   Set site parameters
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # Compute prediction
        lmean_ask14, sd_ask14 = ASK14.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_ask14, sd_ask14

    
    # Scenario II: If M is a single value and Rrup is an array:
    if isinstance(M,float) & isinstance(Rrup,np.ndarray):
        # Set them as intended...Rrup should be in logspace
        rctx.mag = M
        dctx.rrup = Rrup
        
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        lmean_ask14, sd_ask14 = ASK14.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_ask14, sd_ask14
    
    
    # Scenario III: If M is an array and Rrup is a single value:
    if isinstance(M,np.ndarray) & isinstance(Rrup,float):
        # Set dctx to be a single value array, like in scenario I:
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_ask14 = np.zeros_like(M)
        sd_ask14 = np.zeros_like(M)
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_ask14, i_sd_ask14 = ASK14.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_ask14[iMag] = i_lmean_ask14
            sd_ask14[iMag] = i_sd_ask14[0]
            
        return lmean_ask14,sd_ask14
    
     
    # If both M and Rrup are arrays: 
    if isinstance(M,np.ndarray) & isinstance(Rrup,np.ndarray):
        # Set dctx to be its array as intended:
        dctx.rrup = Rrup
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:	
        #dctx.rjb = np.log10(np.sqrt((10**dctx.rrup)**2 - rctx.ztor**2))
        dctx.rjb = dctx.rrup
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_ask14 = np.zeros((len(M),len(Rrup)))
        sd_ask14 = np.zeros((len(M),len(Rrup)))
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_ask14, i_sd_ask14 = ASK14.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_ask14[iMag,:] = i_lmean_ask14
            sd_ask14[iMag,:] = i_sd_ask14[0]
        
        return lmean_ask14,sd_ask14



#%%

def oq_boore2014(M,Rrup,predictive_parameter='pga',vs30=760,ztor=7.13,rake=0.0,dip=90.0,width=10.0,z1pt0 = 0.05):
    '''
    Compute the predicted ground motions with Abrahamson, Silva, and Kamai 2014 model
        from OpenQuake engine.  Assuming all events are a point source.
    Input:
        M:                      Float or array with magnitudes to compute
        Rrup:                   Float or array with rrups - if it's an array, it should be np.logspace(log10(start),log10(stop),num)
        predictive_parameter:   Predictive parameter to compute: 'pga','pgv', or float with SA period (i.e., 1.0).  Default: 'pga'
        vs30:                   Value or array with Vs30 to use.  Default: 760. 
        ztor:                   Depth to top of rupture. Default: 7.13 from Annemarie. ASSUMPTION IS RRUP > ZTOR!!!!
        rake:                   Rake.  Default: 0.0 degrees.
        dip:                    Dip.  Default: 90.0 degrees.
        width:                  Fault width.  Default: 10.0
        z1pt0:                  Soil depth to Vs = 1.0km/s, in km.  Default: 0.05.
    Output:
        lmean_b14:            Mean ground motion. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        sd_boore14:               Standard deviation. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        
    '''
    from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
    from openquake.hazardlib import imt, const
    from openquake.hazardlib.gsim.base import RuptureContext
    from openquake.hazardlib.gsim.base import DistancesContext
    from openquake.hazardlib.gsim.base import SitesContext
    import numpy as np
    
    # Initiate which model:
    B14  = BooreEtAl2014()

    # Predictive parameter:
    if predictive_parameter=='pga':
        IMT = imt.PGA()
    elif predictive_parameter=='pgv':
        IMT = imt.PGV()
    else:
        IMT = imt.SA(predictive_parameter)
        
    # Initiate the rupture, distances, and sites objects:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    sctx_rock = SitesContext()

    # Fill the rupture context...assuming rake is 0, dip is 90,
    rctx.rake = rake
    rctx.dip = dip
    rctx.ztor = ztor
    rctx.width = width   
    
    # Scenario I: If M and Rrup are both single values:
    #   Then set the magnitude as a float, and the rrup/distance as an array
    #   of one value
    if isinstance(M,float) & isinstance(Rrup,float):
        rctx.mag = M
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)

        # Then compute everything else...
	#    Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        #   Set site parameters
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # Compute prediction
        lmean_b14, sd_boore14 = B14.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_b14, sd_boore14

    
    # Scenario II: If M is a single value and Rrup is an array:
    if isinstance(M,float) & isinstance(Rrup,np.ndarray):
        # Set them as intended...Rrup should be in logspace
        rctx.mag = M
        dctx.rrup = Rrup
        
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        lmean_b14, sd_boore14 = B14.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_b14, sd_boore14
    
    
    # Scenario III: If M is an array and Rrup is a single value:
    if isinstance(M,np.ndarray) & isinstance(Rrup,float):
        # Set dctx to be a single value array, like in scenario I:
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_b14 = np.zeros_like(M)
        sd_boore14 = np.zeros_like(M)
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_b14, i_sd_boore14 = B14.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_b14[iMag] = i_lmean_b14
            sd_boore14[iMag] = i_sd_boore14[0]
            
        return lmean_b14,sd_boore14
    
     
    # If both M and Rrup are arrays: 
    if isinstance(M,np.ndarray) & isinstance(Rrup,np.ndarray):
        # Set dctx to be its array as intended:
        dctx.rrup = Rrup
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:	
        #dctx.rjb = np.log10(np.sqrt((10**dctx.rrup)**2 - rctx.ztor**2))
        dctx.rjb = dctx.rrup
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_b14 = np.zeros((len(M),len(Rrup)))
        sd_boore14 = np.zeros((len(M),len(Rrup)))
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_b14, i_sd_boore14 = B14.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_b14[iMag,:] = i_lmean_b14
            sd_boore14[iMag,:] = i_sd_boore14[0]
        
        return lmean_b14,sd_boore14

#%%

def oq_Chioy2014(M,Rrup,predictive_parameter='pga',vs30=760,ztor=7.13,rake=0.0,dip=90.0,width=10.0,z1pt0 = 0.05):
    '''
    Compute the predicted ground motions with Abrahamson, Silva, and Kamai 2014 model
        from OpenQuake engine.  Assuming all events are a point source.
    Input:
        M:                      Float or array with magnitudes to compute
        Rrup:                   Float or array with rrups - if it's an array, it should be np.logspace(log10(start),log10(stop),num)
        predictive_parameter:   Predictive parameter to compute: 'pga','pgv', or float with SA period (i.e., 1.0).  Default: 'pga'
        vs30:                   Value or array with Vs30 to use.  Default: 760. 
        ztor:                   Depth to top of rupture. Default: 7.13 from Annemarie. ASSUMPTION IS RRUP > ZTOR!!!!
        rake:                   Rake.  Default: 0.0 degrees.
        dip:                    Dip.  Default: 90.0 degrees.
        width:                  Fault width.  Default: 10.0
        z1pt0:                  Soil depth to Vs = 1.0km/s, in km.  Default: 0.05.
    Output:
        lmean_cy14:            Mean ground motion. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        sd_cy14:               Standard deviation. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        
    '''
    from openquake.hazardlib.gsim.chiou_youngs_2014 import ChiouYoungs2014
    from openquake.hazardlib import imt, const
    from openquake.hazardlib.gsim.base import RuptureContext
    from openquake.hazardlib.gsim.base import DistancesContext
    from openquake.hazardlib.gsim.base import SitesContext
    import numpy as np
    
    # Initiate which model:
    CY14  = ChiouYoungs2014()

    # Predictive parameter:
    if predictive_parameter=='pga':
        IMT = imt.PGA()
    elif predictive_parameter=='pgv':
        IMT = imt.PGV()
    else:
        IMT = imt.SA(predictive_parameter)
        
    # Initiate the rupture, distances, and sites objects:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    sctx_rock = SitesContext()

    # Fill the rupture context...assuming rake is 0, dip is 90,
    rctx.rake = rake
    rctx.dip = dip
    rctx.ztor = ztor
    rctx.width = width   
    
    # Scenario I: If M and Rrup are both single values:
    #   Then set the magnitude as a float, and the rrup/distance as an array
    #   of one value
    if isinstance(M,float) & isinstance(Rrup,float):
        rctx.mag = M
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)

        # Then compute everything else...
	#    Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        #   Set site parameters
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # Compute prediction
        lmean_cy14, sd_cy14 = CY14.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_cy14, sd_cy14

    
    # Scenario II: If M is a single value and Rrup is an array:
    if isinstance(M,float) & isinstance(Rrup,np.ndarray):
        # Set them as intended...Rrup should be in logspace
        rctx.mag = M
        dctx.rrup = Rrup
        
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        lmean_cy14, sd_cy14 = CY14.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_cy14, sd_cy14
    
    
    # Scenario III: If M is an array and Rrup is a single value:
    if isinstance(M,np.ndarray) & isinstance(Rrup,float):
        # Set dctx to be a single value array, like in scenario I:
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_cy14 = np.zeros_like(M)
        sd_cy14 = np.zeros_like(M)
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_cy14, i_sd_cy14 = CY14.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_cy14[iMag] = i_lmean_cy14
            sd_cy14[iMag] = i_sd_cy14[0]
            
        return lmean_cy14,sd_cy14
    
     
    # If both M and Rrup are arrays: 
    if isinstance(M,np.ndarray) & isinstance(Rrup,np.ndarray):
        # Set dctx to be its array as intended:
        dctx.rrup = Rrup
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:	
        #dctx.rjb = np.log10(np.sqrt((10**dctx.rrup)**2 - rctx.ztor**2))
        dctx.rjb = dctx.rrup
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_cy14 = np.zeros((len(M),len(Rrup)))
        sd_cy14 = np.zeros((len(M),len(Rrup)))
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_cy14, i_sd_cy14 = CY14.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_cy14[iMag,:] = i_lmean_cy14
            sd_cy14[iMag,:] = i_sd_cy14[0]
        
        return lmean_cy14,sd_cy14


#%%

def oq_Skarlatoudis2013(M,Rrup, hypo_depth=10,predictive_parameter='pga',vs30=760,ztor=7.13,rake=0.0,dip=90.0,width=10.0,z1pt0 = 0.05):
    '''
    Compute the predicted ground motions with Abrahamson, Silva, and Kamai 2014 model
        from OpenQuake engine.  Assuming all events are a point source.
    Input:
        M:                      Float or array with magnitudes to compute
        Rrup:                   Float or array with rrups - if it's an array, it should be np.logspace(log10(start),log10(stop),num)
        predictive_parameter:   Predictive parameter to compute: 'pga','pgv', or float with SA period (i.e., 1.0).  Default: 'pga'
        vs30:                   Value or array with Vs30 to use.  Default: 760. 
        ztor:                   Depth to top of rupture. Default: 7.13 from Annemarie. ASSUMPTION IS RRUP > ZTOR!!!!
        rake:                   Rake.  Default: 0.0 degrees.
        dip:                    Dip.  Default: 90.0 degrees.
        width:                  Fault width.  Default: 10.0
        z1pt0:                  Soil depth to Vs = 1.0km/s, in km.  Default: 0.05.
    Output:
        lmean_cy14:            Mean ground motion. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        sd_cy14:               Standard deviation. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        
    '''
    from skarlatoudis_2013 import SkarlatoudisEtAlSSlab2013
    from openquake.hazardlib import imt, const
    from openquake.hazardlib.gsim.base import RuptureContext
    from openquake.hazardlib.gsim.base import DistancesContext
    from openquake.hazardlib.gsim.base import SitesContext
    import numpy as np
    
    # Initiate which model:
    SK13  = SkarlatoudisEtAlSSlab2013()

    # Predictive parameter:
    if predictive_parameter=='pga':
        IMT = imt.PGA()
    elif predictive_parameter=='pgv':
        IMT = imt.PGV()
    else:
        IMT = imt.SA(predictive_parameter)
        
    # Initiate the rupture, distances, and sites objects:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    sctx_rock = SitesContext()

    # Fill the rupture context...assuming rake is 0, dip is 90,
    rctx.rake = rake
    rctx.dip = dip
    rctx.ztor = ztor
    rctx.width = width   
    rctx.hypo_depth = hypo_depth
    sctx.backarc = np.full_like(Rrup,False,dtype='bool')

    
    
    # Scenario I: If M and Rrup are both single values:
    #   Then set the magnitude as a float, and the rrup/distance as an array
    #   of one value
    if isinstance(M,float) & isinstance(Rrup,float):
        rctx.mag = M
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)

        # Then compute everything else...
	#    Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        rctx.rhypo = dctx.rrup
        #   Set site parameters
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # Compute prediction
        lmean_sk13, sd_sk13 = SK13.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_sk13, sd_sk13

    
    # Scenario II: If M is a single value and Rrup is an array:
    if isinstance(M,float) & isinstance(Rrup,np.ndarray):
        # Set them as intended...Rrup should be in logspace
        rctx.mag = M
        dctx.rrup = Rrup
        
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        rctx.rhypo = dctx.rrup
        
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        lmean_sk13, sd_sk13 = SK13.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_sk13, sd_sk13
    
    
    # Scenario III: If M is an array and Rrup is a single value:
    if isinstance(M,np.ndarray) & isinstance(Rrup,float):
        # Set dctx to be a single value array, like in scenario I:
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        rctx.rhypo = dctx.rrup
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_sk13 = np.zeros_like(M)
        sd_sk13 = np.zeros_like(M)
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_sk13, i_sd_sk13 = SK13.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_sk13[iMag] = i_lmean_sk13
            sd_sk13[iMag] = i_sd_sk13[0]
            
        return lmean_sk13,sd_sk13
    
     
    # If both M and Rrup are arrays: 
    if isinstance(M,np.ndarray) & isinstance(Rrup,np.ndarray):
        # Set dctx to be its array as intended:
        dctx.rrup = Rrup
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:	
        #dctx.rjb = np.log10(np.sqrt((10**dctx.rrup)**2 - rctx.ztor**2))
        dctx.rjb = dctx.rrup
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        rctx.rhypo = dctx.rrup
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_sk13 = np.zeros((len(M),len(Rrup)))
        sd_sk13 = np.zeros((len(M),len(Rrup)))
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_sk13, i_sd_sk13 = SK13.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_sk13[iMag,:] = i_lmean_sk13
            sd_sk13[iMag,:] = i_sd_sk13[0]
        
        return lmean_sk13,sd_sk13
#%%


def oq_Zhao_2006(M,Rrup,hypo_depth=10, predictive_parameter='pga',vs30=760,ztor=7.13,rake=0.0,dip=90.0,width=10.0,z1pt0 = 0.05):
    '''
    Compute the predicted ground motions with Abrahamson, Silva, and Kamai 2014 model
        from OpenQuake engine.  Assuming all events are a point source.
    Input:
        M:                      Float or array with magnitudes to compute
        Rrup:                   Float or array with rrups - if it's an array, it should be np.logspace(log10(start),log10(stop),num)
        predictive_parameter:   Predictive parameter to compute: 'pga','pgv', or float with SA period (i.e., 1.0).  Default: 'pga'
        vs30:                   Value or array with Vs30 to use.  Default: 760. 
        ztor:                   Depth to top of rupture. Default: 7.13 from Annemarie. ASSUMPTION IS RRUP > ZTOR!!!!
        rake:                   Rake.  Default: 0.0 degrees.
        dip:                    Dip.  Default: 90.0 degrees.
        width:                  Fault width.  Default: 10.0
        z1pt0:                  Soil depth to Vs = 1.0km/s, in km.  Default: 0.05.
    Output:
        lmean_cy14:            Mean ground motion. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        sd_cy14:               Standard deviation. If M and Rrup floats returns float, if M float and Rrup array returns array like Rrup,
                                    if M array and Rrup float returns array like M, if M and rrup arrays returns array like len(M) x len(Rrup)
        
    '''
    from openquake.hazardlib.gsim.zhao_2006 import ZhaoEtAl2006SSlab
    from openquake.hazardlib import imt, const
    from openquake.hazardlib.gsim.base import RuptureContext
    from openquake.hazardlib.gsim.base import DistancesContext
    from openquake.hazardlib.gsim.base import SitesContext
    import numpy as np
    
    # Initiate which model:
    Z16  = ZhaoEtAl2006SSlab()

    # Predictive parameter:
    if predictive_parameter=='pga':
        IMT = imt.PGA()
    elif predictive_parameter=='pgv':
        IMT = imt.PGV()
    else:
        IMT = imt.SA(predictive_parameter)
        
    # Initiate the rupture, distances, and sites objects:
    rctx = RuptureContext()
    dctx = DistancesContext()
    sctx = SitesContext()
    sctx_rock = SitesContext()

    # Fill the rupture context...assuming rake is 0, dip is 90,
    rctx.rake = rake
    rctx.dip = dip
    rctx.ztor = ztor
    rctx.width = width   
    rctx.hypo_depth = hypo_depth
    # Scenario I: If M and Rrup are both single values:
    #   Then set the magnitude as a float, and the rrup/distance as an array
    #   of one value
    if isinstance(M,float) & isinstance(Rrup,float):
        rctx.mag = M
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)

        # Then compute everything else...
	#    Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        #   Set site parameters
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # Compute prediction
        lmean_z16, sd_z16 = Z16.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_z16, sd_z16

    
    # Scenario II: If M is a single value and Rrup is an array:
    if isinstance(M,float) & isinstance(Rrup,np.ndarray):
        # Set them as intended...Rrup should be in logspace
        rctx.mag = M
        dctx.rrup = Rrup
        
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        lmean_z16, sd_z16 = Z16.get_mean_and_stddevs(
            sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
            
        return lmean_z16, sd_z16
    
    
    # Scenario III: If M is an array and Rrup is a single value:
    if isinstance(M,np.ndarray) & isinstance(Rrup,float):
        # Set dctx to be a single value array, like in scenario I:
        dctx.rrup = np.logspace(np.log10(Rrup),np.log10(Rrup),1)
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:
        dctx.rjb = np.sqrt(dctx.rrup**2 - rctx.ztor**2)
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_z16 = np.zeros_like(M)
        sd_z16 = np.zeros_like(M)
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_z16, i_sd_z16 = Z16.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_z16[iMag] = i_lmean_z16
            sd_z16[iMag] = i_sd_z16[0]
            
        return lmean_z16,sd_z16
    
     
    # If both M and Rrup are arrays: 
    if isinstance(M,np.ndarray) & isinstance(Rrup,np.ndarray):
        # Set dctx to be its array as intended:
        dctx.rrup = Rrup
        
        # The rest of dctx depends only on rrup, as wella s site, so populate those:
	# Assuming average ztor, get rjb:	
        #dctx.rjb = np.log10(np.sqrt((10**dctx.rrup)**2 - rctx.ztor**2))
        dctx.rjb = dctx.rrup
        dctx.rhypo = dctx.rrup
        dctx.rx = dctx.rjb
        dctx.ry0 = dctx.rx
        
        # Site: 
        sctx.vs30 = np.ones_like(dctx.rrup) * vs30
        sctx.vs30measured = np.full_like(dctx.rrup, False, dtype='bool')
        sctx.z1pt0 = np.ones_like(dctx.rrup) * z1pt0
        
        # But rctx depends on M and can only take a float, so will have to run many times.
        # Initiate mean and std lists:
        lmean_z16 = np.zeros((len(M),len(Rrup)))
        sd_z16 = np.zeros((len(M),len(Rrup)))
        
        # Then loop over M's for rctx:
        for iMag in range(len(M)):
            # Set mag:
            rctx.mag = M[iMag]
            
            # Set 
            i_lmean_z16, i_sd_z16 = Z16.get_mean_and_stddevs(
                sctx, rctx, dctx, IMT, [const.StdDev.TOTAL])
                
            lmean_z16[iMag,:] = i_lmean_z16
            sd_z16[iMag,:] = i_sd_z16[0]
        
        return lmean_z16,sd_z16






# #%%
# # coeff_file = '/Users/chris/Documents/Valerie_work/Crete/Abrahamson_2014_coefficients.txt'
# # vs30_file = np.genfromtxt('/Users/chris/Documents/Valerie_work/Crete/xy_vs30_tmpfile')[:,2]
# kk = pd.read_csv('/Users/chris/Documents/Valerie_work/Crete/obs_IMs_main**.csv', header = 'infer' )
# Rrup_1 = np.array([np.log10(5), np.log10(10), np.log10(50), np.log10(100),np.log10(200),np.log10(300),np.log10(400) ,np.log10(500),np.log10(600),np.log10(700)])
# lmean_ask14, sd_ask14 = oq_ask2014(6.5, np.log10(Rrup))
# # lmean_ask14 = np.log10(np.exp(lmean_ask14))

#%%

def compute_baltay_anza_fixeddist(Mw,Rhyp):
    '''
    Given magnitude, compute PGA for the GMPE of Baltay et al. (2017) for fixed distance
    Input:
        Mw:             Array with magnitudes to compute for
        Rhyp:           Value of fixed hypocentral distance to plot
    Output:
        ln_PGA:         Array with PGA values in ln PGA
    '''
    
    import numpy as np
    
    log10pga = -6.13 + 1.5*Mw - np.log10(Rhyp)
    
    pga = 10**log10pga
    
    ln_pga = np.log(pga)
    
    return ln_pga

#%%

#P_pga = compute_baltay_anza_fixeddist(station_flatfile['Mag'].values, Rhypo)

#%%

def peakResponse(resp):
    '''
    Function borrowed from pyrotd.
    
    Compute the maximum absolute value of a response.
    Parameters
    ----------
    resp: numpy.array
        time series of a response
    Returns
    -------
    peakResponse: float
        peak response
    '''
    from numpy import max,abs
    
    return max(abs(resp))

#%%

def oscillatorTimeSeries(freq, fourierAmp, oscFreq, oscDamping):
    '''
    Function borrowed from pyrotd.
    
    Compute the time series response of an oscillator.
    Parameters
    ----------
    freq: numpy.array
        frequency of the Fourier acceleration spectrum [Hz]
    fourierAmp: numpy.array
        Fourier acceleration spectrum [g-sec]
    oscFreq: float
        frequency of the oscillator [Hz]
    oscDamping: float
        damping of the oscillator [decimal]
    Returns
    -------
    response: numpy.array
        time series response of the oscillator
    '''
    
    from numpy import power,fft
    
    # Single-degree of freedom transfer function
    h = (-power(oscFreq, 2.)
            / ((power(freq, 2.) - power(oscFreq, 2.))
                - 2.j * oscDamping * oscFreq * freq))

    # Adjust the maximum frequency considered. The maximum frequency is 5
    # times the oscillator frequency. This provides that at the oscillator
    # frequency there are at least tenth samples per wavelength.
    n = len(fourierAmp)
    m = max(n, int(2. * oscFreq / freq[1]))
    scale = float(m) / float(n)

    # Scale factor is applied to correct the amplitude of the motion for the
    # change in number of points
    return scale * fft.irfft(fourierAmp * h, 2 * (m-1))

#%%

def rotateTimeSeries(foo, bar, angle):
    '''
    Function borrowed from pyrotd.
    
    Compute the rotated time series.
    Parameters
    ----------
    foo: numpy.array
        first time series
    bar: numpy.array
        second time series that is perpendicular to the first
    Returns
    -------
    foobar: numpy.array
        time series rotated by the specified angle
    '''

    angleRad = np.radians(angle)
    # Rotate the time series using a vector rotation
    return foo * np.cos(angleRad) + bar * np.sin(angleRad)

#%%

def rotatedResponseSpectrum(timeStep, accelA, accelB, oscFreqs, oscDamping=0.05,
        percentiles=[50], angles=np.arange(0, 180, step=1)):
    '''
    Function borrowed from pyrotd.
    
    Compute the response spectrum for a time series.
    Parameters
    ----------
    timeStep: float
        time step of the time series [s]
    accelA: numpy.array
        acceleration time series of the first motion [g]
    accelB: numpy.array
        acceleration time series of the second motion that is perpendicular to the first motion [g]
    oscFreqs: numpy.array
        natural frequency of the oscillators [Hz]
    oscDamping: float
        damping of the oscillator [decimal]. Default of 0.05 (i.e., 5%)
    percentiles: numpy.array
        percentiles to return. Default of [0, 50, 100],
    angles: numpy.array
        angles to which to compute the rotated time series. Default of
        np.arange(0, 180, step=1) (i.e., 0, 1, 2, .., 179).
    Returns
    -------
    oscResps: list(numpy.array)
        computed psuedo-spectral acceleartion [g] at each of the percentiles
    '''

    from numpy import array

    assert len(accelA) == len(accelB), 'Time series not equal lengths!'

    # Compute the Fourier amplitude spectra
    fourierAmps = [np.fft.rfft(accelA), np.fft.rfft(accelB)]
    freq = np.linspace(0, 1./(2 * timeStep), num=fourierAmps[0].size)

    values = []
    for i, oscFreq in enumerate(oscFreqs):
        # Compute the oscillator responses
        oscResps = [oscillatorTimeSeries(freq, fa, oscFreq, oscDamping)
                for fa in fourierAmps]

        # Compute the rotated values of the oscillator response
        vals,orients = rotatedPercentiles(oscResps[0], oscResps[1], angles, percentiles)
        values.append(vals[0])

    # Reorganzie the arrays grouping by the percentile
#    oscResps = [np.array([v[i] for v in values],
#        dtype=[('value', '<f8'), ('orientation', '<f8')]) for i in range(len(percentiles))]

    return array(values)

def rotatedPercentiles(accelA, accelB, angles, percentiles=[50]):
    '''
    Function borrowed from pyrotd.
    
    Compute the response spectrum for a time series.
    Parameters
    ----------
    accelA: numpy.array
        first time series
    accelB: numpy.array
        second time series that is perpendicular to the first
    angles: numpy.array
        angles to which to compute the rotated time series
    percentiles: numpy.array
        percentiles to return
    Returns
    -------
    values: numpy.array
        rotated values and orientations corresponding to the percentiles
    '''
    assert all(0 <= p <= 100 for p in percentiles), 'Invalid percentiles.'

    # Compute the response for each of the specified angles and sort this array
    # based on the response
    rotated = np.array(
            [(a, peakResponse(rotateTimeSeries(accelA, accelB, a))) for a in angles],
            dtype=[('angle', '<f8'), ('value', '<f8')])
    rotated.sort(order='value')

    # Interpolate the percentile from the values
    values = np.interp(percentiles,
            np.linspace(0, 100, len(angles)), rotated['value'])

    # Can only return the orientations for the minimum and maximum value as the
    # orientation is not unique (i.e., two values correspond to the 50%
    # percentile).
    orientationMap = {
            0 : rotated['angle'][0],
            100 : rotated['angle'][-1],
            }
    orientations = [orientationMap.get(p, np.nan) for p in percentiles] 

#    out=np.array(zip(values, orientations), dtype=[('value', '<f8'), ('orientation', '<f8')])

    return values,orientations

#%%


