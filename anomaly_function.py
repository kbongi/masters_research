# Functions
    
    # function to read in all the model files, **note it assumes a time period from 1850-2014 with 1980 timesteps
def read_models(institution_dir, variable_dir, start_date, end_date):
    """ Read in all the CMIP6 histtorical run model files of specified variable from specified start and end date.
        **note it assumes a time period from 1850-2014 with 1980 timesteps 
        
        Args:
        institution_directory (str): directory of CMIP6 institutions
        variable_directory (str): climate variable (e.g. tas, pr, sst, ts etc.)
        start_date (date_str): start date for data
        end_date (date_str): end date for data
    """
    
    # import relevant libraries
    import os
    import xarray as xr
    import pandas as pd
    # store all institutions found in the institution directory
    institution_list = os.listdir(f'{institution_dir}')
    
    # creates a dictionary containing the model and model path
    models = {}

    # find the models from each instituion and store them in a list
    for institution in institution_list:
        model_list = os.listdir(f'{institution_dir}{institution}')
        # find the 
        for model in model_list:
            # check if the historical model with the right variable exists and if so save the version number for the file
            if os.path.exists(f'{institution_dir}{institution}/{model}{variable_dir}'):
                version = os.listdir(f'{institution_dir}{institution}/{model}{variable_dir}')
                # for each version, save the path and then store with the model in a dictionary 'models'
                for v in version:
                    path = f'{institution_dir}{institution}/{model}{variable_dir}{v}/*.nc'
                    models[model] = path
                     
    # Prints the number of models loaded into the dictionary
    print(f'{len(models)} model paths found and loaded into the dictionary "models"') 
    
    # Now open each dataset, and store the dataset and model name in two arrays, 'names' and 'ds'.  
    # The try-except method allows all other files to be read even if one file does not exist or has some issues. 
    # By deleting the time and bnds coordinates we are removing any differences in time.  
    # (We add the time back later). 
    names = []
    ds = []

    for name, path in models.items():
        try:
            d = xr.open_mfdataset(path, combine='by_coords', chunks={'time': 12})
            # checks if there is data for each month in the time period, and if so stores the model file data
            if len(d['time'])==1980:
                # remove any differences in time
                del d['time_bnds']
                del d['time']
                # select times from the specified period
                time_month = pd.date_range(start = f'{start_date}',end = f'{end_date}', freq ='M')
                # add the time coordinate back in
                d.coords['time'] = time_month
                ds.append(d)
                names.append(name)
                #print(name, path)
            else:
                print(f'Model {name} has different time so is now removed')
        except OSError:
            # No files read, move on to the next
            print(f'Path for {name} does not exist')
            continue

    # Combine the individual data sets into a single xarray dataset, with the coordinate
    # 'model' representing the source model
    multi_model = xr.concat(ds, dim='model', coords = 'minimal')
    multi_model.coords['model'] = names
    
    # del multi_model.model['height']
                
    # print the number of models that have been successfully loaded into the xarray
    print(f'{len(multi_model.model)} models have been successfully loaded into an xarray')
    
    return multi_model


# define function to calculate the seasonal mean used in seasonal anomaly calculation:
def seasonal_mean(data):
    """ Calculate the seasonal mean used in seasonal anomaly calculation.  
        
        Args:
        data (xarray): data set of climate variable (e.g tas)
    """
    return data.groupby('time.season').mean()


# function to calculate a seasonal anomaly for a multidimensional xarray over a time period entered by user
def seasonal_anomaly(dataset, start_date, end_date):
    """ Calculate a seasonal anomaly for a multidimensional xarray over a time period entered by user.  
        
        Args:
        dataset (xarray): data set of climate variable (e.g tas)
        start_date (date_str): start date to calculate seasonal anomaly
        end_date (date_str): end date to calculate seasonal anomaly
    """
    # first I need to define a new coordinate (seasonyear) so that december gets counted with the adjoining jan and feb
    seasonyear = (dataset.time.dt.year + (dataset.time.dt.month//12)) 
    dataset.coords['seasonyear'] = seasonyear
    
        
    # group data into seasons and calculate the seasonal mean for each year in the dataset 
    yearly_seasonal = dataset.groupby('seasonyear').apply(seasonal_mean)

    # calculate the mean climatology along each season for the time period 
    clim_seasonal = yearly_seasonal.sel(seasonyear = slice(f'{start_date}',f'{end_date}')).mean(dim = 'seasonyear')

    # calculate the anomaly and returns it as an xarray
    multi_seasonal_anom = (yearly_seasonal - clim_seasonal)
        
    return multi_seasonal_anom


# define function to calculate monthly anomalies for a multidimensional array of models
def monthly_anomaly(dataset, start_date, end_date):
    
    """ Calculate monthly anomalies for a multidimensional array of models.  
        
        Args:
        dataset (xarray): data set of climate variable (e.g tas)
        start_date (date_str): start date of climatology to calculate monthly anomaly
        end_date (date_str): end date of climatology to calculate monthly anomaly
    """
    
    # group the data into months
    variable_monthly = dataset.groupby('time.month')

    # calculate the mean climatology along each month for the time period 1850-1900 
    clim_monthly = dataset.sel(time = slice(f'{start_date}', f'{end_date}')).groupby('time.month').mean(dim = 'time')

    # caclulate the anomalies for each month and return it as an array
    multi_monthly_anom = (variable_monthly - clim_monthly)

    return multi_monthly_anom

# define a function to calculate the mmm plus/minus standard deviation over all models
def std_bounds(dataset, mmm):
    
    """ Calculate the multi-model mean plus or minus the standard deviation over all models.  
        
        Args:
        dataset (xarray): data set of climate variable (e.g tas)
        mmm (xarray): data set that is the multimodel mean of the dataset argument
    """
    
    import xarray as xr
    ds = []
    names = ['lower','upper']
    
    # calculate the std over the model dimension
    std = dataset.std(dim = ['model'])
    
    # calculate the upper and lower bounds
    std_lower = mmm - std
    std_upper = mmm + std
    
    ds.append(std_lower)
    ds.append(std_upper)
        
    std_bnds = xr.concat(ds, dim='bnds', coords = 'minimal')
    std_bnds.coords['bnds'] = names
    
    return std_bnds

# define a function to calculate the mmm plus/minus standard deviation over all models
def std_bounds_MV(dataset, mmm):
    """ Calculate the multi-model mean plus or minus the standard deviation over all models and volcanoes.  
        
        Args:
        dataset (xarray): data set of climate variable (e.g tas)
        mmm (xarray): data set that is the multimodel mean of the dataset argument
    """
    
    import xarray as xr
    
    ds = []
    names = ['lower','upper']
    
    # calculate the std over the model dimension
    std = dataset.std(dim = ['model','volcano'])
    
    # calculate the upper and lower bounds
    std_lower = mmm - std
    std_upper = mmm + std
    
    ds.append(std_lower)
    ds.append(std_upper)
        
    std_bnds = xr.concat(ds, dim='bnds', coords = 'minimal')
    std_bnds.coords['bnds'] = names
    
    return std_bnds
    
    
# define function to reset the time axis for superposed epoch analysis
def reset_time(K, S, A, E, P, names):  
    """ Reset the time axis for data used in the superposed epoch analysis  
        
        Args:
        K (xarray): data set of first eruption 
        S (xarray): data set of second eruption 
        A (xarray): data set of third eruption 
        E (xarray): data set of fourth eruption 
        P (xarray): data set of fifth eruption 
        names (dict): dictionary of names for each of the five eruptions
    """
    import numpy as np
    import xarray as xr
    
    times = np.arange(-60,61)
    # reset all the times so 0 corresponds to the eruption year and month
    K['time'] = times
    S['time'] = times
    A['time'] = times
    E['time'] = times 
    P['time'] = times
    
    # delete the months dimension (but not the time)
    if 'month' in P:
        del K['month']
        del S['month']
        del A['month']
        del E['month']
        del P['month']
        
    # select out the 10 year time frame around each eruption, (5yrs before and after)
    # then combine all eruptions into a single array
    #tas
    #volcanoes = ['Krakatau', 'Santa Maria','Agung','El Chichon', 'Pinatubo']
    ds=[]
    ds.append(K)
    ds.append(S)
    ds.append(A)
    ds.append(E)
    ds.append(P)

    # store all eruptions in an array
    composite = xr.concat(ds, dim='volcano', coords = 'minimal')
    composite.coords['volcano'] = names
    
    return composite


# define function to check if model has a postive or negative anomaly
def anomaly_check(dataset, start_date, end_date):
    """ Check if each model has a postive or negative anomaly
        
        Args:
        dataset (xarray): data set of climate variable(s)
        start_date (date_str): start date
        end_date (date_str): end date 
    """
    import xarray as xr, numpy as np
    
    # select out the desired time period and then take the mean over the time period
    ds_slice = dataset.sel(time=slice(f'{start_date}',f'{end_date}'))
    ds_mean = ds_slice.mean(dim='time')
                                      
    # replace the value with + for positive anomaly and - for negative anomaly 
    ds_mean['tas'] = xr.where(ds_mean.tas > 0, 1, 0)
    ds_mean['pr'] = xr.where(ds_mean.pr > 0, 1, 0)
    
    return ds_mean


# add in the multi-model mean to an array of all the models 
def add_mmm(dataset):
    """ Add in the multi-model mean to an array of all the models. 
        
        Args:
        dataset (xarray): data set of climate variable(s)
    """
    
    import xarray as xr
    
    # calculate the mmm
    mmm_dataset = dataset.mean(dim='model')
    
    # add model dimension to mmm array
    ds = []
    ds.append(mmm_dataset)
    mmm = xr.concat(ds, dim='model', coords = 'minimal')
    mmm.coords['model'] = ['Multi-model mean']
    
    # append the mmm to the rest of the anomaly array
    ds2=[]
    ds2.append(dataset)
    ds2.append(mmm)
    multi_model = xr.concat(ds2, dim='model', coords = 'minimal')
    
    return multi_model
    
    
# find the date of each minimum anomaly
def min_date(anom_dataset, min_dataset):
    """ Find the date when the minimum temperature and rainfall anomaly occurred for each model. 
        
        Args:
        anom_dataset (xarray): data set of tas and pr
        min_dataset (xarray): data set of minimum calues for tas and pr
    """
    import xarray as xr

    # find the index of the minimum anomaly
    min_index = anom_dataset.where(anom_dataset == min_dataset).argmin('time')
    
    # These indices can be converted to time values all at once by using min_index as an array index
    # find the date the minimum value occured for both tas and pr and remove time axis (using drop)
    # (There's a "ghost" time axis in the result which might cause problems, let's get rid of it.)
    # (Our returned values aren't time dependent)
    min_date_tas = anom_dataset.tas.time[min_index.tas].drop('time').drop('month')
    min_date_pr = anom_dataset.pr.time[min_index.pr].drop('time').drop('month')
    
    # combine the two into a dataset 
    min_date = xr.Dataset({
    'tas': min_date_tas,
    'pr': min_date_pr
    })
    
    return min_date


# combine the various event stats into one xarray for convenience 
def combine_stats(std2_count, std3_count, min_dataset, min_date):
    """ Combine the event stats of std counts, minimum anomaly and minimum date into one xarray for convenience.
        
        Args:
        std2_count (xarray): data set of the number of times anomalies exceeded 2 standard deviations
        std3_count (xarray): data set of the number of times anomalies exceeded 3 standard deviations
        min_dataset (xarray): data set of tas and pr
        min_date (xarray): data set of minimum calues for tas and pr
    """
    import xarray as xr 
    
    event_stats = xr.Dataset({
        'Count std': std_count,
        'Count 2*std': std2_count,
        'Count 3*std': std3_count,
        'Min': min_dataset,
        'Min Date': min_date,
    })
    
    return event_stats



   
# calculate the nino 3.4 index 
def nino34(sst_dataset, start_date, end_date, std):
    """ Calculate the NINO34 index from SST values and normalise by dividing by the standard deviation calculate over user specified time period.   
        
        Args:
        sst_dataset (xarray): data set of sea surface temperature values
        start_date (date_str): start date of std climatology
        end_date (date_str): end date of std climatology
        std (int): if std==1, calculate the std and divide NINO34 index by std
    """
    # select out the region for nino34 definition
    region = sst_dataset.sel(lat=slice(-5,5), lon=slice(190,240))
    
    # calculate the mean climatology along each month
    clim = region.sel(time = slice(f'{start_date}', f'{end_date}')).groupby('time.month').mean(dim = ['time','lat','lon'])
    
    # calculate the anomaly using input dates for climatology and take the lat lon mean 
    #anom = monthly_anom_xr(region, f'{start_date}', f'{start_date}').mean(dim=['lat','lon'])
    anom = (region.groupby('time.month') - clim).mean(dim=['lat','lon'])
    
    # chunk the data into groups of 5 timepoints so I can then use rolling mean 
    anom = anom.chunk({'time': 5})
    
    if std == 1:
        # calculate the standard deviation so we can normalise the model data 
        std = region.sel(time = slice(f'{start_date}', f'{end_date}')).mean(dim=['lat', 'lon']).std(dim = ['time'])
        
        # calculate the nino3.4 index using a rolling 5 month mean and normalised by the std
        nino34_index = anom.rolling(time=5).mean() / std
    elif std == 0:
            nino34_index = anom.rolling(time=5).mean()
    
    return nino34_index
    


# check if anomalies are less than 1x, 2x, 3x the standard deviation in the specificed time period post-eruption
def std_count(anom_dataset, mean, std, start_date, end_date):
    """ Count how many times anomalies are less than 1x, 2x, 3x the standard deviation in the specificed time period post-eruption and combine the results into one xarray with 3 columns (one for each count). 
        
        Args:
        anom_dataset (xarray): data set of anomalies
        mean (xarray): data set of the mean anomaly (close to 0)
        std (xarray): data set of one standard deviation 
        start_date (date_str): start date for count period
        end_date (date_str): end date for count period
    """
    import xarray as xr
    
    # end date is the start date plus specified months later 
    #end_date = start_date + (months+1) * pd.offsets.MonthBegin()
    
    # select out time period
    dataset_time = anom_dataset.sel(time = slice(f'{start_date}', f'{end_date}'))
    
    # check if the anomalies are less than 1,2,3 standard deviations in the specificed time periods post eruption
    std1_count = dataset_time.where(dataset_time < mean - 1*std).count('time')
    std2_count = dataset_time.where(dataset_time < mean - 2*std).count('time')
    std3_count = dataset_time.where(dataset_time < mean - 3*std).count('time')
    
    ds=[]
    ds.append(std1_count)
    ds.append(std2_count)
    ds.append(std3_count)
    std_count = xr.concat(ds, dim='std_count', coords = 'minimal')
    std_count.coords['std_count'] = ['1*std', '2*std', '3*std']
    
    return std_count


# check if anomalies are less than a multiple of the standard deviation in the specificed time period post-eruption
def std_check(anom_dataset, mean, std, e_date):
    """Count how many times anomalies are less than a multiple of the standard deviation in 5 time periods after a specified date (0-6, 6-12, 12-18, 18-24 and 24-60 months after the specified eruption date). Then combine results into a single xarray. 
        
        Args:
        anom_dataset (xarray): data set of anomalies
        mean (xarray): data set of the mean anomaly (close to 0)
        std (xarray): data set of one standard deviation 
        e_date (date_str): eruption date
    """
    import xarray as xr, pandas as pd
    
    # set the initial start_date as the eruption date
    start_date = e_date
    
    # define the time periods to iterate over
    ds_time = []
    months = [6, 12, 18, 24, 60]
    names = ['00-06 mon', '06-12 mon', '12-18 mon', '18-24 mon', '24-60 mon']
    
    # iterate over all the time periods indicated in the months array
    for i, vals in enumerate(months):
        
        # set the end date for each iteration as the eruption date plus the no. of months specified in "months" array
        end_date = e_date + (vals+1) * pd.offsets.MonthBegin()
        
        # use the std_count function to count the occurence of 1, 2, 3 std devs for each time period
        count = std_count(anom_dataset, mean, std, start_date, end_date)
        
        # apprend the 
        ds_time.append(count)
        
        # set the start date for the next iteration as the end-date of the current iteration
        # (so I can go e.g. 0-6 months then 6-12 months)
        start_date = end_date
    
    # concatenate the results into a single array 
    std_times = xr.concat(ds_time, dim='count_months', coords = 'minimal')
    std_times.coords['count_months'] = names
    
    return std_times
    
    
    
# creates pandas dataframe table for event stats (min, min date, std counts, std) for 6 month time periods after Krakatoa
def stats_df(std_count_ds, model_min, min_date, std):
    """ Create a pandas dataframe table for event statistics for 6 month time periods after Krakatoa.  Rows for the models and columns for std counts, min, min date, std. (Ouput is suitable to make into a table.) 
        
        Args:
        std2_count_ds (xarray): data set of std counts (as output by the function "std_check")
        model_min (xarray): data set of minimum anomaly of each model
        min_date (xarray): data set of dates corresponding to the minimum anomaly of each model
        std (xarray): data set of the standard deviation for each model
    """
    
    import pandas as pd, numpy as np
    
    # convert to pd dataframe
    event_stats_df = std_count_ds.to_dataframe()
    # round to 2 decimal places
    event_stats_df = event_stats_df.round(decimals=2)
    
    # pivot table so models are in the rows
    event_stats = event_stats_df.pivot_table(index='model', columns=['count_months', 'std_count'])
    # add in other data as columns
    event_stats['Min'] = model_min.round(decimals=2)
    event_stats['Min Date'] = min_date.dt.strftime("%b %Y")
    event_stats['Std'] = std.round(decimals=2)
    
    # put mmm in the last row
    # gets a list of all the row names
    rows = list(event_stats.index)
    # removes the multi_model mean row
    rows.remove('Multi-model mean')
    # then add the row back at the end 
    event_stats = event_stats.loc[[*rows, 'Multi-model mean']]
    
    return event_stats
    
    
