# functions used across models, reanalysis and obs sections 

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
    

