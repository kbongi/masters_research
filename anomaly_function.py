# Functions

import xarray as xr

# # define function to call models
# def model_path(institution, model, version):
#     return f'/g/data/lp01/CMIP6/CMIP/{institution}/{model}/historical/r1i1p1f1/Amon/tas/gr1.5/{version}/tas_Amon_{model}_historical_r1i1p1f1_gr1.5_185001-201412.nc'

# define function to call models without filename extension
#def model_path(institution, model, version):
    #return f'/g/data/lp01/CMIP6/CMIP/{institution}/{model}/historical/r1i1p1f1/Amon/tas/gr1.5/{version}/*.nc'
    
    # function to read in all the model files, **note it assumes a time period from 1850-2014 with 1980 timesteps
def read_models(institution_dir, variable_dir, start_date, end_date):
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
            d = xr.open_mfdataset(path, combine='by_coords')
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
                print(name, path)
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
    return data.groupby('time.season').mean()

# function to calculate a seasonal anomaly over a time period entered by user
def seasonal_anomaly_xr(dataset, start_date, end_date):

    # first I need to define a new coordinate (seasonyear) so that december gets counted with the adjoining jan and feb
    seasonyear = (dataset.time.dt.year + (dataset.time.dt.month//12)) 
    dataset.coords['seasonyear'] = seasonyear

    # group data into seasons and calculate the seasonal mean for each year in the dataset 
    yearly_seasonal = dataset.groupby('seasonyear').apply(seasonal_mean)

    # calculate the mean climatology along each season for the time period 
    clim_seasonal = yearly_seasonal.sel(seasonyear = slice(f'{start_date}',f'{end_date}')).mean(dim = 'seasonyear')
    
    # calculate the anomaly and returns it as an xarray
    anom_seasonal = (yearly_seasonal - clim_seasonal)
    return anom_seasonal

# function to calculate a seasonal anomaly for a multidimensional xarray over a time period entered by user
def seasonal_anomaly(dataset, start_date, end_date):
    # define an array to store the anomalies for each model
    seasonal_anom=[]
    
    # first I need to define a new coordinate (seasonyear) so that december gets counted with the adjoining jan and feb
    seasonyear = (dataset.time.dt.year + (dataset.time.dt.month//12)) 
    dataset.coords['seasonyear'] = seasonyear
    
    # loop over each model 
    for m in dataset.model:
        
        # group data into seasons and calculate the seasonal mean for each year in the dataset 
        yearly_seasonal = dataset.sel(model=m).groupby('seasonyear').apply(seasonal_mean)

        # calculate the mean climatology along each season for the time period 
        clim_seasonal = yearly_seasonal.sel(seasonyear = slice(f'{start_date}',f'{end_date}')).mean(dim = 'seasonyear')

        # calculate the anomaly and returns it as an xarray
        anom_seasonal = (yearly_seasonal - clim_seasonal)
        
        # append each model into the array
        seasonal_anom.append(anom_seasonal)
    
    # Combine the individual data sets into a single xarray dataset.
    multi_seasonal_anom = xr.concat(seasonal_anom, dim='model')
    return multi_seasonal_anom

# define function to calculate monthly anomalies for a singular xarray
def monthly_anom_xr(variable, start_date, end_date):

    # group the data into months
    variable_monthly = variable.groupby('time.month')
    
    # calculate the mean climatology along each month for the time period 1850-1900 
    clim_monthly = variable.sel(time = slice(f'{start_date}', f'{end_date}')).groupby('time.month').mean(dim = 'time')
    
    # caclulate the anomalies for each month and return it as an array
    anom_monthly = (variable_monthly - clim_monthly)
    return anom_monthly


# define function to calculate monthly anomalies for a multidimensional array of models
def monthly_anomaly(dataset, start_date, end_date):
    # define an array to store the anomalies for each model
    monthly_anom=[]
    
    # loop over each model 
    for m in dataset.model:
        # group the data into months
        variable_monthly = dataset.sel(model=m).groupby('time.month')
    
        # calculate the mean climatology along each month for the time period 1850-1900 
        clim_monthly = dataset.sel(model=m).sel(time = slice(f'{start_date}', f'{end_date}')).groupby('time.month').mean(dim = 'time')

        # caclulate the anomalies for each month and return it as an array
        anom_monthly = (variable_monthly - clim_monthly)
        
        # append each model into the array
        monthly_anom.append(anom_monthly)
        
    # Combine the individual data sets into a single xarray dataset.
    multi_monthly_anom = xr.concat(monthly_anom, dim='model')
    return multi_monthly_anom

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
