# define a simple graph function 
def custom_plot(dataset, ax=None, **plt_kwargs):
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
        
    dataset.plot(ax=ax, **plt_kwargs) ## example plot here
    return(ax)

    
# define a function for subplots in the timeseries
def timeseries_graph(mmm_dataset, p10 = None, p90 = None, ax = None, **kwargs):
    """Create subplots of a time series, use shading to indicate 10th and 90th percentiles.  
    Add lines to show dates of five major eruptions between 1850-2014.  
    Return the axis.  
    
    Args:
        mmm_dataset (array): array of values (multi-model mean of climate variable) to be plotted in time series 
        p10 (array): array of values of 10th percentile
        p90 (array): array of values of 90th percentile
        ax (axis): axis
        **kwargs
    """
    import matplotlib.pyplot as plt, numpy as np
    
    # checking if an axis has been defined and if not creates one with function "get current axes"
    if ax is None:
        ax = plt.gca()
        
    # SUBPLOT
    # plot the percentiles (.data isn't necessary but maybe helps speed it up??)
    if p10 != None:
        ax.fill_between(p10.time.data, p10.data, p90.data, **kwargs)#, color='lightcoral')
    # plot the multi_model mean
    mmm_dataset.plot(color = 'k', ax=ax)#, **plt_kwargs)

    ax.grid(which='major', linestyle='-', linewidth='0.5', color='k') # customise major grid
    ax.minorticks_on() # need this line in order to get the minor grid lines 
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='k')
    
    # specify an array of eruption dates so I can mark the dates where erutpions occur on the plot
    e_dates = [np.array('1883-08-31T00:00:00.000000000', dtype='datetime64[ns]'),
     np.array('1902-10-31T00:00:00.000000000', dtype='datetime64[ns]'),
     np.array('1963-03-31T00:00:00.000000000', dtype='datetime64[ns]'),
     np.array('1982-04-30T00:00:00.000000000', dtype='datetime64[ns]'),
     np.array('1991-06-30T00:00:00.000000000', dtype='datetime64[ns]')]
    
    # Plot a dashed line to show the eruption time for the 5 major eruptions
    for date in e_dates:
        if date in mmm_dataset.time.data:
            ax.axvline(x=date, color = 'r', linestyle = '--', alpha = 0.9, linewidth='1.5')
    
    #label axes
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    
    return(ax) 


# define function to plot figures for composite graphs 
def SEA_plots(mmm_dataset = None, comp_dataset, p10 = None, p90 = None, color_cycle, ax = None, **plt_kwargs):
    """Create subplots for a superposed epoch analysis (SEA) graph.  SEA graph is composed of time series of each eruption contained in the mmm_dataset and the composite (of all eruptions in the mmm_dataset).  Shading is used to show the 10th and 90th percentiles of the composite.   
    Return the axis.  
    
    Args:
        mmm_dataset (xarray): xarray of eruptions and the values (multi-model mean of climate variable) for each to be plotted
        comp_dataset (xarray): xarray of composite values (multi-eruption multi-model mean of climate variable) to be plotted
        p10 (array): array of values of 10th percentile
        p90 (array): array of values of 90th percentile
        color_cycle (dict): dictionary of colours (as strings) 
        ax (axis): axis
        **kwargs
    """
    import xarray as xr, matplotlib.pyplot as plt, numpy as np, seaborn as sns
    
    # checking if an axis has been defined and if not creates one with function "get current axes"
    if ax is None:
        ax = plt.gca()  
    
    # SUBPLOT 1
    i=0
    # loop over all eruptions and plot the seasonal anomalies on one graph
    for v in mmm_dataset.volcano:
        mmm_dataset.sel(volcano=v).plot(ax=ax, label = v.data, color = color_cycle[i]) # plot the anomalies 
        i = i+1

    ax.fill_between(p10.time.data, p10.data, p90.data, color='lightgrey')

    comp_dataset.plot(color = 'k', ax=ax, label = 'Composite') 

    ax.set_facecolor('white')
    ax.legend(loc="upper left")
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black') # customise major grid
    ax.minorticks_on() # need this line in order to get the minor grid lines 
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # set's axis ticks to every 12 months 
    ax.set_xticks(np.arange(min(mmm_dataset.time), max(mmm_dataset.time)+1, 12))
    
    # set lables
    ax.set_xlabel(None) 
    ax.set_ylabel(None)
    
    return(ax) 

# define function to plot table of models with positive and negative anomalies colour coded input ds.var
def mod_check_plot(dataset, ax=None):
    """Create a table showing if anomalies are postive or negative for each model, each eruption at 3 averaged time periods.  Models as rows and eruptions as columns.    
    Return the axis.  
    
    Args:
        dataset (xarray): xarray of table values (use output from function "anomaly_check")
        ax (axis): axis
    """
    import xarray as xr, numpy as np, matplotlib.pyplot as plt
    
    # checking if an axis has been defined and if not creates one with function "get current axes"
    if ax is None:
        ax = plt.gca() 
    
    # change the axes so that rows correspond to each model and cols correspond to volcanoes
    ds = dataset.values
    ds = np.swapaxes(ds,0,1)
    
    # don't show axis
    ax.axis('off')
    
    # make an array 'colour_ar' to store the colour, if anom is positive colour =red an if neg colour=blue
    colour_ar = []
    for i in ds: 
        colour_row = []
        for j in i:
            c = 'lightcoral' if j == 1 else 'lightblue' 
            colour_row.append(c)
        colour_ar.append(colour_row)
        
    # change to postivie and negative sign correspponding to sign of anomaly
    ds = xr.where(ds > 0, '+', '-')
    
    # set the data for rows and columns 
    rows = dataset.model.data
    cols = dataset.volcano.data
        
    # make table
    tab = ax.table(cellText = ds, cellColours= colour_ar, colLabels=cols, rowLabels=rows, loc='upper center',cellLoc='center')

    tab.set_fontsize(14)

    # ways to change the colour of a specific cell
    # tab[0,0].set_facecolor('red')
    # table_props = table.properties()
    # table_cells = table_props['children'] # access data in cells, 'children' is a code word
    
    return(ax)



# define function to plot table of models with positive and negative anomalies colour coded input ds.var
def mod_check_plot2(dataset1, dataset2, dataset3, colour_ar, ax=None):
    """Create a table showing if anomalies are postive or negative for each model, each eruption at 3 averaged time periods (data for each time is in each of the three data sets).  Models as rows and eruptions as columns.    
    Return the axis.  
    
    Args:
        dataset1 (xarray): xarray of table values for first time period (use output from function "anomaly_check")
        dataset2 (xarray): xarray of table values for first time period (use output from function "anomaly_check")
        dataset3 (xarray): xarray of table values for first time period (use output from function "anomaly_check")
        colour_ar (dict): dictionary of colour strings 
        ax (axis): axis
    """
    import xarray as xr, numpy as np, matplotlib.pyplot as plt
    
    # checking if an axis has been defined and if not creates one with function "get current axes"
    if ax is None:
        ax = plt.gca() 
    ax.axis('off')
    
    # sum the datsets so I can use the sum to set the cell colour
    dataset = dataset1 + dataset2 + dataset3
    
    # set the data for rows and columns 
    rows = dataset.model.data
    cols = dataset.volcano.data
    
    # convert to data array (rather than xarray)
    ds = dataset.values
    ds1 = dataset1.values
    ds2 = dataset2.values
    ds3 = dataset3.values
    
    # change the axes so that rows correspond to each model and cols correspond to volcanoes
    ds = np.swapaxes(ds,0,1)
    ds1 = np.swapaxes(ds1,0,1)
    ds2 = np.swapaxes(ds2,0,1)
    ds3 = np.swapaxes(ds3,0,1)
    
    # change to postivie and negative sign correspponding to sign of anomaly
    ds1 = xr.where(ds1 == 1, '+ ', '-   ')
    ds2 = xr.where(ds2 == 1, '+ ', '-   ')
    ds3 = xr.where(ds3 == 1, '+ ', '-   ')

    # set the cell text to +/- at each time point
    cell_text = []
    for i, val in enumerate(ds1):
        row_text = []
        for j, val2 in enumerate(val):
            text = ds1[i,j] + ds2[i,j] + ds3[i,j]
            row_text.append(text)
        cell_text.append(row_text)
        
    # make an array of colour names 
    #colour_ar = np.array(['royalblue', 'lightblue', 'lightcoral', 'crimson'])
    
    # Use the data values as indices into the colour name array
    colours = colour_ar[ds]
    
    # make table
    tab = ax.table(cellText = cell_text, cellColours= colours, colLabels=cols, rowLabels=rows, loc='upper center',cellLoc='center')

    tab.set_fontsize(14)

    # ways to change the colour of a specific cell
    # tab[0,0].set_facecolor('red')
    # table_props = table.properties()
    # table_cells = table_props['children'] # access data in cells, 'children' is a code word
    
    return(ax)


# find where the anomalies are outside a threshold of +/- 2 standard deviations 
def stat_sig(dataset):
    """Find where the anomalies are outside a threshold of +/- 2 standard deviations.  Standard deviation calculated based on an 1850-1880 climatology.  
    
    Args:
        dataset (xarray): xarray of climate variable(s)
    """    
    import xarray as xr
    
    # calculate the standard deviation 
    std = dataset.sel(time = slice('1850-01', '1881-01')).std(dim = ['time'])
    
    # mark points oustide the 2 standard deviation threshold with a 100 (and non significant points with a zero)
    sig = xr.where((dataset < - 2*std) | (dataset > 2*std), 100, 0)
    
    return sig 


#define a function for spatial plots, plotting a dataset at 4 different time intervals
def spatial_plot(rows, cols, dataset, cmax, times, titles, colours, units, std):
    """Create a figure of spatial graphs with subplots for each time snapshot as specified in the dataset and times array. 
    Can use stippling to show areas where anomaly exceeds 2 standard deviations. 
    
    Args:
        rows (int): number of rows of subplots
        cols (int): number of columns of subplots
        dataset (xarray): data array of climate variable to be plotted
        cmax (float): maximum magnitude for colourbar
        times (date_str): dictionary of dates (date_str) for each time to be plotted
        titles (date_str): dictionary of titles (str) for each subplot
        colours (dict): colour palette for cmap
        units (str): units for axes label
        std (int): if std==1: use stippling
    """
    import matplotlib.pyplot as plt, cartopy.crs as ccrs, numpy as np
    
    fig = plt.figure()
    axs = []

#     rows = 2
#     cols = 2
    # vmin = np.min(dataset)
    # vmax = np.max(dataset)
    
    # calculate the standard deviation (if desired, and input is 1)
    if std == 1:
        sig_dataset = stat_sig(dataset)
    
    # set discrete colourbar with 15 intervals
    cmap = plt.get_cmap(f'{colours}')#, 15)
    
    for i, d in enumerate(times):    
        # Add a subplot with a projection    
        ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree(180))        
        # Select the date and corresponding data and plot it    
        # We'll add a separate colour bar, but make sure all plots share the same min and max colour value    
        data = dataset.sel(time = times[i])   
        C = data.plot(ax=ax, add_colorbar=False, transform=ccrs.PlateCarree(), cmap = cmap, vmin=-cmax, vmax=cmax)
        # hatching where anomalies exceed a threshold of 2 standard deviations
        if std == 1:
            data2 = sig_dataset.sel(time = times[i]).mean(dim='time')
            data2.plot.contourf(levels=[99, 1e10], hatches=[None,'..'], colors='none', add_colorbar=False, transform=ccrs.PlateCarree())
        
        # axes
        ax.coastlines()
        # set the axis limits to be slihtly larger (5 degrees) than the upper and lower bounds of the dataset 
        if (len(data.lon) < int(175/1.5)) & (len(data.lat) < int(175/1.5)):
            ax.set_extent([data.lon[0] - 2.5, data.lon[-1] + 2.5, data.lat[0] - 2.5, data.lat[-1] + 2.5], crs=ccrs.PlateCarree())
        # add on grid lines for longitude and latitude at specified range and spacing
        #ax.gridlines(xlocs=range(-180,181,20), ylocs=range(-80,81,20),draw_labels=False) 
        ax.gridlines(xlocs=range(-160,181,20), ylocs=range(-80,81,20),draw_labels=True)
        # add in different grid lines for tropics
        tropics = ax.gridlines(ylocs=[-66.5,-23.43691,23.43691,66.5],draw_labels=False,linewidth=2,linestyle='--', edgecolor='dimgrey')
        tropics.xlines=False
        # add titles for each subplot
        ax.set_title(titles[i])
        # Gather all axes for when we make the colour bar    
        axs.append(ax)    

    #Put the colour bar to the left of all axes
    cbar = plt.colorbar(C, orientation='horizontal', ax=axs, shrink=0.5, pad=0.05)
    cbar.ax.set_xlabel(f'{units}', fontsize=14)
    
    return fig


# define a function for subplots of the nino3.4 index over time 
def nino34_plot(ds, e_date, thold, ax = None, **kwargs):
    """Create subplot of timeseries of SST anomalies for NINO34 index.  
    Values that exceed the specified threshold are shaded. 
    Shows the timing of the Krakatoa 1883 eruption (dotted vertical line).  
    Return the axis.  
    
    Args:
        ds (xarray): dataset of SST anomalies (use output from function "nino34")
        e_date (dict): dict of eruption dates
        thold (float): threshold value of NINO34 index
        ax (axis): axis
        **kwargs
    """
    import matplotlib.pyplot as plt, numpy as np, pandas as pd
    
    # checking if an axis has been defined and if not creates one with function "get current axes"
    if ax is None:
        ax = plt.gca()
        
    # SUBPLOT
    #plot data and fill if it's over the thresholds
    ds.plot(color='k', lw=1)
    ax.fill_between(ds.time.values, ds.values, thold, where=ds.values>thold, interpolate =True, color='crimson', alpha=0.6)
    ax.fill_between(ds.time.values, ds.values, -thold, where=ds.values<-thold, interpolate =True, color='royalblue', alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(thold, color='k', lw=0.8, linestyle = ':')
    ax.axhline(-thold, color='k', lw=0.8, linestyle = ':')

    # plot gridlines
    ax.grid(which='major', ls=':', lw='0.5', color='grey') # customise major grid
    ax.minorticks_on() # need this line in order to get the minor grid lines 
    ax.grid(which='minor', ls=':', lw='0.5', color='grey')
    ax.set_axisbelow(True) # sets the gridlines behind the data

    #set the frequency of the xticks 
    years = pd.date_range(ds.time.data[0], ds.time.data[-1], freq='YS')
    ax.set_xticks(years.values)
    ax.set_xticklabels(years.year) # .year shows only the year (not month)

    # remove xlabels and set ylabel
    ax.set_xlabel(None)
    ax.set_title(None)
    ax.set_ylabel(f'Sea surface temperature anomaly [$\degree$C]', size=12)

    # add dotted lineshowing the year of the krakatoa eruption
    ax.axvline(x=e_date[0], color = 'red', linestyle = '--', alpha = 0.9, linewidth='0.8')
    
    return ax


# make a table for nino3.4 index, showing models in rows and months in columns (blue indicates el nino, red la nina)
def nino34_table(dataset, y0_header, y1_header, y0, y1, ax=None):
    """Create a colour-coded table showing ENSO events over time for each model. 
    Blue indicates el nino, red la nina, white neutral.  
    Models as rows and months as columns.    
    Return the axis.  
    
    Args:
        dataset (dataframe): pandas dataframe table of values (output of function "ENSO_models")
        y0_header (float): value determines the extent of the header
        y1_header (float): value determines the extent of the header
        y0 (float): value determines the extent of the table
        y1 (float): value determines the extent of the table
        colour_ar (dict): dictionary of colour strings 
        ax (axis): axis
    """
    import matplotlib.pyplot as plt, numpy as np, pandas as pd
    
    if ax is None:
        ax = plt.gca() 
    ax.axis('off')
    
    # convert nan values to 0 so they're a number that I can use in my colours array
    dataset = dataset.replace(np.nan, 0)
    
    # convert data to the right format so it can be used in my table 
    ds = dataset.to_xarray()
    ds2 = ds.to_array()
    ds2 = ds2.values
    # swap the axes so the models are in the columns 
    ds2 = np.swapaxes(ds2,0,1)
    
    # convert data to integers so I can use it to index my colour array
    ds2 = ds2.astype(int)
    
    # define an array of colours to be used (red el nion, blue la nina, white neutral)
    colour_ar = np.array(['white', 'crimson', 'royalblue'])

    # Use the data values as indices into the colour name array
    colours = colour_ar[ds2]
    
    # set column labels to each month
    cols = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    cols = cols*6
    
    # set row labels as the model names
    rows = ds.model.data
    
    # define array for column headers
    years = ['1882', '1883', '1884', '1885', '1886', '1887']

    # bbox has x0, x1, y0, y1 so determines the size (and location??) of the table
    # make header first (bbox=[x0=0, y0=0.67, x1=1, y1=0.06)
    header_0 = ax.table(cellText=[['']*len(years)], colLabels=years, loc='bottom', bbox=[0, y0_header, 1, y1_header])
    # then make table below (bbox=[x0=0, y0=-0.3, x1=1, y1=1)
    the_table = ax.table(cellColours= colours, rowLabels=rows, colLabels=cols, loc='bottom',cellLoc='center', bbox=[0, y0, 1, y1])
    
    return ax 

    
    
# function that converts pandas dataframe of event statistics (std_count, min etc) to a table that can be saved as a figure   
# use the xarray dataset to get month names and columns 
def stats_table(stats_df, dataset, ax):
    """Converts pandas dataframe of event statistics (std_count, min etc) to a table that can be saved as a figure.
    
    Args:
        stats_df (dataframe): pandas dataframe table of values (output of function "stats_df")
        dataset (xarray): use dataset to extract header labels 
        ax (axis): axis
    """
    import xarray as xr, numpy as np, pandas as pd, matplotlib.pyplot as plt
    
    # gets a list of all the row names
    rows = list(stats_df.index)
    
    # convert data to a list so I can use it to generate cell text 
    stats_list = stats_df.values.tolist()

    # set the data for subcolumns and columns 
    subcol = ['1*std', '2*std', '3*std']
    cols = subcol*len(dataset.count_months.data) + ['Min'] + ['Min Date'] + ['Std']
    header_times = ['0-6 mon', '6-12 mon', '12-18 mon', '18-24 mon', '24-60 mon'] # can change to dataset.count_months.data if desired
    
    # remove axis
    ax.axis('off')
    # Add headers and a table at the bottom of the axes
    header_0 = ax.table(cellText=[['']*len(header_times)], colLabels=header_times, loc='bottom', bbox=[0, 0.87, 0.834, 0.06])
    the_table = ax.table(cellText=stats_list, rowLabels=rows, colLabels=cols, loc='bottom',cellLoc='center', bbox=[0, -0.3, 1.0, 1.2])
    return ax
    
    
    
    
    
    




