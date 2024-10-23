#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from urllib.parse import urlencode
from urllib.request import urlretrieve
import pandas as pd
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.colors as mpltcols
import plotly.express as px
import numpy as np


"""
I developed the ECCC station data download functions in previous work at the
Pacific Climate Impacts Consortium
"""
def download_station_csv(ecid,year,storagePath,month='1',timeframe=2):
    #note that a day has to be specified else the month increments back by one and sometimes the year too....
    day='1'
    if timeframe == 1:
        storename = storagePath+'EC_autodownload/Hourly_CSVs/' + str(year) + '/' + ecid + "_" + str(year) + "_" + str(month) + ".csv"
    elif timeframe == 2:
        storename = storagePath+'EC_autodownload/Daily_CSVs/' + str(year) + '/' + ecid + "_" + str(year) + ".csv"
        day = '1'
        month = '12'
    elif timeframe == 3:
        storename = storagePath+'EC_autodownload/Monthly_CSVs/' + ecid + ".csv"
    elif timeframe == 4:
        storename = storagePath+'EC_autodownload/Almanac_CSVs/' + ecid + ".csv"
    #  https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=53479&Year=2024&Month=10&Day=21&time=LST&timeframe=1&submit=Download+Data
    print(storename)
    params = urlencode(
        {
            'format': 'csv',
            'stationID': str(ecid),
            'Year': str(int(year)),
            'Month': month,
            'Day': day,
            'time': 'LST',
            'timeframe': str(timeframe),
            'submit': 'Download+Data',
        }
    )
    result=urlretrieve("http://climate.weather.gc.ca/climate_data/bulk_data_e.html?%s" % params,storename)

def get_all_annual_daily_data(ecidxs,year,storagePath):
    for ecid in ecidxs:
        download_station_csv(ecid,year,storagePath,month='1',timeframe=2)

def get_all_month_hourly_data(ecidxs,year,month,storagePath):
    for ecid in ecidxs:
        download_station_csv(ecid,year,storagePath,month=str(month),timeframe=1)

def download_ec_wx_data(
    gdf_meta,
    yoi=2023.0,
    monthoi=10,
    nstations = 10,
    inventory_filename='../data/Station_Inventory_EN.csv',
    ):

    '''
    Function to download weather station data from EC's websites based on the provimity
    to the stations given in the variable gdf_meta, which are the sampling sites.

    Facility to vary the year, the number of stations, the month of iterest. Changing the downloaded
    sampling sites will lead to differend meta data variable and a different result in what's downloaded.
    
    Grab the data from Environment Canada for October, 2023.
    Seems that Vancouver Airport only reports daily precipitation totals
    Which means that custom calculation of 24 hour precip prior to sampling isn't possible with
    That weather station.

    The station inventory is available here:
    https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv
    '''
    #Get the station inventory
    result=urlretrieve(
        "https://collaboration.cmc.ec.gc.ca/cmc/climate/Get_More_Data_Plus_de_donnees/Station%20Inventory%20EN.csv",
        inventory_filename)
    
    #load in the station inventory from ECCC/MSC
    inventory = pd.read_csv(inventory_filename,skiprows=3)
    
    
    #We want to find stations with hourly data and with the year in the desired range.
    #First, get the stations operating in the correct year
    inventory = inventory.loc[(inventory["First Year"] < yoi) & (inventory["Last Year"] >= yoi)]
    
    #Find the station of interest
    #First, get the center of mass of the observing sites
    avg_lon = np.average(gdf_meta['LONGITUDE (Decimal Degrees)'])
    avg_lat = np.average(gdf_meta['LATITUTE (Decimal Degrees)'])
    
    #Calculate the distance
    #Here we are clumsily multiplying x*x instead of x^2 because of error can't track down.
    inventory['dist_to_sites'] = np.sqrt((inventory['Latitude (Decimal Degrees)'] - avg_lat)*(inventory['Latitude (Decimal Degrees)'] - avg_lat) + (inventory['Longitude (Decimal Degrees)'] - avg_lon)*(inventory['Longitude (Decimal Degrees)'] - avg_lon))
    inventory = inventory.sort_values(by=['dist_to_sites'])
    
    #Grab the daily data (that contains the 24 hour precipitation totals.
    #Let's pull the top five stations and find the ones that have data on the days that we want. 
    #Again, we don't have hourly precipitation from ECCC data AFAIK
    ecids = list(inventory['Station ID'][0:nstations])
    ecids = [str(x) for x in ecids]
    
    #Force Vancouver airport for the purposes of the exercise
    ecids.append('51442')
    
    #Download the data
    #get_all_annual_daily_data(ecidxs=ecids,year=int(yoi),storagePath='../data/')
    get_all_month_hourly_data(ecidxs=ecids,year=int(yoi),month=int(monthoi),storagePath='../data/')




# In[2]:


'''
Here's a simplistic way to slightly more flexibly pull water quality data. This relies on having
the most recent 4 years of data as made available here:
https://pub.data.gov.bc.ca/datasets/949f2233-9612-4b06-92a9-903e817da659/ems_sample_results_4yr_current_expanded.csv
. This is a large file at about 2.2 Gb. But, it only needs to be downloaded once.

This method relies on a bash command line in linux and uses the python native os package
to execute a simple filtering command. In the below function I filter by station and monitoring
species.
'''

def extract_and_concat_ems_data(
    stnids=['0300083', '0300085', 'E282871'],
    species='6PPD-Quinone',
    bulkfilename='../data/ems_sample_results_4yr_current_expanded.csv',
    outputfilename='../data/ems_subset.csv'):

    """
    Function to extract data for specific stations and specific variables from a bulk download
    4-year dataset from the EMS web portal.
    
    There is no error trapping to make sure the station or the species exists!
    """
    for stnid in stnids:
        outputfilename = '../data/ems_stnid_'+stnid+'_species_'+species+'.csv'
        commandstring = "grep '"+stnid+"' "+bulkfilename+" | grep '"+species+"' > "+outputfilename
        print(commandstring)
        os.system(commandstring)

'''
Uncomment this line to extrace the station data required for this assignment.
'''
#extract_and_concat_ems_data()


# In[3]:


'''
Read in the water quality monitoring dataset as pre downloaded. I used the web tool to subset
by time and by station identifier and then use code to extract out the species of interest.
Ideally, this would have been done with an API.
'''

df = pd.read_csv('../data/EMSWR_Rus_050802.csv')
df = df[df['PARAMETER NAME'] == '6PPD-Quinone']
df['LONGITUDE (Decimal Degrees)'] = -1 * df['LONGITUDE (Decimal Degrees)']

#convert mg/L to ng/L and make the heading nicer for plotting
df['[6PPD – Quinone] (ng/L)'] = df['Results']*1000000


# In[4]:


#Lets make a map of the station data
#Convert the pandas data frame into a geopandas and then plot onto a figure
#There's a typo in the data filter output from dataBC. LATITUTE instead of LATITUDE...
gdf = gpd.GeoDataFrame(
    df,geometry=gpd.points_from_xy(df['LONGITUDE (Decimal Degrees)'],df['LATITUTE (Decimal Degrees)']), crs="EPSG:4326"
)
#Extract the unique water sampling stations and lat/lon/geometry for them into a meta data variable.
gdf_meta = gdf[
    [
        'EMS ID', 
        'LOCATION NAME', 
        'LONGITUDE (Decimal Degrees)',
        'LATITUTE (Decimal Degrees)',
        'geometry'
    ]
]
#Get rid of duplicate rows, leaving one row per station.
gdf_meta = gdf_meta.drop_duplicates()
#Convert the timescale to datetime object
gdf['START DATE'] = pd.to_datetime(gdf['START DATE'])

#Round the timestamps to the nearest hour and give a nicer column name for plotting.
gdf['Sample Date'] = gdf['START DATE'].dt.round(freq='h')

'''
Make a plot of the data. Here I am using plotly because it makes nice looking plots
and lends itself to making dashboards throught the dash toolset.

Here, I am making a facet plot with three panels, one for each sample class. The class is given in the
title of each subpanel. The points are colourized by station and also match the colours used in the
location map.
'''
fig = px.scatter(
    gdf,
    x='Sample Date',
    y='[6PPD – Quinone] (ng/L)',
    color='LOCATION NAME',
    facet_col='CLASS DESCRIPTION',
    log_y=True,
    range_y=[0.5,150],
    title="Temporal evolution of 6PPD-Quinone at 3 North Vancouver Sites"
)
#Clean up the sub-panel titles so they only give the sample class
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(marker=dict(size=12))
fig.update_layout(
    legend=dict(
        title=None,
        font=dict(size=9),
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="right",
        x=0.9,
    ),
    title=dict(automargin=True,
              font=dict(size=20))
)
#fig.update_layout(legend_t
fig.write_image("../figures/fig_3b_quinone_evolution_timeseries.pdf")
fig.write_image("../figures/fig_3b_quinone_evolution_timeseries.png")


# In[5]:


'''
This next section of code creates the location map. Here I use geopandas and matplotlib to make the 
plot of the station locations. The package contextily enables the addition of basemaps and has
a layer for satellite imagery. 

Axes padding away from station locations in meters. This would have to change if stations
were to be plotted over a larger area.
'''

padding = 3000

'''
Load in the shape file for the streams and plot. I downloaded this shape file from the 
BC data catalogue. Unfortuntely, I don't know how to hit the API to make this download 
automatic over a region of interest.
'''
streams = gpd.read_file('../gis/FWA_STREAM_NETWORKS_SP/FWSTRMNTWR_line.shp')

#make a figure and axes
fig,ax_small = plt.subplots(figsize=(10,8))
plt.axis('Off')
ax_small = fig.add_subplot(1, 1, 1)

'''
This forces colours to stations to match what is plotted in the timeseries. This would have to be changed if
more flexible code was desired that could handle arbitrary station locations and IDs.
'''
colordict = {'CAPILANO R NEAR MARINE DRIVE': 'blue','LYNN CR NEAR HWY 1': 'red','MCKAY CREEK NEAR MOUTH': 'green'}
cmap = mpltcols.ListedColormap([colordict[b] for b in gdf_meta['LOCATION NAME']])

#Convert to BC Albers for mapping. EPSG 3005
ax_small=gdf_meta.to_crs('EPSG:3005').plot(ax=ax_small,cmap=cmap,zorder=10,markersize=200,alpha=0.7)

#Put labels on the sampling locations
gdf_meta['bc_albers_x'] = gdf_meta.to_crs('EPSG:3005').get_coordinates()['x']
gdf_meta['bc_albers_y'] = gdf_meta.to_crs('EPSG:3005').get_coordinates()['y']
for index,row in gdf_meta.to_crs('EPSG:3005').iterrows():
    name = row['LOCATION NAME']
    ax_small.annotate(
        name,
        (row['bc_albers_x']-1500,row['bc_albers_y']+500),
        fontsize=14,
        bbox=dict(facecolor='white', edgecolor='none', pad=3.0,alpha=0.6),
    )
    
#Pad out the axes based on the padding parameter in meters
xlims = ax_small.get_xlim() #[1205000,1220000]
ax_small.set_xlim([(xlims[0]-padding),(xlims[1]+padding)])
ylims = ax_small.get_ylim() #[480000,485000]
ax_small.set_ylim([(ylims[0]-padding),(ylims[1]+padding)])
#print(xlims,ylims)
ax_small=streams.plot(ax=ax_small,linewidth=3)

#use contextily to put a base map on the figure
cx.add_basemap(ax_small, source=cx.providers.Esri.WorldImagery, crs=streams.crs)
ax_small.set_axis_off()

#Give the mapa title
ax_small.set_title('North Vancouver 6PPD-Quinone Sampling Locations')
fig = plt.gcf()

#Save as png and PDF
fig.savefig('../figures/fig_3a_Quinone_sampling_sites_map.pdf')
fig.savefig('../figures/fig_3a_Quinone_sampling_sites_map.png')


# In[6]:


#Execute the download of the EC weather data. 
download_ec_wx_data(
    yoi=2023.0,
    monthoi=10,
    nstations = 10,
    gdf_meta=gdf_meta,
    inventory_filename='../data/Station_Inventory_EN.csv'
)




# In[7]:


'''
Through manual data exploration, I see that station VANCOUVER SEA ISLAND CCG with climate
ID 1108380 and station ID 51357 has hourly data. Also see Sea Island and Vancouver Harbour CS
Have hourly precipitation data.
'''
#station_id = 51357 #Sea Island
station_id = 6833 #West Vancouver Auto
#station_id = 888 #Vancouver Harbour CS

'''
Load the weather data for the station of interest. A cleaner way to do this would be to 
programmatically go through the hourly data from the top ten nearest stations and find 
the nearest one with hourly data
'''
wxdata = pd.read_csv('../data/EC_autodownload/Hourly_CSVs/2023/'+str(station_id)+'_2023_10.csv')
wxdata['Date/Time (LST)'] = pd.to_datetime(wxdata['Date/Time (LST)'])

'''
We need to pick out all of the sampling times and days and create 24 hour precipitation totals 
for each of those intervals. Doing this the dumb way with a loop through all of the observations.

This should be a function!
'''
precip_totals = []
for index, row in gdf.iterrows():
    #extract the precipitation data from the wxdata
    #get the row with the sample hour
    endidx = list(wxdata.index[wxdata['Date/Time (LST)'] == row['Sample Date']])[0] + 1
    
    #Pick the start index assuming the data have a complete timescale.
    startidx = endidx - 24
    
    #Subset
    wxdata_use = wxdata.iloc[startidx:endidx,]
    
    #Make the sum
    precip_totals.append(wxdata_use['Precip. Amount (mm)'].sum())

#assign the vector to the data frame
gdf['24hr_preceding_precip (mm)'] = precip_totals

#Make a plot of the data
fig = px.scatter(
    gdf,
    x='24hr_preceding_precip (mm)',
    y='[6PPD – Quinone] (ng/L)',
    title='Comparison between 6PPD-Quinone concentration and antecedent rainfall',
    color='LOCATION NAME',
    facet_col='CLASS DESCRIPTION',
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
fig.update_traces(marker=dict(size=12))
#Add colour zone for low risk morbidity
# fig.add_hrect(
#     y0=0, 
#     y1=41, 
#     annotation_text="low risk",
#     annotation_position="top left",
#     fillcolor="green",
#     opacity=0.2,
#     line_width=0,
# )
# #Add colour zone for risk of morbidity to juvenile salmon
fig.add_hrect(
    y0=41,
    y1=95,
    annotation_text="risk to juvenile salmon",
    annotation_position="bottom left",
    fillcolor="yellow",
    opacity=0.2,
    line_width=0
)
# #Add colour zone for risk of morbidity to adult salmon
fig.add_hrect(
    y0=95, 
    y1=105, 
    annotation_text="risk to juvenile and adult salmon",
    annotation_position="bottom left",
    fillcolor="red",
    opacity=0.2,
    line_width=0
)
fig.update_layout(
    legend=dict(
        title = None,
        font=dict(size=9),
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="right",
        x=0.9
    ),
    title=dict(
        automargin=True,
        font=dict(
            size=20
        )
    )
)

#write the figure to files
fig.write_image("../figures/fig_3c_and_3d_quinone_vs_precipitation_linscale.pdf")
fig.write_image("../figures/fig_3c_and_3d_quinone_vs_precipitation_linscale.png")


# In[ ]:




