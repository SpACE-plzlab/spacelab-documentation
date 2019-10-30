from google.cloud import storage
import glob
from osgeo import gdal
import os
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from rasterio.merge import merge
import multiprocessing as mp
import rasterstats as rs
import skbio

# mount google cloud storage bucket
os.system('gcsfuse scaling_analysis /home/annie/scaling-analysis/')
    
# convert all to functions!

### PREPARE POINTS ###
# get coordinates file
coords = pd.read_csv('/home/annie/scaling-analysis/coordinates_fia_bbs.csv')
coords.head()

# creating a geometry column 
geometry = [Point(xy) for xy in zip(coords['lon'], coords['lat'])]

# Coordinate reference system : WGS84
crs = {'init': 'epsg:4326'}

# Creating a Geographic data frame 
gdf = gpd.GeoDataFrame(coords, crs=crs, geometry=geometry)
gdf.head()

# convert coordinates to equal area (proj 5070)
coords_proj = gdf.copy()

# Reproject the geometries by replacing the values with projected ones
coords_proj['geometry'] = coords_proj['geometry'].to_crs(epsg=5070)
coords_proj['index'] = range(0, coords_proj.shape[0])
coords_proj.head()

# new df with coords
coords_data = {'lon': coords_proj['geometry'].x, 'lat': coords_proj['geometry'].y, 'radius': coords_proj['radius']}
coords_clean = pd.DataFrame(data = coords_data)

### PREPARE RASTERS ###
# need a dictionary with data name, list of files
lstnames = glob.glob('/home/annie/scaling-analysis/lst*')
prcpnames = glob.glob('/home/annie/scaling-analysis/chirp*')
elevnames = glob.glob('/home/annie/scaling-analysis/elev_mosaic*')
aspectnames = glob.glob('/home/annie/scaling-analysis/aspect_mosaic*')
slopenames = glob.glob('/home/annie/scaling-analysis/slope_mosaic*')
nightlightnames = glob.glob('/home/annie/scaling-analysis/night*')
nlcd2001names = glob.glob('/home/annie/scaling-analysis/nlcd2001_mosaic*')
nlcd2016names = glob.glob('/home/annie/scaling-analysis/nlcd2016_mosaic*')
soilclassnames = glob.glob('/home/annie/scaling-analysis/TAXOUSDA_250m_aea.tif')
soildepthnames = glob.glob('/home/annie/scaling-analysis/BDTICM_M_250m_aea.tif')

# MAKE SURE TO ADD IN ASPECT/SLOPE LATER
datalist = {'lst': lstnames, 'prcp': prcpnames, 'elev': elevnames, 'aspect': aspectnames,
'slope': slopenames, 'nightlight': nightlightnames, 'soildepth': soildepthnames, 'nlcd2001': nlcd2001names, 
'nlcd2016': nlcd2016names, 'soilclass': soilclassnames}

# list of month names
monthlist = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# create data frame to iterate over
df_parallel = pd.DataFrame(columns = ['var', 'month', 'radius'])
df_parallel['var'] = ['lst'] * 36 + ['prcp'] * 36 + ['elev'] * 3 + ['aspect'] * 3 + ['slope'] * 3 + ['nightlight'] * 3 + ['nlcd2001'] * 3 + ['nlcd2016'] * 3 + ['soilclass'] * 3 + ['soildepth'] * 3
df_parallel['month'] = list(range(1, 13)) * 3 + list(range(1, 13)) * 3 + ['NaN'] * 24
df_parallel['radius'] = ([1000] * 12 + [10000] * 12 + [50000] * 12) * 2 + [1000, 10000, 50000] * 8

out_data = list()

def shannondiv(x):
	m = np.array(x < 0)
	x = np.ma.MaskedArray(x, m)
	m = np.ma.getmask(x)
	x = np.array(x)
	x[x<0] = 9999 # these are masked out but need to be diff
	x = np.ma.MaskedArray(x, m)
	nvals = x.count()
	if (nvals <= 0):
		return 'NaN'
	else:
		x = x.flatten()
		return skbio.diversity.alpha.shannon(x)

def shannoneven(x):
	m = np.array(x < 0)
	x = np.ma.MaskedArray(x, m)
	m = np.ma.getmask(x)
	x = np.array(x)
	x[x<0] = 9999
	x = np.ma.MaskedArray(x, m)
	nvals = x.count()
	if (nvals <= 0 or x.shape[0] <= 1):
		return 'NaN'
	else:
		x = x.flatten()
		return skbio.diversity.alpha.pielou_e(x)

def calc_stats(i, info_df, pt_data, datalist):
	r = info_df.loc[i]['radius']
	key = info_df.loc[i]['var']
	month = info_df.loc[i]['month']
	print('beginning summary for radius %s, var %s, month %s' % (str(r), str(key), str(month)))
	# create buffered polygons
	pt_data_r = pt_data.loc[pt_data['radius'] == r].buffer(r)
	# Create a buffered polygon layer from your plot location points
	pt_data_poly_r = pt_data.loc[pt_data['radius'] == r].copy()
	# replace the point geometry with the new buffered geometry
	pt_data_poly_r["geometry"] = pt_data_r
	f = datalist[key][0]
	if (month == 'NaN'):
		with rasterio.open(f) as src:
			temp_data = src.read(1, masked=True)
			temp_meta = src.profile
	else:
		with rasterio.open(f) as src:
			temp_data = src.read(month, masked=True)
			temp_meta = src.profile
	src.close()
	if (month == 'NaN'):
		point_stats_temp = rs.zonal_stats(pt_data_poly_r,
			temp_data,
			affine=temp_meta['transform'],
			copy_properties=True,
			geojson_out=True,
			stats="mean std")
	else:
		point_stats_temp = rs.zonal_stats(pt_data_poly_r,
			temp_data,
			affine=temp_meta['transform'],
			copy_properties=True,
			geojson_out=True,
			stats="mean std")
	# Turn extracted data into a pandas geodataframe
	point_stats_df = gpd.GeoDataFrame.from_features(point_stats_temp)
	point_stats_df.head()
	print('done calculating zonal stats for radius %s, var %s, month %s' % (str(r), str(key), str(month)))
	# change column names
	if (month == 'NaN'):
		colname_sd = key + '_sd'
		colname_mean = key + '_mean'
	else:
		colname_sd = key + '_sd_' + monthlist[month - 1]
		colname_mean = key + '_mean_' + monthlist[month - 1]
	point_stats_df = point_stats_df.rename(index=str, columns={"mean": colname_mean, "std": colname_sd})
	return(point_stats_df)
	print('completed summary for radius %s, var %s, month %s' % (str(r), str(key), str(month)))
	
def calc_stats_class(i, info_df, pt_data, datalist):
	r = info_df.loc[i]['radius']
	key = info_df.loc[i]['var']
	month = info_df.loc[i]['month']
	print('beginning summary for radius %s, var %s, month %s' % (str(r), str(key), str(month)))
	# create buffered polygons
	pt_data_r = pt_data.loc[pt_data['radius'] == r].buffer(r)
	# Create a buffered polygon layer from your plot location points
	pt_data_poly_r = pt_data.loc[pt_data['radius'] == r].copy()
	# replace the point geometry with the new buffered geometry
	pt_data_poly_r["geometry"] = pt_data_r
	f = datalist[key][0]
	if (month == 'NaN'):
		with rasterio.open(f) as src:
			temp_data = src.read(1, masked=True)
			temp_meta = src.profile
	else:
		with rasterio.open(f) as src:
			temp_data = src.read(month, masked=True)
			temp_meta = src.profile
	src.close()
	if (month == 'NaN'):
		point_stats_temp = rs.zonal_stats(pt_data_poly_r,
			temp_data,
			affine=temp_meta['transform'],
			copy_properties=True,
			geojson_out=True,
			stats = 'unique',
			add_stats = {'shannondiv': shannondiv, 'shannoneven': shannoneven})
	else:
		point_stats_temp = rs.zonal_stats(pt_data_poly_r,
			temp_data,
			affine=temp_meta['transform'],
			copy_properties=True,
			geojson_out=True,
			stats = 'unique',
			add_stats = {'shannondiv': shannondiv, 'shannoneven': shannoneven})
	# Turn extracted data into a pandas geodataframe
	point_stats_df = gpd.GeoDataFrame.from_features(point_stats_temp)
	point_stats_df.head()
	print('done calculating zonal stats for radius %s, var %s, month %s' % (str(r), str(key), str(month)))
	# change column names
	if (month == 'NaN'):
		colname_shannoneven = key + '_shannoneven'
		colname_shannondiv = key + '_shannondiv'
	else:
		colname_shannoneven = key + '_shannoneven_' + monthlist[month - 1]
		colname_shannondiv = key + '_shannondiv_' + monthlist[month - 1]
	point_stats_df = point_stats_df.rename(index=str, columns={"shannondiv": colname_shannondiv, "shannoneven": colname_shannoneven})
	return(point_stats_df)
	print('completed summary for radius %s, var %s, month %s' % (str(r), str(key), str(month)))
	
# run in parallel over all datasets
# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(processes = 10)

#df_parallel.shape[0]
#idxlist = list(range(0, 72))
#idxlist = idxlist + [72, 75, 78, 81, 82, 83] 
#idxlist = [73, 74, 76, 77, 79, 80]
#idxlist = [93,94,95] # soil depth
idxlist = list(range(84, 93)) # class variables
# Step 2: `pool.apply` the `calc_stats()`
#idxlist = list(range(0, df_parallel.shape[0])) # RUN THIS BUT WITHOUT THE 50000M OPTIONS FOR 30M LAYERS
results = [pool.apply_async(calc_stats_class, args=(row, df_parallel, coords_proj, datalist)) for row in idxlist]
output = [p.get() for p in results]

# Step 3: Don't forget to close
pool.close()    

# make radii dataframes
final_data1000 = coords_proj.loc[coords_proj['radius']==1000]
final_data10000 = coords_proj.loc[coords_proj['radius']==10000]
final_data50000 = coords_proj.loc[coords_proj['radius']==50000]

# join together output dfs
for d in list(range(0, len(output))):
	if (output[d].iloc[0]['radius'] == 1000):
		final_data1000 = pd.merge(final_data1000, output[d].iloc[:, [2,6,7]], on = 'index', how = 'left')
	elif (output[d].iloc[0]['radius'] == 10000):
		final_data10000 = pd.merge(final_data10000, output[d].iloc[:, [2,6,7]], on = 'index', how = 'left')
	else:
		final_data50000 = pd.merge(final_data50000, output[d].iloc[:, [2,6,7]], on = 'index', how = 'left')
		
# add together
final_data = final_data1000.append(final_data10000).append(final_data50000)
#final_data = final_data10000.append(final_data50000)
final_data.head()

# write out
final_data.to_csv(path_or_buf='/home/annie/temp/alldata_out_catvars.csv', sep = ',')
