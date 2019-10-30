**radii_extraction.py:** 
* Created: June/July 2019
* Author: AC Smith
* Overview: This is the initial radius extraction script for the nasabio scaling project. The idea here is to calculate geodiversity over circles of a given radius using rasters.
* Specifics:
  - Takes point locations with coordinates, and raster images.
  - Summarizes stats (e.g., std dev) over circles of given radius around each individual point.
  - Exports table of summary stats.
  - Initially run on Google Cloud (see [wiki](https://github.com/SpACE-plzlab/spacelab-documentation/wiki/using-google-cloud) for more info)
