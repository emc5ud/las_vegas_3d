from shapely.geometry import Polygon
import shapely.wkt
import pandas as pd
import gdal
import tqdm
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np


def get_extent_poly(image_id, extents):
    left_x, xres, _, upper_y, _, yres = extents[image_id]
    right_x = left_x + (650 * xres)
    lower_y = upper_y + (650 * yres)
    p = Polygon([(left_x, lower_y), (left_x, upper_y), (right_x, upper_y), (right_x, lower_y)])
    return p

fn_summary = '/data/train/AOI_2_Vegas_Train/summaryData/AOI_2_Vegas_Train_Building_Solutions.csv'

df = pd.read_csv(fn_summary)
extents = {}

print ('reading images')
for image_id in set(df.ImageId):
    fn_im = "/data/train/AOI_2_Vegas_Train/PAN/PAN_" + image_id + ".tif"
    extents[image_id] = gdal.Open(fn_im).GetGeoTransform()

print('writing out extents')
f_out = open('image_id.csv', 'w')
for image_id in set(df.ImageId):
    f_out.write(str(get_extent_poly(image_id, extents)) + "\n")

