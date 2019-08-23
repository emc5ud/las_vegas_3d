from shapely.geometry import Polygon
import shapely.wkt
import pandas as pd
import gdal
import tqdm
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import numpy as np


def georeference_pixel_point_train(point, image_id, extents):
    #print(fn_im)
    x = point[0]
    y = point[1]

    #print("inside: " , image_id, extents[image_id])
    ulx, xres, xskew, uly, yskew, yres = extents[image_id]
    lrx = ulx + (650 * xres)
    lry = uly + (650 * yres)
    x = x * xres + ulx
    y = y * yres + uly
    return (x, y)

def getHeightInsidePoly(polygon, image_id, extents):
    geoms = [mapping(polygon)]
    with rasterio.open("/data/train/AOI_2_Vegas_Train/dsm/dsm_" + image_id + '.tif') as src:
        out_image, out_transform = mask(src, geoms, crop=True)
        print(out_image.shape)
        data = out_image.data[0]
        return np.percentile(data, 85)

fn_val_pred = '/data/working/models/v17/AOI_2_Vegas_eval_poly.csv'
df = pd.read_csv(fn_val_pred)
df['PolygonWKT_Geo'] = df['PolygonWKT_Pix']
df['Height'] = 0


extents = {}

for image_id in set(df.ImageId):
    fn_im = "/data/train/AOI_2_Vegas_Train/PAN/PAN_" + image_id + ".tif"
    extents[image_id] = gdal.Open(fn_im).GetGeoTransform()


for idx, row in df.iterrows():
    if row.PolygonWKT_Pix == 'POLYGON EMPTY':
        continue
    shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
    coords = list(shape_obj.exterior.coords)
    newPoly = Polygon(map(lambda x: georeference_pixel_point_train(x, row.ImageId, extents), coords))
    #print(list(newPoly.exterior.coords))
    df.set_value(idx, 'PolygonWKT_Geo', newPoly)
    h = getHeightInsidePoly(newPoly, row.ImageId, extents)
    df.set_value(idx, 'Height', h)
    print(idx, h)


df.to_csv('converted.csv')


