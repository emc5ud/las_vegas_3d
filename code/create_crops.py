import gdal
from gdalconst import GA_ReadOnly
import glob

dataPath = "/data/train/AOI_2_Vegas_Train/"

maskPath = "/data/train/AOI_2_Vegas_Train/MUL-PanSharpen/"
dsmFN = dataPath + "overall_raster_00002_max.tif"
outputPath = dataPath + 'dsm/'


for maskFN in glob.glob(maskPath + "*.tif"):
    maskImage = gdal.Open(maskFN)
    geoTransform = maskImage.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * maskImage.RasterXSize
    miny = maxy + geoTransform[5] * maskImage.RasterYSize
    ds = gdal.Open(dsmFN)
    outputFN = outputPath + "dsm_" + "_".join(maskFN.split("_")[-4:])
    ds = gdal.Translate(outputFN, ds, projWin = [minx, maxy, maxx, miny], width= 650, height=650, resampleAlg="cubicspline")
    ds = None
    #call('gdal_translate -projwin ' + ' '.join([str(x) for x in [minx, maxy, maxx, miny]]) +
    #     ' -of GTiff ' + dsmFN + ' ' + outputFN, shell=True)