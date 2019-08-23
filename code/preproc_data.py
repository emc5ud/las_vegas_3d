# -*- coding: utf-8 -*-
"""
Image preprocessing module

* 256/650 rescale MUL images
* used by v9s and v17

Author: Kohei <i@ho.lc>
"""
from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path
import argparse
import math
import glob
import warnings

import click
import scipy
import tqdm
import tables as tb
import pandas as pd
import numpy as np
import skimage.transform
import rasterio
import shapely.wkt

ORIGINAL_SIZE = 650
INPUT_SIZE = 256

BASE_TRAIN_DIR = "/data/train/roads"
WORKING_DIR = "/data/working"
IMAGE_DIR = "/data/working/images/preproc"

# Input files
FMT_TRAIN_SUMMARY_PATH = str(
    Path(BASE_TRAIN_DIR) /
    Path("{prefix:s}_Train/") /
    Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
FMT_TRAIN_RGB_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TEST_RGB_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TRAIN_MSPEC_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
FMT_TEST_MSPEC_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
FMT_TRAIN_HEIGHT_IMAGE_PATH = str(
    Path("{datapath:s}/") /
    Path("dsm/dsm_{image_id:s}.tif"))
FMT_TEST_HEIGHT_PATH = str(
    Path("{datapath:s}/") /
    Path("dsm/dsm_{image_id:s}.tif"))

# Preprocessing result
FMT_RGB_BANDCUT_TH_PATH = IMAGE_DIR + "/rgb_bandcut{}.csv"
FMT_MUL_BANDCUT_TH_PATH = IMAGE_DIR + "/mul_bandcut{}.csv"
FMT_HEIGHT_BANDCUT_TH_PATH = IMAGE_DIR + "/height_bandcut{}.csv"
# Image list, Image container and mask container
FMT_VALTRAIN_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTRAIN_IM_STORE = IMAGE_DIR + "/valtrain_{}_im.h5"
FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"

FMT_VALTEST_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_VALTEST_IM_STORE = IMAGE_DIR + "/valtest_{}_im.h5"
FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"

FMT_IMMEAN = IMAGE_DIR + "/{}_immean.h5"
FMT_MULMEAN = IMAGE_DIR + "/{}_mulmean.h5"

FMT_TEST_IMAGELIST_PATH = IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
FMT_TEST_IM_STORE = IMAGE_DIR + "/test_{}_im.h5"
FMT_TEST_MUL_STORE = IMAGE_DIR + "/test_{}_mul.h5"

# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger('spacenet2')
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)


# Fix seed for reproducibility
np.random.seed(1145141919)


def directory_name_to_area_id(datapath):
    """
    Directory name to AOI number

    Usage:

        >>> directory_name_to_area_id("/data/test/AOI_2_Vegas")
        2
    """
    dir_name = Path(datapath).name
    if dir_name.startswith('AOI_2_Vegas'):
        return 2
    elif dir_name.startswith('AOI_3_Paris'):
        return 3
    elif dir_name.startswith('AOI_4_Shanghai'):
        return 4
    elif dir_name.startswith('AOI_5_Khartoum'):
        return 5
    else:
        raise RuntimeError("Unsupported city id is given.")


# calculates the 2nd and 98th percentile band value for each band
# stores output in a csv in /data/working
def calc_rgb_multiband_cut_threshold(area_id, datapath):
    rows = []
    band_cut_th = __calc_rgb_multiband_cut_threshold(area_id, datapath)
    prefix = area_id_to_prefix(area_id)
    row = dict(prefix=area_id_to_prefix(area_id))
    row['area_id'] = area_id
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv(
        FMT_RGB_BANDCUT_TH_PATH.format(prefix), index=False)


# only calculates based on the first 500 images in train then test... smart
# stores the 98th and 2nd percentile for each band
def __calc_rgb_multiband_cut_threshold(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    band_values = {k: [] for k in range(3)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(3)}

    image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(image_id, datapath)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(3):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(image_id, datapath)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(3):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    logger.info("Calc percentile point ...")
    for i_chan in range(3):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    return band_cut_th


# calculates the 2nd and 98th percentile band value for each band
# stores output in a csv in /data/working
def calc_mul_multiband_cut_threshold(area_id, datapath):
    rows = []
    band_cut_th = __calc_mul_multiband_cut_threshold(area_id, datapath)
    prefix = area_id_to_prefix(area_id)
    row = dict(prefix=area_id_to_prefix(area_id))
    row['area_id'] = area_id
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv(
        FMT_MUL_BANDCUT_TH_PATH.format(prefix),
        index=False)


# calculates the 2nd and 98th percentile band value for each band
# stores output in a csv in /data/working
def calc_height_cut_threshold(area_id, datapath):
    rows = []
    band_cut_th = __calc_height_cut_threshold(area_id, datapath)
    prefix = area_id_to_prefix(area_id)
    row = dict(prefix=area_id_to_prefix(area_id))
    row['area_id'] = area_id
    for chan_i in band_cut_th.keys():
        row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
        row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
    rows.append(row)
    pd.DataFrame(rows).to_csv(
        FMT_HEIGHT_BANDCUT_TH_PATH.format(prefix),
        index=False)


def __calc_mul_multiband_cut_threshold(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    band_values = {k: [] for k in range(8)}
    band_cut_th = {k: dict(max=0, min=0) for k in range(8)}

    image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(
            image_id, datapath, mul=True)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(8):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_image_path_from_imageid(
            image_id, datapath, mul=True)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(8):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(
                    [v for v in values_ if v != 0]
                )  # Remove sensored mask
                band_values[i_chan].append(values_)

    logger.info("Calc percentile point ...")
    for i_chan in range(8):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = scipy.percentile(
            band_values[i_chan], 98)
        band_cut_th[i_chan]['min'] = scipy.percentile(
            band_values[i_chan], 2)
    return band_cut_th


def __calc_height_cut_threshold(area_id, datapath):
    prefix = area_id_to_prefix(area_id)
    band_values = {0: []}
    band_cut_th = {0: dict(max=0, min=0)}

    image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_height_path_from_imageid(image_id, datapath)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(1):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(values)
                band_values[i_chan].append(values_)

    image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
        prefix=prefix)).ImageId.tolist()
    for image_id in tqdm.tqdm(image_id_list[:500]):
        image_fn = get_train_height_path_from_imageid(image_id, datapath)
        with rasterio.open(image_fn, 'r') as f:
            values = f.read().astype(np.float32)
            for i_chan in range(1):
                values_ = values[i_chan].ravel().tolist()
                values_ = np.array(values)
                band_values[i_chan].append(values_)

    logger.info("Calc height percentile point ...")
    for i_chan in range(1):
        band_values[i_chan] = np.concatenate(
            band_values[i_chan]).ravel()
        band_cut_th[i_chan]['max'] = np.max(band_values[i_chan])
        band_cut_th[i_chan]['min'] = np.min(band_values[i_chan])
    return band_cut_th


def area_id_to_prefix(area_id):
    area_dict = {
        2: 'AOI_2_Vegas',
        3: 'AOI_3_Paris',
        4: 'AOI_4_Shanghai',
        5: 'AOI_5_Khartoum',
    }
    return area_dict[area_id]


def image_id_to_prefix(image_id):
    """
    `AOI_3_Paris_img585` -> `AOI_3_Paris`
    """
    prefix = image_id.split('img')[0][:-1]
    return prefix


def get_train_image_path_from_imageid(image_id, datapath, mul=False):
    prefix = image_id_to_prefix(image_id)
    if mul:
        return FMT_TRAIN_MSPEC_IMAGE_PATH.format(
            datapath=datapath, prefix=prefix, image_id=image_id)
    else:
        return FMT_TRAIN_RGB_IMAGE_PATH.format(
            datapath=datapath, prefix=prefix, image_id=image_id)


def get_train_height_path_from_imageid(image_id, datapath):
    prefix = image_id_to_prefix(image_id)
    return FMT_TRAIN_HEIGHT_IMAGE_PATH.format(
        datapath=datapath, prefix=prefix, image_id=image_id)


def get_test_image_path_from_imageid(image_id, datapath, mul=False):
    if mul:
        return FMT_TEST_MSPEC_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)
    else:
        return FMT_TEST_RGB_IMAGE_PATH.format(
            datapath=datapath, image_id=image_id)


def get_test_height_path_from_imageid(image_id, datapath):
    return FMT_TEST_HEIGHT_PATH.format(
        datapath=datapath, image_id=image_id)


def __load_rgb_bandstats(area_id):
    """
    Usage:

        >>> __load_rgb_bandstats(3)
        {
          0: {
              'max': 462.0,
              'min': 126.0,
          },
          1: {
              'max': 481.0,
              'min': 223.0,
          },
          2: {
              'max': 369.0,
              'min': 224.0,
          },
        }
    """
    prefix = area_id_to_prefix(area_id)
    fn_stats = FMT_RGB_BANDCUT_TH_PATH.format(prefix)
    df_stats = pd.read_csv(fn_stats, index_col='area_id')
    r = df_stats.loc[area_id]

    stats_dict = {}
    for chan_i in range(3):
        stats_dict[chan_i] = dict(
            min=r['chan{}_min'.format(chan_i)],
            max=r['chan{}_max'.format(chan_i)])
    return stats_dict


def __load_mul_bandstats(area_id):
    prefix = area_id_to_prefix(area_id)
    fn_stats = FMT_MUL_BANDCUT_TH_PATH.format(prefix)
    df_stats = pd.read_csv(fn_stats, index_col='area_id')
    r = df_stats.loc[area_id]

    stats_dict = {}
    for chan_i in range(8):
        stats_dict[chan_i] = dict(
            min=r['chan{}_min'.format(chan_i)],
            max=r['chan{}_max'.format(chan_i)])
    return stats_dict


def __load_band_cut_th(band_fn):
    df = pd.read_csv(band_fn, index_col='area_id')
    all_band_cut_th = {area_id: {} for area_id in range(2, 6)}
    for area_id, row in df.iterrows():
        for chan_i in range(3):
            all_band_cut_th[area_id][chan_i] = dict(
                min=row['chan{}_min'.format(chan_i)],
                max=row['chan{}_max'.format(chan_i)],
            )
    return all_band_cut_th


# loads, shuffles image IDs and splits to a 70/30 train/val split
# extracts the image IDs from the summary data and writes to a new file
def prep_valtrain_valtest_imagelist(area_id):
    prefix = area_id_to_prefix(area_id)
    df = _load_train_summary_data(area_id)
    df_agg = df.groupby('ImageId').agg('first')

    image_id_list = df_agg.index.tolist()
    np.random.shuffle(image_id_list)
    size_valtrain = int(len(image_id_list) * 0.7)
    #size_valtest = len(image_id_list) - size_valtrain

    base_dir = Path(FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)).parent
    if not base_dir.exists():
        base_dir.mkdir(parents=True)

    pd.DataFrame({'ImageId': image_id_list[:size_valtrain]}).to_csv(
        FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
        index=False)
    pd.DataFrame({'ImageId': image_id_list[size_valtrain:]}).to_csv(
        FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
        index=False)


# grabs test list from filenames of test folder
def prep_test_imagelist(area_id, datapath):
    prefix = area_id_to_prefix(area_id)

    image_id_list = glob.glob(str(
        Path(datapath) /
        Path("./PAN/PAN_{prefix:s}_*.tif")).format(prefix=prefix))
    image_id_list = [path.split("PAN_")[-1][:-4] for path in image_id_list]
    pd.DataFrame({'ImageId': image_id_list}).to_csv(
        FMT_TEST_IMAGELIST_PATH.format(prefix=prefix),
        index=False)


def _load_train_summary_data(area_id):
    prefix = area_id_to_prefix(area_id)
    fn = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df = pd.read_csv(fn)
    # df.loc[:, 'ImageId'] = df.ImageId.str[4:]
    return df

# >>> -------------------------------------------------------------


@click.group()
def cli():
    pass


@cli.command()
@click.argument('datapath', type=str)
def preproc_train(datapath):
    """ train.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id) #eg. AOI_2_Vegas
    logger.info("Preproc for training on {}".format(prefix))

    # Imagelist
    # Generate List of Image IDs for train and test and save to file
    if Path(FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)).exists():
        logger.info("Generate IMAGELIST csv ... skip")
    else:
        logger.info("Generate IMAGELIST csv")
        prep_valtrain_valtest_imagelist(area_id)

    # Band stats (RGB)
    if Path(FMT_RGB_BANDCUT_TH_PATH.format(prefix)).exists():
        logger.info("Generate band stats csv (RGB) ... skip")
    else:
        logger.info("Generate band stats csv (RGB)")
        calc_rgb_multiband_cut_threshold(area_id, datapath)

    # Band stats (MUL)
    if Path(FMT_MUL_BANDCUT_TH_PATH.format(prefix)).exists():
        logger.info("Generate band stats csv (MUL) ... skip")
    else:
        logger.info("Generate band stats csv (MUL)")
        calc_mul_multiband_cut_threshold(area_id, datapath)

    #Band stats (HEIGHT)
    if Path(FMT_HEIGHT_BANDCUT_TH_PATH.format(prefix)).exists():
        logger.info("Generate band stats csv (HEIGHT) ... skip")
    else:
        logger.info("Generate band stats csv (HEIGHT)")
        calc_height_cut_threshold(area_id, datapath)


    # DONE!
    logger.info("Preproc for training on {} ... done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def preproc_test(datapath):
    """ test.sh """
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info("preproc_test for {}".format(prefix))

    # Imagelist
    if Path(FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)).exists():
        logger.info("Generate IMAGELIST for inference ... skip")
    else:
        logger.info("Generate IMAGELIST for inference")
        prep_test_imagelist(area_id, datapath)

    logger.info("preproc_test for {} ... done".format(prefix))


if __name__ == '__main__':
    cli()