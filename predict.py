import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os
import rasterio as rio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


def process_predict_DIC():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    pd.set_option('display.max_columns', None)

    data_path = r"DIC.csv"
    df = pd.read_csv(data_path)
    feature_df = df.loc[:, 'CO2':]
    feature_ls = feature_df.columns.tolist()
    feature_df.loc[:, 'RunoffSur'] = feature_df.loc[:, 'RunoffSur'].fillna(np.mean(feature_df.loc[:, 'RunoffSur']))
    feature_df.loc[:, 'RunoffSub'] = feature_df.loc[:, 'RunoffSub'].fillna(np.mean(feature_df.loc[:, 'RunoffSub']))
    label = 'DIC'
    label_df = df[label]
    label_df *= 1e-3 * (feature_df['RunoffSur'] + feature_df['RunoffSub'])

    mask_array = (rio.open("pH_mask.tif", 'r').read(1).astype(bool)
                  & rio.open("China_mask.tif", 'r').read(1).astype(bool))

    params = {
        'learning_rate': 0.02721535118367937,
        'num_leaves': 20,
        'max_depth': 9,
        'colsample_bytree': 0.5357762802027288,
        'reg_lambda': 19.935687038351997,
        'n_estimators': 9842
    }
    optimizer = LGBMRegressor(random_state=42, objective='regression', device="gpu", **params)
    optimizer.fit(feature_df, label_df)

    rows = 4452
    columns = 7792
    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': np.float32,
        'width': columns,
        'height': rows,
        'crs': 'EPSG:4326',
        'transform': rio.transform.from_bounds(west=70, south=15, east=140, north=55, height=rows, width=columns),
        'nodata': np.nan,
        'compress': 'lzw',
        'blockxsize': 512,
        'blockysize': 512,
        'tiled': True,
        'interleave': 'band',
        'predictor': 2
    }

    year_month_feature_ls = ['CO2', 'ETtotal', 'GPP', 'LAIHigh', 'LAILow', 'PDSI', 'Precipitation', 'RA', 'RH', 'RunoffSur',
                             'RunoffSub', 'SMSur', 'SMSub', 'Transpiration', 'TsoilSur', 'TsoilSub', 'VP']
    year_feature_ls = ['GDP', 'HFP', 'PopDens']
    permanent_feature_ls = ['Aspects', 'ClaySur', 'ClaySub', 'Elevation', 'Bareland', 'Grassland', 'Forest', 'Cropland',
                            'Urban', 'Other', 'Lithology', 'OCContSur', 'OCContSub', 'pHSur', 'pHSub', 'RainErosivity',
                            'SiltSur', 'SiltSub', 'SandSur', 'SandSub', 'TWI']
    assert set(year_month_feature_ls + year_feature_ls + permanent_feature_ls) == set(feature_ls)

    folder_path = r'H:\Hydrochemistry\Predict'

    predict_feature_df = pd.DataFrame(data=np.zeros(shape=(int(rows * columns), len(feature_ls))),
                                      columns=feature_ls, dtype=np.float32)

    for var in permanent_feature_ls:
        predict_feature_df[var] = np.mean(np.array([rio.open(tif).read(1).astype(np.float32) for tif in
                                            sorted(glob.glob(os.path.join(folder_path, '*', var, f'{var}.tif')))]), axis=0).flatten()
    del var

    for year in tqdm(range(2000, 2021), total=21, desc='Year Updating...'):
        for var in year_feature_ls:
            predict_feature_df[var] = np.mean(np.array([rio.open(tif).read(1).astype(np.float32) for tif in
                                                        sorted(glob.glob(os.path.join(folder_path, '*', var, f'{var}_{year}.tif')))]), axis=0).flatten()
        del var

        for month in range(1, 13):
            output_path = f'DIC_flux_{year}_{month:02}.tif'
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            if os.path.exists(output_path):
                continue

            for var in year_month_feature_ls:
                predict_feature_df[var] = np.mean(np.array([rio.open(tif).read(1).astype(np.float32) for tif in
                                                        sorted(glob.glob(os.path.join(folder_path, '*', var, f'{var}_{year}_{month:02}.tif')))]), axis=0).flatten()
            del var

            predict_feature_df = predict_feature_df.reindex(columns=feature_ls)
            predict_label_array = optimizer.predict(predict_feature_df).T.reshape((rows, columns))
            predict_label_array = np.clip(predict_label_array, 0, None)
            predict_label_array = np.ma.masked_where(~mask_array, predict_label_array)

            # plt.imshow(predict_label_array, cmap='gray')
            # plt.show()

            with rio.open(output_path, 'w', **metadata) as dst:
                dst.write(predict_label_array, 1)


if __name__ == '__main__':
    process_predict_DIC()