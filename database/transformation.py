import pandas as pd


def getSTN(param):
    if param == 'temperature':
        url = 'https://data.geo.admin.ch/ch.meteoschweiz.messwerte-lufttemperatur-10min/ch.meteoschweiz.messwerte-lufttemperatur-10min_fr.csv'
        df = pd.read_csv(url, sep=';', header=0, skipfooter=4, engine='python',
                         encoding='latin1',
                         usecols=[0, 1, 3, 4, 5, 7, 8, 9, 10],
                         names=['station', 'short', 'temperature', 'datetime',
                                'elevation', 'coordE', 'coordN', 'lat', 'lon'])
    if param == 'pressure':
        url = 'https://data.geo.admin.ch/ch.meteoschweiz.messwerte-luftdruck-qfe-10min/ch.meteoschweiz.messwerte-luftdruck-qfe-10min_fr.csv'
        df = pd.read_csv(url, sep=';', header=0, skipfooter=4, engine='python', encoding='latin1', usecols=[0,1,3,4,5,6,7,8,9], names=['station', 'short', 'pressure', 'datetime', 'elevation', 'coordE', 'coordN', 'lat', 'lon'])
    if param == 'dewpoint':
        url = 'https://data.geo.admin.ch/ch.meteoschweiz.messwerte-taupunkt-10min/ch.meteoschweiz.messwerte-taupunkt-10min_fr.csv'
        df = pd.read_csv(url, sep=';', header=0, skipfooter=4, engine='python', encoding='latin1', usecols=[0,1,3,4,5,7,8,9,10], names=['station', 'short', 'dewpoint', 'datetime', 'elevation', 'coordE', 'coordN', 'lat', 'lon'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.dropna()


df = getSTN('temperature')
curTime = pd.read_csv("database/curTime.csv", parse_dates=['Current Time'])

if df['datetime'][0] > curTime['Current Time'][0]:

    pd.DataFrame({'Current Time': df['datetime'][0]}, index=[0]).to_csv("database/curTime.csv")

    import xarray as xr
    import rioxarray as rxr
    import numpy as np
    import gstools as gs
    from shapely.geometry import Point
    from geopandas import GeoDataFrame
    from pygam import LinearGAM, s, te
    import metpy.calc as mpcalc
    from metpy.units import units
    from functools import reduce
    import os.path


    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))


    def getDEM(res):
        rds = xr.open_dataarray('database/dem.nc')
        rds = rds.coarsen(x=res, y=res, boundary='trim').mean()
        return rds, rds.x.values, rds.y.values


    def getTopo(res):
        topo = xr.open_dataarray('database/topo.nc')
        topo = topo.coarsen(x=res, y=res, boundary='trim').mean()
        topo.name = 'topo'
        return topo


    def appendTopo(df, topo):
        topoLst = []
        for i, row in df.iterrows():
            topoLst.append(float(topo.sel(y=row['coordN'], x=row['coordE'],
                                          method='nearest').values))
        df['topo'] = topoLst
        return df


    def fitTrend(df, y, X):
        lams = np.random.rand(100, X.shape[1])
        lams = lams * 6 - 3
        lams = 10 ** lams
        parse = 'te(0,1) + s(2) + s(3)'
        gam = LinearGAM(eval(parse)).gridsearch(X, y, lam=lams)
        pList = ['Nombre de stations', 'Trend Pseudo R-Squared',
                 'P-value x*y', 'P-value altitude', 'P-value topograhie']
        sList = [gam.statistics_['n_samples'], gam.statistics_['pseudo_r2']['explained_deviance']] + gam.statistics_['p_values'][:-1]
        return gam, pd.DataFrame({'Paramètres': pList, 'Scores': np.array(sList).round(3).astype(str)})


    def genTrend(gam, topo):
        df_pred = rds.to_dataframe().reset_index().rename(columns={'y': 'coordN', 'x': 'coordE', 'data': 'elevation'})
        df_pred = df_pred.merge(topo.to_dataframe().reset_index().rename(columns={'y': 'coordN', 'x': 'coordE'})).dropna()
        dat_gam = xr.DataArray(gam.predict(df_pred[['coordN' , 'coordE', 'elevation', 'topo']].to_numpy()).reshape((len(gridy),len(gridx))),
                               coords={'y': gridy, 'x': gridx}, dims=["y", "x"])
        return dat_gam


    def fitVariogram(df):
        bin_center, gamma = gs.vario_estimate((df['coordE'], df['coordN']), df['residuals'], np.arange(0, 50000, 2000))
        cov_model = gs.Gaussian(dim=2)
        cov_model.fit_variogram(bin_center, gamma, nugget=False)
        return cov_model


    def genKriging(df, cov_model):
        krig = gs.krige.Simple(cov_model, [df['coordE'], df['coordN']], df['residuals'], exact=True)
        krig.structured([gridx, gridy])
        dat_krig = xr.DataArray(krig.all_fields[0].transpose(), coords={'y': gridy, 'x': gridx}, dims=["y", "x"])
        return dat_krig


    def interpolate(param):
        df = getSTN(param)
        df = appendTopo(df, topo)
        y = df[param]
        X = df[['coordN', 'coordE', 'elevation', 'topo']].to_numpy()
        gam, scores = fitTrend(df, y, X)
        dat_gam = genTrend(gam, topo)
        df['residuals'] = gam.deviance_residuals(X, y)
        cov_model = fitVariogram(df)
        dat_krig = genKriging(df, cov_model)
        dat = dat_gam + dat_krig
        dat = dat.rio.write_crs('EPSG:2056').rio.reproject('EPSG:3857')
        dat = dat.where(dat != dat.attrs['_FillValue'], np.nan)
        dat = dat.round(1)
        dat = dat.drop('spatial_ref')
        dat.name = 'data'
        dat.attrs['parameter'] = param
        dat.attrs['datetime'] = str(df['datetime'][0])
        return dat, df.iloc[:, [1, 7, 8, 2]], scores


    def mergeDF(listDF):
        df_merged = reduce(lambda x, y: pd.merge(x, y, on='short'), listDF)
        return df_merged


    def computeThetaE(listXR, listDF):
        dat = df = []
        if listXR:
            dat = mpcalc.equivalent_potential_temperature(listXR[0] * units.hPa,
                                                          listXR[1] * units.degC,
                                                          listXR[2] * units.degC).metpy.convert_units('degC').metpy.dequantify()
            dat.attrs['parameter'] = 'thetaE'
            dat.attrs['datetime'] = str(listXR[0].attrs['datetime'])
        if listDF:
            df = mpcalc.equivalent_potential_temperature(listDF[0].values * units.hPa,
                                                  listDF[1].values * units.degC,
                                                  listDF[2].values * units.degC).to('degC').magnitude

        return dat, df

    def to_csv_proc(df, param):
        geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
        gdf = GeoDataFrame(df[param], geometry=geometry)
        gdf.set_crs(epsg=4326, inplace=True)
        gdf.to_crs(epsg=3857, inplace=True)
        gdf['x'] = gdf.geometry.apply(lambda x: x.x)
        gdf['y'] = gdf.geometry.apply(lambda x: x.y)
        gdf.drop('geometry', axis=1, inplace=True)
        gdf.to_csv('database/' + param + '.csv')

    def to_csv_Hscores(df, param, curTime):
        if param == 'thetaE':
            out = pd.DataFrame({'time': curTime,
                                'num sta': df.iloc[0, 1]}, index=[0])
        else:
            out = pd.DataFrame({'time': curTime,
                                'num sta': df.iloc[0, 1],
                                'trend r2': df.iloc[1, 1],
                                'pval x*y': df.iloc[2, 1],
                                'pval alt': df.iloc[3, 1],
                                'pval topo': df.iloc[4, 1]}, index=[0])
        if os.path.isfile('database/scoresH_' + param + '.csv'):
            out.to_csv('database/scoresH_' + param + '.csv',
                       index=False, mode='a', header=False)
        else:
            out.to_csv('database/scoresH_' + param + '.csv', index=False)

    rds, gridx, gridy = getDEM(2)
    topo = getTopo(2)

    T, df_T, scores_T = interpolate('temperature')
    P, df_P, scores_P = interpolate('pressure')
    D, df_D, scores_D = interpolate('dewpoint')
    df_merged = mergeDF([df_T, df_P, df_D])
    E, Ea = computeThetaE([P, T, D], [df_merged['pressure'], df_merged['temperature'], df_merged['dewpoint']])
    df_E = pd.concat([df_merged.loc[:, ['lat', 'lon']], pd.Series(Ea, name='thetaE')], axis=1)

    T.to_netcdf("database/temperature.nc")
    P.to_netcdf("database/pressure.nc")
    D.to_netcdf("database/dewpoint.nc")
    E.to_netcdf("database/thetaE.nc")

    to_csv_proc(df_T, 'temperature')
    to_csv_proc(df_P, 'pressure')
    to_csv_proc(df_D, 'dewpoint')
    to_csv_proc(df_E, 'thetaE')

    scores_T.to_csv("database/scores_temperature.csv", index=False)
    scores_P.to_csv("database/scores_pressure.csv", index=False)
    scores_D.to_csv("database/scores_dewpoint.csv", index=False)
    scores_E = pd.DataFrame({'Paramètres': ['Nombre de stations'], 'Scores': [len(df_E.index)]})
    scores_E.to_csv("database/scores_thetaE.csv", index=False)

    to_csv_Hscores(scores_T, 'temperature', T.attrs['datetime'])
    to_csv_Hscores(scores_P, 'pressure', P.attrs['datetime'])
    to_csv_Hscores(scores_D, 'dewpoint', D.attrs['datetime'])
    to_csv_Hscores(scores_E, 'thetaE', E.attrs['datetime'])
