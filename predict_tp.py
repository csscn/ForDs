import numpy as np
import argparse
import pickle
import netCDF4 as nc
#import xarray as xr
#import tensorflow as tf
from topo import downscale_precipitation
from utils import save2nc
from multiprocessing import Process
import cloudpickle
import time



def pred(args, day_of_month,rf_regressor, rf_classifer, region_name):
    # load coare DEM data （20x20)
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>=args.blat) & (lat_coarse<=args.ulat))[0]
    lon_coarse_index = np.where((lon_coarse>=args.llon) & (lon_coarse<=args.rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['hgt'][lat_coarse_index][:,lon_coarse_index]

    # load fine DEM data (1100x1100)
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_height.nc', 'r')
    lat_fine, lon_fine = f['latitude'][:], f['longitude'][:]
    lat_fine_index = np.where((lat_fine>=lat_coarse[-1]) & (lat_fine<=lat_coarse[0]))[0]
    lon_fine_index = np.where((lon_fine>=lon_coarse[0]) & (lon_fine<=lon_coarse[-1]))[0]
    lat_fine, lon_fine = lat_fine[lat_fine_index], lon_fine[lon_fine_index]
    elev_fine = f['hgt'][lat_fine_index][:,lon_fine_index]
    nx, ny = elev_fine.shape

    # load test data
    x = np.load('x_test/{region_name}/x_test_{year:04}_{month}_{day}.npy'.format(
        year=args.year, month=args.month, day=day_of_month, region_name=region_name))
    idx = np.unique(np.where(np.isnan(x))[0])
    all_idx = np.arange(x.shape[0])
    rest_idx = np.delete(all_idx, idx, axis=0)
    x1 = np.delete(x, idx, axis=0)
    y = np.full((x.shape[0],1), np.nan)
    del x

    # pred by rf
    value_fine = rf_regressor.predict(x1)
    mask_fine = rf_classifer.predict(x1)
    value_fine = 10**(value_fine)-1 # log-trans
    value_fine[mask_fine==0] = 0
    del mask_fine
    y[rest_idx] = value_fine[:,np.newaxis]
    del value_fine
    y = y.reshape(-1, nx, ny)
    y[y<0] = 0
    save2nc('tp_rf', args.year, args.month, day_of_month, np.array(y), lat_fine, lon_fine)


def pred_automl(args, day_of_month, am_regressor, am_classifer, region_name):
    print("enter pred_automl")
    # load coare DEM data （20x20)
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>=args.blat) & (lat_coarse<=args.ulat))[0]
    lon_coarse_index = np.where((lon_coarse>=args.llon) & (lon_coarse<=args.rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['hgt'][lat_coarse_index][:,lon_coarse_index]

    # load fine DEM data (1100x1100)
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_height.nc', 'r')
    lat_fine, lon_fine = f['latitude'][:], f['longitude'][:]
    lat_fine_index = np.where((lat_fine>=lat_coarse[-1]) & (lat_fine<=lat_coarse[0]))[0]
    lon_fine_index = np.where((lon_fine>=lon_coarse[0]) & (lon_fine<=lon_coarse[-1]))[0]
    lat_fine, lon_fine = lat_fine[lat_fine_index], lon_fine[lon_fine_index]
    elev_fine = f['hgt'][lat_fine_index][:,lon_fine_index]
    nx, ny = elev_fine.shape
    print("elev_fine.shape: ",nx,ny)

    # load test data
    #m = int(138240000/150)
    #x = np.load('x_test/{region_name}/x_test_{year:04}_{month}_{day}.npy'.format(
    #    year=args.year, month=args.month, day=day_of_month, region_name=region_name))#[m*i:m*(i+1)]
    """
    f_t = nc.Dataset("/tera05/lilu/ForDs/ERA5_DS/HRB/t2m/ERA5_{region_name}_fine_{year:04}_{month:02}_t2m.nc".format(
                  year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    f_q = nc.Dataset("/tera05/lilu/ForDs/ERA5_DS/HRB/q/ERA5_{region_name}_fine_{year:04}_{month:02}_q.nc".format(
                  year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    f_d2m = nc.Dataset("/tera05/lilu/ForDs/ERA5_DS/HRB/d2m/ERA5_{region_name}_fine_{year:04}_{month:02}_d2m.nc".format(
                  year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    f_sp = nc.Dataset("/tera05/lilu/ForDs/ERA5_DS/HRB/sp/ERA5_{region_name}_fine_{year:04}_{month:02}_sp.nc".format(
                  year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    f_sw = nc.Dataset("/tera05/lilu/ForDs/ERA5_DS/HRB/msdwswrf/ERA5_{region_name}_fine_{year:04}_{month:02}_msdwswrf.nc".format(
                  year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    f_lw = nc.Dataset("/tera05/lilu/ForDs/ERA5_DS/HRB/msdwlwrf/ERA5_{region_name}_fine_{year:04}_{month:02}_msdwlwrf.nc".format(
                  year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    f_ws =nc.Dataset("/tera05/lilu/ForDs/ERA5_DS/HRB/ws/ERA5_{region_name}_fine_{year:04}_{month:02}_ws.nc".format(
                  year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    #f_ws_modify = nc.Dataset("ERA5_HH_fine_{year:04}_{month:02}_{day:02}_ws.nc".format(
    #              year=args.year, month=args.month, day=day_of_month, region_name=region_name),"r")
    #ws_modify = np.array(f_ws_modify["ws"].data[(day_of_month-1)*24:(day_of_month-1)*24+24])
    ws = np.array(f_ws["ws"][(day_of_month-1)*24:(day_of_month-1)*24+24])
    print("ws.shape:",ws.shape)
    t2m = np.array(f_t["t2m"][(day_of_month-1)*24:(day_of_month-1)*24+24])
    q = np.array(f_q["q"][(day_of_month-1)*24:(day_of_month-1)*24+24])
    d2m = np.array(f_d2m["d2m"][(day_of_month-1)*24:(day_of_month-1)*24+24])
    sp = np.array(f_sp["sp"][(day_of_month-1)*24:(day_of_month-1)*24+24])
    sw = np.array(f_sw["msdwswrf"][(day_of_month-1)*24:(day_of_month-1)*24+24])
    lw = np.array(f_lw["msdwlwrf"][(day_of_month-1)*24:(day_of_month-1)*24+24])
    downscale_precipitation(t2m, d2m, sp, q, lw, sw, ws, lat_fine, lon_fine, elev_fine, args.year, args.month, day_of_month)
    """
    x = np.load('/tera05/lilu/ForDs/run/x_test/HH/x_test_{year:04}_{month}_{day}.npy'.format(
        year=args.year, month=args.month, day=day_of_month, region_name=region_name))#[m*i:m*(i+1)]
    #x[6] = ws_modify # modified by chenss:replace wrong wind
    idx = np.unique(np.where(np.isnan(x))[0])
    all_idx = np.arange(x.shape[0])
    rest_idx = np.delete(all_idx, idx, axis=0)
    x1 = np.delete(x, idx, axis=0)
    y = np.full((x.shape[0],1), np.nan)
    del x
    print(x1.shape)
    # pred by rf
    t1 = time.time()
    value_fine = am_regressor.predict(x1)
    t2 = time.time()
    print(t2-t1)
    t1 = time.time()
    mask_fine = am_classifer.predict(x1)
    t2 = time.time()
    print(t2-t1)
    value_fine = 10**(value_fine)-1 # log-trans
    value_fine[mask_fine==0] = 0
    del mask_fine
    y[rest_idx] = value_fine[:,np.newaxis]
    del value_fine
    y = y.reshape(-1, nx, ny)
    y[y<0] = 0    
    #np.save('tp_automl_{year}_{month}_{day}_{slice}.npy'.format(
    #    year=args.year, month=args.month, day=day_of_month, slice=i), y)
    save2nc('tp_automl', args.year, args.month, day_of_month, np.array(y), lat_fine, lon_fine)


def pred_mtl(args, day_of_month, mtl_dnn):
    # load coare DEM data （20x20)
    f = nc.Dataset(DATA_PATH+'DEM/SRTM/ERA5_height.nc', 'r')
    lat_coarse, lon_coarse = f['latitude'][:], f['longitude'][:]
    lat_coarse_index = np.where((lat_coarse>=args.blat) & (lat_coarse<=args.ulat))[0]
    lon_coarse_index = np.where((lon_coarse>=args.llon) & (lon_coarse<=args.rlon))[0]
    lat_coarse, lon_coarse = lat_coarse[lat_coarse_index], lon_coarse[lon_coarse_index]
    elev_coarse = f['hgt'][lat_coarse_index][:,lon_coarse_index]

    # load fine DEM data (1100x1100)
    f = nc.Dataset(DATA_PATH+'DEM/MERITDEM/MERITDEM_height.nc', 'r')
    lat_fine, lon_fine = f['latitude'][:], f['longitude'][:]
    lat_fine_index = np.where((lat_fine>=lat_coarse[-1]) & (lat_fine<=lat_coarse[0]))[0]
    lon_fine_index = np.where((lon_fine>=lon_coarse[0]) & (lon_fine<=lon_coarse[-1]))[0]
    lat_fine, lon_fine = lat_fine[lat_fine_index], lon_fine[lon_fine_index]
    elev_fine = f['hgt'][lat_fine_index][:,lon_fine_index]
    nx, ny = elev_fine.shape

    # pred by multi-dnn
    x = np.load('x_test/x_test_{year:04}_{month}_{day}.npy'.format(
        year=args.year, month=args.month, day=day_of_month))

    # normalize
    t2m_fine = x[:,0]
    d2m_fine = x[:,1]
    sp_fine = x[:,2]
    Q_fine = x[:,3]
    strd_fine = x[:,4]
    ssrd_fine = x[:,5]
    ws_fine = x[:,6]
    julian_day = x[:,7]
    lat_fine_ = x[:,-3]
    lon_fine_ = x[:,-2]
    elev_fine = x[:,-1]
    print(x.shape)
    
    del x
    norm_const = np.load('norm_params.npy')
    t2m_min, t2m_max = norm_const[0,0], norm_const[0,1]
    d2m_min, d2m_max = norm_const[1,0], norm_const[1,1]
    Q_min, Q_max = norm_const[2,0], norm_const[2,1]
    strd_min, strd_max = norm_const[3,0], norm_const[3,1]
    ssrd_min, ssrd_max = norm_const[4,0], norm_const[4,1]
    sp_min, sp_max = norm_const[5,0], norm_const[5,1]
    ws_min, ws_max = norm_const[6,0], norm_const[6,1]
    elev_max = norm_const[7,0]
    
    t2m_fine = (t2m_fine-t2m_min)/(t2m_max-t2m_min)
    d2m_fine = (d2m_fine-d2m_min)/(d2m_max-d2m_min)
    Q_fine = (Q_fine-Q_min)/(Q_max-Q_min)
    strd_fine = (strd_fine-strd_min)/(strd_max-strd_min)
    ssrd_fine = (ssrd_fine-ssrd_min)/(ssrd_max-ssrd_min)
    sp_fine = (sp_fine-sp_min)/(sp_max-sp_min)
    ws_fine = (ws_fine-ws_min)/(ws_max-ws_min)
    elev_fine = elev_fine/elev_max
    lat_fine_ = lat_fine_/360
    lon_fine_ = lon_fine_/360
    
    x = np.stack([t2m_fine, d2m_fine, Q_fine, strd_fine, ssrd_fine, sp_fine, ws_fine, julian_day, lat_fine_, lon_fine_, elev_fine], axis=1)
    print(x.shape)
    idx = np.unique(np.where(np.isnan(x))[0])
    all_idx = np.arange(x.shape[0])
    rest_idx = np.delete(all_idx, idx, axis=0)
    x1 = np.delete(x, idx, axis=0)
    y = np.full((x.shape[0],1), np.nan)
    del x

    value_fine, mask_fine_prob = mtl_dnn(x1)
    value_fine, mask_fine_prob = np.array(value_fine), np.array(mask_fine_prob)
    mask_fine = np.zeros_like(mask_fine_prob)
    mask_fine[mask_fine_prob>0.5] = 1
    value_fine = 10**(value_fine)-1 # log-trans
    value_fine[mask_fine==0] = 0
    y[rest_idx] = value_fine
    tp_fine = y.reshape(-1, nx, ny)
    tp_fine[tp_fine<0] = 0
    print(tp_fine.shape)
    save2nc('tp_mtl_dnn', args.year, args.month, args.begin_day, np.array(tp_fine), lat_fine, lon_fine)


def par_pred(args, rf_regressor, rf_classifer):
    # generate hour length according to year and month
    if ((args.year%4==0) and (args.year%100!=0)) or args.year%400==0:
        month_day = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    for i in range(args.begin_day, args.end_day):
        job = Process(target=pred,  args=(args,i,rf_regressor, rf_classifer, args.region_name))
        job.start()
    job.join()


def par_pred_automl(args, rf_regressor, rf_classifer):
    # generate hour length according to year and month
    if ((args.year%4==0) and (args.year%100!=0)) or args.year%400==0:
        month_day = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    for i in range(args.begin_day, args.end_day):
        #for j in range(150):
        print("Process begin")
        job = Process(target=pred_automl,  args=(args,i,rf_regressor, rf_classifer, args.region_name))
        print("Process OK")
        job.start()
    job.join()

if __name__ == '__main__':

    # Perform downscaling of precipitation if you have prepared the trained model
    DATA_PATH='/tera05/lilu/ForDs/data/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2018)
    parser.add_argument('--month', type=int, default=1)
    parser.add_argument('--begin_day', type=int, default=1)
    parser.add_argument('--end_day', type=int, default=2)
    parser.add_argument('--blat', type=float, default=23.5)
    parser.add_argument('--ulat', type=float, default=25.5)
    parser.add_argument('--llon', type=float, default=112.5)
    parser.add_argument('--rlon', type=float, default=114.5)
    parser.add_argument('--region_name', type=str, default='SG')
    args = parser.parse_args()
    DATA_PATH=DATA_PATH+args.region_name+'/'


    # load trained model
    #mtl_dnn = tf.keras.models.load_model("mtl_dnn_2cls_{year}.h5".format(year=args.year))
    #pred_(args, args.begin_day, mtl_dnn)

    
    f = open("/tera05/lilu/ForDs/run/models/{region_name}/automl_origin/automl_cls_{year}_era5.pkl".format(year=args.year, region_name=args.region_name),'rb')
    rf_classifer = cloudpickle.load(f)
    f = open("/tera05/lilu/ForDs/run/models/{region_name}/automl_origin/automl_reg_{year}_era5.pkl".format(year=args.year, region_name=args.region_name),'rb') 
    rf_regressor = cloudpickle.load(f)
    
    # generate hour length according to year and month
    if ((args.year%4==0) and (args.year%100!=0)) or args.year%400==0:
        month_day = [31,29,31,30,31,30,31,31,30,31,30,31]
    else:
        month_day = [31,28,31,30,31,30,31,31,30,31,30,31]
    
    rf_classifer.n_jobs = 1
    rf_regressor.n_jobs = 1
    
    # # generate multiprocessing for each 24 hour interval
    # #for i in range(month_day[args.month-1]):
    # #    print("Now we predict {year}-{month:02}".format(year=args.year, month=args.month))
    # #    pred(args, i, rf_regressor, rf_classifer)
    print("OK")
    par_pred_automl(args, rf_regressor, rf_classifer)
    """
    
    f = open("/tera05/lilu/ForDs/run/models/{region_name}/automl_cls_{year}_era5.pkl".format(year=args.year, region_name=args.region_name),'rb')
    rf_classifer = cloudpickle.load(f)
    f = open("/tera05/lilu/ForDs/run/models/{region_name}/automl_reg_{year}_era5.pkl".format(year=args.year, region_name=args.region_name),'rb') 
    rf_regressor = cloudpickle.load(f)

    #rf_classifer.n_jobs = 1
    #rf_regressor.n_jobs = 1
    pred_automl(args, args.begin_day, rf_regressor, rf_classifer, args.region_name)
    #par_pred_automl(args, rf_regressor, rf_classifer)
    """
