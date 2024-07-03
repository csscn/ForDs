#-----------------------------------------------------------------------------
# The machine learning downscale methods for precipitation 
#
# Author: Lu Li
# Reference:
#   Mei et al. (2020): A Nonparametric Statistical Technique for Spatial 
#       Downscaling of Precipitation Over High Mountain Asia, 
#       Water Resourse Research, 56, e2020WR027472.
#-----------------------------------------------------------------------------
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Layer, Add, Conv2D, \
    LeakyReLU, BatchNormalization, Dense, UpSampling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from flaml import AutoML
import cloudpickle


#-----------------------------------------------------------------------------#
# Multitask DNN
#-----------------------------------------------------------------------------#
def mtl_dnn_2cls():
    """Simple edition"""
    inputs = Input(shape=(11))
    out = Dense(32, activation='relu')(inputs)
    out = Dense(32, activation='relu')(out)
    out = Dense(64, activation='relu')(out)
    out = Dense(100)(out)
    out = Dropout(0.5)(out)
    reg_out = Dense(1)(out)
    cls_out = Dense(1, activation='sigmoid')(out)
    mdl = Model(inputs, [reg_out, cls_out])
    mdl.summary()
    return mdl


def train_mtl_dnn(air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  precipitation_coarse,
                  latitude_coarse,
                  longtitude_coarse,
                  elevation_coarse,
                  year,
                  RainThres=0.1):
    # TODO: preprocessing standard
    # calculate precipitation mask in coarse resolution 
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>RainThres] = 1
    
    # build julian/lat/lon/dem
    Nt, Nlat, Nlon = air_temperature_coarse.shape
    julian_day = []
    for i in range(int(Nt/24)):
        for j in range(24): 
            julian_day.append(i+1)
    julian_day = np.array(julian_day)
    julian_day = np.tile(julian_day[:,np.newaxis,np.newaxis], (1, Nlat, Nlon)) 
    latitude_coarse = np.tile(latitude_coarse[np.newaxis,:,np.newaxis], (julian_day.shape[0], 1, Nlon))
    longtitude_coarse = np.tile(longtitude_coarse[np.newaxis,np.newaxis], (julian_day.shape[0], Nlat, 1))
    print(elevation_coarse[np.newaxis].shape)
    elevation_coarse = np.tile(elevation_coarse[np.newaxis], (air_pressure_coarse.shape[0],1,1))
    print(elevation_coarse.shape)
    print(julian_day.shape)
    print(latitude_coarse.shape)
    print(longtitude_coarse.shape)

    # construct input features
    x = np.stack([air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  julian_day,
                  latitude_coarse,
                  longtitude_coarse,
                  elevation_coarse], axis=-1)
    x = x.reshape(-1, x.shape[-1])
    mask = precipitation_mask_coarse.reshape(-1,1)
    value = precipitation_coarse.reshape(-1,1)
    value = np.log10(1+value) 

    # clean np.nan
    mask = np.delete(mask, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(x))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(mask))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(value))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(value))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(value))[0], axis=0)
        
    # train
    mdl = mtl_dnn_2cls()
    mdl.compile(optimizer=Adam(lr=1e-5),
                loss=['mean_squared_error','binary_crossentropy'],
                metrics=['mean_squared_error','accuracy'])
    mdl.fit(x, [value, mask], 
            batch_size=1024, 
            epochs=100, 
            validation_split=0.2)
    
    # save
    mdl.save('mtl_dnn_2cls_{year}.h5'.format(year=year))
    

#-----------------------------------------------------------------------------#
# Singletask DNN
#-----------------------------------------------------------------------------#
def stl_dnn():
    """Simple edition"""
    inputs = Input(shape=(10))
    out = Dense(64, activation='relu')(inputs)
    out = Dense(64, activation='relu')(out)
    out = Dense(1)(out)
    mdl = Model(inputs, out)
    mdl.summary()
    return mdl


def train_stl_dnn(air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  precipitation_coarse,
                  latitude_coarse,
                  longtitude_coarse,
                  elevation_coarse,
                  year,
                  RainThres=0.1):
    # calculate precipitation mask in coarse resolution 
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>RainThres] = 1
    
    # build julian/lat/lon/dem
    Nt, Nlat, Nlon = air_temperature_coarse.shape
    julian_day = []
    for i in range(int(Nt/24)):
        for j in range(24): 
            julian_day.append(i+1)
    julian_day = np.array(julian_day)
    julian_day = np.tile(julian_day[:,np.newaxis,np.newaxis], (1, Nlat, Nlon)) 
    latitude_coarse = np.tile(latitude_coarse[np.newaxis,:,np.newaxis], (julian_day.shape[0], 1, Nlon))
    longtitude_coarse = np.tile(longtitude_coarse[np.newaxis,np.newaxis], (julian_day.shape[0], Nlat, 1))
    elevation_coarse = np.tile(elevation_coarse[np.newaxis], (julian_day.shape[0]))

    # construct input features
    x = np.stack([air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  julian_day,
                  latitude_coarse,
                  longtitude_coarse,
                  elevation_coarse], axis=-1)
    x = x.reshape(-1, x.shape[-1])
    mask = precipitation_mask_coarse.reshape(-1,1)
    value = precipitation_coarse.reshape(-1,1)
    value = np.log10(1+value) 

    # clean np.nan
    mask = np.delete(mask, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(x))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(mask))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(value))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(value))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(value))[0], axis=0)
        
    # train classifer
    classifer = stl_dnn()
    classifer.compile(optimizer=Adam(lr=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    classifer.fit(x, mask, 
                  batch_size=128, 
                  epochs=50, 
                  validation_split=0.2)
    classifer.save('stl_dnn_classifer_{year}.h5'.format(year=year))

    # train regressor
    regressor = stl_dnn()
    regressor.compile(optimizer=Adam(lr=1e-3), loss='mse', metrics=['mse'])
    regressor.fit(x[np.where(mask==1)[0]], value[np.where(mask==1)[0]],
                  batch_size=128, 
                  epochs=50, 
                  validation_split=0.2)
    regressor.save('stl_dnn_regressor_{year}.h5'.format(year=year))


#-----------------------------------------------------------------------------#
# RF from Mei et al. (2022), WRR
#-----------------------------------------------------------------------------#
def train_rf(air_temperature_coarse,
             dew_temperature_coarse,
             air_pressure_coarse,
             specific_pressure_coarse,
             in_longwave_radiation_coarse,
             in_shortwave_radiation_coarse,
             wind_speed_coarse,
             precipitation_coarse,
             latitude_coarse,
             longtitude_coarse,
             elevation_coarse,
             year,
             RainThres=0.1):
    # calculate precipitation mask in coarse resolution 
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>RainThres] = 1
    
    # build julian/lat/lon/dem
    Nt, Nlat, Nlon = air_temperature_coarse.shape
    julian_day = []
    for i in range(int(Nt/24)): # day
        for j in range(24): 
            julian_day.append((i+1))
    julian_day = np.array(julian_day)
    julian_day = np.tile(julian_day[:,np.newaxis,np.newaxis], (1, Nlat, Nlon)) 
    latitude_coarse = np.tile(latitude_coarse[np.newaxis,:,np.newaxis], (julian_day.shape[0], 1, Nlon))
    longtitude_coarse = np.tile(longtitude_coarse[np.newaxis,np.newaxis], (julian_day.shape[0], Nlat, 1))
    elevation_coarse = np.tile(elevation_coarse[np.newaxis], (julian_day.shape[0],1,1))
    
    # construct input features
    x = np.stack([air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  julian_day,
                  latitude_coarse,
                  longtitude_coarse, 
                  elevation_coarse], axis=-1)
    x = x.reshape(-1, x.shape[-1])
    mask = precipitation_mask_coarse.reshape(-1,1)
    value = precipitation_coarse.reshape(-1,1)
    value = np.log10(1+value) 

    # clean np.nan
    mask = np.delete(mask, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(x))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(mask))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(value))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(value))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(value))[0], axis=0)

    # train classifer
    classifer = RandomForestClassifier(n_jobs=-1)
    classifer.fit(x, mask)
    f = open('rf_classifer_{year}_era5.pickle'.format(year=year), 'wb')
    pickle.dump(classifer, f)
    f.close()

    # train regressor
    regressor = RandomForestRegressor(n_jobs=-1)
    regressor.fit(x[np.where(mask==1)[0]], value[np.where(mask==1)[0]])
    f = open('rf_regressor_{year}_era5.pickle'.format(year=year), 'wb')
    pickle.dump(regressor, f)
    f.close()
    

#-----------------------------------------------------------------------------#
# AutoML
#-----------------------------------------------------------------------------#
def train_automl(air_temperature_coarse,
             dew_temperature_coarse,
             air_pressure_coarse,
             specific_pressure_coarse,
             in_longwave_radiation_coarse,
             in_shortwave_radiation_coarse,
             wind_speed_coarse,
             precipitation_coarse,
             latitude_coarse,
             longtitude_coarse,
             elevation_coarse,
             year,
             RainThres=0.1):
    # calculate precipitation mask in coarse resolution 
    precipitation_mask_coarse = np.zeros_like(precipitation_coarse)
    precipitation_mask_coarse[precipitation_coarse>RainThres] = 1
    
    # build julian/lat/lon/dem
    Nt, Nlat, Nlon = air_temperature_coarse.shape
    julian_day = []
    for i in range(int(Nt/24)): # day
        for j in range(24): 
            julian_day.append((i+1))
    julian_day = np.array(julian_day)
    julian_day = np.tile(julian_day[:,np.newaxis,np.newaxis], (1, Nlat, Nlon)) 
    latitude_coarse = np.tile(latitude_coarse[np.newaxis,:,np.newaxis], (julian_day.shape[0], 1, Nlon))
    longtitude_coarse = np.tile(longtitude_coarse[np.newaxis,np.newaxis], (julian_day.shape[0], Nlat, 1))
    elevation_coarse = np.tile(elevation_coarse[np.newaxis], (julian_day.shape[0],1,1))
    
    # construct input features
    x = np.stack([air_temperature_coarse,
                  dew_temperature_coarse,
                  air_pressure_coarse,
                  specific_pressure_coarse,
                  in_longwave_radiation_coarse,
                  in_shortwave_radiation_coarse,
                  wind_speed_coarse,
                  julian_day,
                  latitude_coarse,
                  longtitude_coarse, 
                  elevation_coarse], axis=-1)
    x = x.reshape(-1, x.shape[-1])
    mask = precipitation_mask_coarse.reshape(-1,1)
    value = precipitation_coarse.reshape(-1,1)
    value = np.log10(1+value) 

    # clean np.nan
    mask = np.delete(mask, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(x))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(x))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(mask))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(mask))[0], axis=0)
    x = np.delete(x, np.where(np.isnan(value))[0], axis=0)
    mask = np.delete(mask, np.where(np.isnan(value))[0], axis=0)
    value = np.delete(value, np.where(np.isnan(value))[0], axis=0)
    print(x.shape, mask.shape, value.shape)
    
    # train classifer
    automl = AutoML()
    automl.fit(x, 
               mask, 
               estimator_list=['lgbm'],
               task='classification',
               metric='accuracy',
               split_ratio=0.2,
               time_budget=500,
               n_jobs=-1)
    am_output = open("automl_cls_{year}_era5.pkl".format(year=year), 'wb')
    cloudpickle.dump(automl, am_output)
    am_output.close()

    # train regressor
    am = AutoML()
    am.fit(x[np.where(mask==1)[0]],
           value[np.where(mask==1)[0]],
           estimator_list=['lgbm'],
           task='regression',
           metric='mse',
           split_ratio=0.2,
           time_budget=500,
           n_jobs=-1)
    am_output = open("automl_reg_{year}_era5.pkl".format(year=year), 'wb')
    cloudpickle.dump(am, am_output)
    am_output.close()
