# import what we need:
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pickle 
import pandas as pd 
from scipy.interpolate import interp1d
from pylab import *

import sys
import pandas.core.indexes
sys.modules['pandas.indexes'] = pandas.core.indexes
import pandas.core.base, pandas.core.indexes.frozen

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def calculate_lin_gia_rates(gias, gia_ages, ave_over, int_len):
    ## usage: calculate_gia_rates(pgia1, ages, 100, 201) find the x-year 
    ## average rates for each gia-model time series (all ages and at once)
    # gias = matrix of size [models x lat,lon x ages] gia_ages = the ages 
    # that we have predications ave_over = the length of time that we are 
    # doing a running averaging for int_len = the period at which we want 
    # the ave-over-year averages to be interpolated AVE_OVER / INT_LEN 
    # must be an integer or you will get an error import what we need:
    import numpy as np
    import matplotlib.pyplot as plt
    ages = gia_ages*1000
    if ave_over/int_len%1 != 0:
        print('error: the running average period must be divisible by the length')
    else:
        skips = np.int(ave_over/int_len)
        mat_len = np.int(round((max(ages)-min(ages))/(int_len)))
        mat0 = np.identity(mat_len, dtype = None)
        for ii in range(mat_len - skips):
            mat0[skips + ii, ii] = -1
        mat = mat0[0:-skips, 0:-skips]
        pred_ages = np.arange(max(ages), min(ages)+int_len*skips, -int_len)
        rates = np.empty([len(gias[:, 0, 0]), len(gias[0, :, 0]), mat_len-skips])
        prsl = np.empty([int_len, len(gias[:, 0, 0])])
        int_obj = {}
        for ii in range(len(gias[0, :, 0])):
            # 300 x 26 matrix: pgia[:, ii, :]
            for jj in range(len(gias[:, 0 , 0])): #(mat_len - skips):
                int_obj[ii] = interp1d(ages, gias[jj, ii, :]*1000, kind='linear')
                rates[jj, ii, :] = -np.matmul(int_obj[ii](pred_ages), mat)/ave_over
    return(rates, pred_ages)

def plot_rate_bands(ages, unc, gias):
    if param_name == 'lt':
        ax = 0
    elif param_name == 'umv':
        ax = 1
    elif param_name == 'lmv':
        ax = 2
    ncols = len(np.unique(gia_params[:, ax]))
    _, idx = np.unique(gia_params[:, ax], return_index=True)
    xi = np.arange(ncols)
    _, xii = np.unique(idx, return_index=True)
    un_params =gia_params[idx, ax]
    unique_par = un_params[xii]
    cols_temp = cmap_object(np.linspace(0., 1. ,ncols)) ## sort cols by the numb
    cols = cols_temp[np.sort(xii)]
    lbl = ()
    lbl = [str(param_name) + ' = ' + str(un_params[ll]) for ll in range(len(un_params))]
    for ii in range(len(gias[0, :, 0])) :
        plt.figure(1, figsize = (11, 8.5))
        for jj in range(len(gias)) :
            kk = np.where(unique_par == gia_params[jj,ax])
            plt.plot(ages, gias[jj,ii,:], c = np.squeeze(cols[kk]))
        fname = df['Location'][ii][0:10] + '_' + ylbl[0:4] + '_' + str(len(ages)) + '.pdf'
        tit = str(df['Location'][ii]) + ' by ' + param_name
        plt.suptitle(tit)
        plt.ylabel(ylbl)
        plt.xlabel('Age (ka)')
        for ll in range(ncols):
            p = plt.plot(0, 0, c = np.squeeze(cols[ll]), label = str(lbl[ll]))
        plt.legend()
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        plt.close()


def extract_param_from_dirs(directory):
    # create the arrays we need for output
    lt0,umv0,lmv0=numpy.empty((len(directory),),dtype='int'),numpy.empty((len(dir)))
    # loop through and extract parameters from diretcory names
    for ii in range(len(directory)):
        dirname=directory[ii]
        skip=len('output')
        startchar=dirname.find('C', skip)
        x1=dirname[skip:startchar]
        x2a=dirname[startchar+1]
        if x2a=='p':
            x2="." + dirname[startchar+2]
            x3=dirname[startchar+3:]
        else:
            x2=dirname[startchar+1]
            x3=dirname[startchar+2:]
        # organize and output
        lt0[ii],umv0[ii],lmv0[ii]=int(x1),float(x2),int(x3)
    return(lt0,umv0,lmv0)


def plot_color_by_param(cmap_object, param_name, gia_params, gias, ages, ylbl, df, bands):
    if param_name == 'lt':
        ax = 0
    elif param_name == 'umv':
        ax = 1
    elif param_name == 'lmv':
        ax = 2
    ncols = len(np.unique(gia_params[:, ax]))
    _, idx = np.unique(gia_params[:, ax], return_index=True)
    xi = np.arange(ncols)
    _, xii = np.unique(idx, return_index=True)
    un_params =gia_params[idx, ax]
    unique_par = un_params[xii]
    cols_temp = cmap_object(np.linspace(0., 1. ,ncols)) ## sort cols by the num
    cols = cols_temp[np.sort(xii)]
    lbl = ()
    lbl = [str(param_name) + ' = ' + str(un_params[ll]) for ll in range(len(un_params))]
    for ii in range(len(gias[0, :, 0])-2):
        plt.figure(1, figsize = (11, 8.5))
        for jj in range(len(gias)):
            kk = np.where(unique_par == gia_params[jj,ax])
            plt.plot(ages, gias[jj,ii,:], c = np.squeeze(cols[kk]))

        fname = df['Location'][ii][0:10] + '_' + ylbl[0:4] + '_' + str(len(ages)) + '.pdf'
        tit = str(df['Location'][ii]) + ' by ' + param_name
        plt.suptitle(tit)
        plt.ylabel(ylbl)
        plt.xlabel('Age (ka)')
        if bands == 1:
            sig2min = df['Begin last'][ii]
            sig2max = df['Begin first'][ii] 
            sig1min = df['Begin 1 last'][ii] 
            sig1max = df['Begin 1 first'][ii] 
            lsig2min = df['End last'][ii] 
            lsig2max = df['End first'][ii] 
            lsig1min = df['End 1 last'][ii] 
            lsig1max = df['End 1 first'][ii] 
            fill([sig2min, sig2max, sig2max, sig2min], [-5,-5,25,25], 'k', alpha=0.1)
            fill([sig1min, sig1max, sig1max, sig1min], [-5,-5,25,25], 'k', alpha=0.1)
            fill([lsig2min, lsig2max, lsig2max, lsig2min], [-5,-5,25,25], 'k', alpha=0.1)
            fill([lsig1min, lsig1max, lsig1max, lsig1min], [-5,-5,25,25], 'k', alpha=0.1)
        ## need legend label
        for ll in range(ncols):
            p = plt.plot(0, 0, c = np.squeeze(cols[ll]), label = str(lbl[ll]))
        plt.legend()
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        plt.close()
        
def plot_site_rates(gias, ages, ylbl, df, bands):
    lbl = ()
    for ii in range(len(gias[0, :, 0])):
        plt.figure(1, figsize = (11, 8.5))
        fill_between(ages, np.quantile(gias[:,ii,:], .025, axis = 0), np.quantile(gias[:,ii,:], .975, axis = 0), color = 'thistle', alpha = 0.5)
        fill_between(ages, np.quantile(gias[:,ii,:], .25, axis = 0), np.quantile(gias[:,ii,:], .75, axis = 0), color = 'thistle', alpha =1)
        plt.plot(ages, np.median(gias[:,ii,:], axis = 0), color = 'purple')
        fname = df['Location'][ii][0:10] + '_' + ylbl[0:4] + '_' + str(df['ID'][ii]) + '.pdf'
        tit = str(df['Location'][ii])
        plt.suptitle(tit)
        plt.ylabel(ylbl)
        plt.xlabel('Age (ka)')
        if lbl == 'Rates (m/ky)':
            plt.ylim(-3.5, 12.5)
            plt.xlim(0, 11000)
        else:
            plt.ylim(-3.5, 10)
            plt.xlim(0, 10000)
        if bands == 1:
            sig2min = df['beg4'][ii]
            sig2max = df['beg1'][ii]
            sig1min = df['beg3'][ii]
            sig1max = df['beg2'][ii] 
            fill([sig2min, sig2max, sig2max, sig2min], [-2.5,-2.5,12,12], 'k', alpha=0.1)
            fill([sig1min, sig1max, sig1max, sig1min], [-2.5,-2.5,12,12], 'k', alpha=0.1)
        if ii == 1:
            plt.show()## need legend label
        plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)
        plt.close()

