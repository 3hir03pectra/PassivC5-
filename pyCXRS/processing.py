import numpy as np
from scipy.interpolate import interp1d
import os
import pickle
import time

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def make_av_groups(kind,group,active_frames):
    av_id=()
    av_frames=()
    if kind=='by':
        # group is number of frames to average
        for i in range(active_frames.shape[0]-group+1):
            av_id=av_id+(np.arange(i,i+group,1),)
            av_frames=av_frames+(active_frames[i:i+group],)
    return av_id, av_frames

def fit_fun(x,A,B,sigma,a_fs,dx_fs):
    # No AppFun included
    # sum of gauss function with a_fs amplitudes and dx_fs shifts
    # with common broadening sigma
    # A = integral(y dx)
    # B is common shift
    y=np.zeros(x.shape[0])
    for j in range(a_fs.shape[0]):
        y=y+a_fs[j]/(sigma*np.sqrt(2*np.pi))*np.exp(- (x-B-dx_fs[j])**2/(2*sigma**2))
    return A*y

def fit_fun_app(x,A,B,sigma,a_fs,dx_fs,app_fun):
    # Convolve `fit_fun` with apparatus function
    # app_fun should be normalised: sum{app_fun} = 1
    dx=x[1]-x[0]
    y=np.zeros(x.shape[0])
    for j in range(a_fs.shape[0]):
        y=y+a_fs[j]/(sigma*np.sqrt(2*np.pi))*np.exp(- (x-B-dx_fs[j])**2/(2*sigma**2))
    y=y/np.sum(y)
    yg = np.convolve(y,app_fun,mode='same')*dx

    x0_yg=np.sum(np.multiply(x,yg))/np.sum(yg) # find max of convoluted spectra
    ygfit=interp1d(x-x0_yg+B,yg,kind='cubic',fill_value='extrapolate')
    yg_i=ygfit(x)
    return A*yg_i

def make_spectra(pxl_dim, ch_dim, id_act, pxl_max, frames_dark, pas_frames, data):
    dark = np.zeros(ch_dim)
    spectra_tot = np.zeros((pxl_dim, id_act.shape[0], ch_dim))
    spectra_pas = np.zeros((pxl_dim, id_act.shape[0], ch_dim))
    spectra_act = np.zeros((pxl_dim, id_act.shape[0], ch_dim))
    for c in range(ch_dim):
        dark[c] = np.mean(data[:pxl_max, frames_dark, c])
        spectra = data[:, :, c] - dark[c]
        spectra_tot[:, :, c] = spectra[:, id_act]
        for k in range(id_act.shape[0]):
            buffer_pas = spectra[:, id_act[k] + pas_frames]
            spectra_pas[:, k, c] = np.mean(buffer_pas, axis=1)
    spectra_act = spectra_tot - spectra_pas
    return spectra_tot, spectra_act, spectra_pas

def Gauss(x, a, sigma, x0):
    return a * np.exp(-( 2 * np.sqrt(np.log(2)) * (x - x0) / sigma) ** 2)

def beam_gauss2(x, a_CX , sigma_CX,\
                   a_E0 , sigma_E0, \
                   x0_cx, x0_E0):
    y = a_CX  * np.exp(-(x - x0_cx) ** 2 / (2 * sigma_CX ** 2)) \
      + a_E0  * np.exp(-(x - x0_E0) ** 2 / (2 * sigma_E0 ** 2))
    return y

def beam_gauss2lorentz2(x, a_CX , sigma_CX,\
                   a_E0 , sigma_E0, \
                   a_E02, sigma_E02, \
                   a_E03, sigma_E03, 
                   x0_cx, x0_E0, delta2, delta3):
    y = a_CX  * np.exp(-(x - x0_cx) ** 2 / (2 * sigma_CX ** 2)) \
      + a_E0  * np.exp(-(x - x0_E0) ** 2 / (2 * sigma_E0 ** 2)) \
      + a_E02 * sigma_E02 ** 2 /((x - x0_E0 + delta2) ** 2 + sigma_E02 ** 2) \
      + a_E03 * sigma_E03 ** 2 /((x - x0_E0 + delta3) ** 2 + sigma_E03 ** 2)
    return y

def PeakShift(L0, E0, cosTeta, mi, spectrometr):
    DeltaLambda = L0 * cosTeta / (3 * 10 ** 8) * np.sqrt(2 * E0 / mi)
    DeltaPixel = DeltaLambda / (spectrometr.D_l * spectrometr.spixel)
    return DeltaPixel

def GetCos(origin, target):
    delta_R=target[0] - origin[0]
    delta_Z=target[2] - origin[2]
    Length = np.sqrt(delta_Z ** 2 + delta_R ** 2) ## "-"
    cosT = - delta_Z / Length
    # CosT = np.cos(np.rad2deg(180 - np.rad2deg(np.arcsin((target[0] + origin[0]) / Length))))
    return cosT

def bubble_sort(arr, sub1, sub2):
    n = len(arr)
    for i in range(n-1):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1] :
               arr[j], arr[j+1]   = arr[j+1], arr[j]
               sub1[j], sub1[j+1] = sub1[j+1], sub1[j]
               sub2[j], sub2[j+1] = sub2[j+1], sub2[j]
    return arr, sub1, sub2