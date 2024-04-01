import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from pyCXRS.io_utils.spe import read_spe
from pyCXRS.io_utils.log import read_log
from pyCXRS.processing import make_av_groups, make_spectra, Gauss, beam_gauss2lorentz2, beam_gauss2, fit_fun_app, GetCos
from pyCXRS.device_settings import Device, AppFun_T10


### Data input

measurments = ['CVI']
SpecDevice = ['HES3']
# spe_file = ["F:\\Spectra_T-15MD\\CXRS_Ti\\HES3_data\\t10_plasma_hes3_71688.SPE",
#                 "F:\\Spectra_T-15MD\\CXRS_Ti\\HES1_data\\t10_plasma_hes1_71688.SPE"]
# log_file = ["F:\\Spectra_T-15MD\\CXRS_Ti\\HES3_data\\log\\71688.log",
#                 "F:\\Spectra_T-15MD\\CXRS_Ti\\HES3_data\\log\\71688.log"]
# settings_path = ["F:\\Spectra_T-15MD\\CXRS_Ti\\HES3_data\\settings",
#                      "F:\\Spectra_T-15MD\\CXRS_Ti\\HES1_data\\settings"]

path = "C:\\Users\\Artem\\Desktop\\Spectra_T-15MD\\CXRS_Ti\\"
shot = 71688
spe_file = [path+"plasma_5320_00860.SPE"]
log_file = [path+"HES3_data\\log\\71688.log"]
settings_path = [path+"HES3_data\\settings"]

fine_dump = ".\\pyCXRS\\atomic_data\\fine_dump"
fine_excel = "Fine structure ACT_scans.xlsx"

frames_dark = np.arange(10, 15, 1)
pas_frames = np.array([-1]) #[-2, -1, 1, 2]

width_pixel = 50
pxl_max = 400

id_hes = 1
L0 = 6561  # Angstrom
E0 = 60 * 1.6 * 10 ** -19 * 10 ** 3  # J
mi = 1.674 * 10 ** -27  # kg

pxl0 = 277 # Where L0 approximately is
pxl_window = 50 # window to find peak position

raw_data = []
spectra_tot, spectra_act, spectra_pas = [], [], []
av_pas, av_act,  = [], []

for i, keys in enumerate(measurments):
    ### read log file

    log_header = read_log(log_file[i])
    frame_start = log_header['frame_start']
    frame_end = log_header['frame_end']
    active_frames = log_header['active_frames']
    central_pixel = log_header['central_pixel']

    ### read SPE

    spe_header = read_spe(spe_file[i])
    data = spe_header['data']
    pxl_dim = spe_header['xdim']
    frame_dim = spe_header['NumFrames']
    ch_dim = spe_header['ydim']
    
    ### data pre-processing

    id_act = active_frames - 1  # since numeration starts with 0
    x1 = int(central_pixel - width_pixel) - 1  # since numeration starts with 0
    x2 = int(central_pixel + width_pixel) - 1  # since numeration starts with 0
    f1 = frame_start - 1  # since numeration starts with 0
    f2 = frame_end - 1  # since numeration starts with 0
    pixels = np.linspace(1, pxl_dim, pxl_dim)
    frames = np.linspace(1, frame_dim, frame_dim)
    spectra_tot_local, spectra_act_local, spectra_pas_local = make_spectra(pxl_dim, ch_dim, id_act, pxl_max, frames_dark,
                                                         pas_frames, data)
    spectra_pas.append(spectra_pas_local)
    spectra_act.append(spectra_act_local)
    spectra_tot.append(spectra_tot_local)
    raw_data.append(data)

    # Load spectrometer settings and apparatus function

    spectrometer = Device(SpecDevice[i], shot)
    spectrometer.load_settings(settings_path[i])
    app_fun_class = AppFun_T10(spectrometer)
    app_fun = np.zeros((pxl_dim, ch_dim))
    app_contour, app_pos = app_fun_class.get_appfun(pixels)

    av_id, av_frames = make_av_groups('by', len(active_frames), active_frames)
    av_act_local = np.zeros((pxl_dim, len(av_id), ch_dim))
    av_pas_local = np.zeros((pxl_dim, len(av_id), ch_dim))

    for i in range(len(av_id)):
        print(f'\033[0;32;40m Start {i + 1:d} of {len(av_id):d} time steps: \033[0;0m')
        for c in range(ch_dim):
            av_pas_local[:, i, c] = np.mean(spectra_pas_local[:, av_id[i], c], axis=1)
            av_act_local[:, i, c] = np.mean(spectra_act_local[:, av_id[i], c], axis=1)
    av_pas.append(av_pas_local)
    av_act.append(av_act_local)


pxl1 = int(pxl0-pxl_window)
pxl2 = int(pxl0+pxl_window)

x0_pas       = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))

y_prefit     = np.zeros((pxl_dim, av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
y_prefit_cx  = np.zeros((pxl_dim, av_act[id_hes].shape[1], av_act[id_hes].shape[2]))

pre_a_CX     = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
pre_sigma_CX = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
pre_x0_cx    = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))

y_fit     = np.zeros((pxl_dim, av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
y_fit_cx  = np.zeros((pxl_dim, av_act[id_hes].shape[1], av_act[id_hes].shape[2]))

a_CX      = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
sigma_CX  = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
x0_cx     = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))

Int_raw  = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
Int_fit  = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))
Int_cx   = np.zeros((av_act[id_hes].shape[1], av_act[id_hes].shape[2]))

Sigma_HES = 3 # pixels

for c in range(av_act[id_hes].shape[2]):
    I = np.zeros((av_act[id_hes].shape[1]))
    Is = np.zeros((av_act[id_hes].shape[1]))
    cosTeta = GetCos(spectrometer.origin[c], spectrometer.target[c])

    for f in range(av_act[id_hes].shape[1]):
        if spectrometer.system[c]!='c':
            # Find location of passive H-alpha
            x0_pas[f,c] = np.sum(av_pas[id_hes][pxl1:pxl2, f, c]*pixels[pxl1:pxl2])\
                           /np.sum(av_pas[id_hes][pxl1:pxl2, f, c])

            # Find preliminary location of E0 fraction assuming active CX located at x0_pas
            #             a_CX ,    sigma_CX,    x0_E0
            pre_p0     = [ 1000,          50,     177]
            pre_bounds = (0,[65e3,       200,     512])
            #                                      0       1     
            pre_opt, pre_cov = curve_fit(lambda x,a_CX,sigma_CX: \
                                         Gauss(x, a_CX,sigma_CX, x0_cx=x0_pas[f,c]),\
                                            pixels, av_act[id_hes][:, f, c],\
                                            p0=pre_p0,bounds=pre_bounds\
                                        )
            pre_a_CX[f,c]     = pre_opt[0]
            pre_sigma_CX[f,c] = pre_opt[1]
            pre_x0_cx[f,c]    = x0_pas[f,c]
            y_prefit_cx[:, f, c]  = Gauss(pixels, pre_a_CX[f,c], pre_x0_cx[f,c], pre_sigma_CX[f,c])
  
            # Construct first guess and bounds 
            p0 = [ pre_a_CX[f,c],      # a_CX
                   pre_sigma_CX[f,c],  # sigma_CX
                   pre_x0_cx[f,c],     # x0_cx  
                 ]  

            lower_bound = [ 0.5*pre_a_CX[f,c],      # a_CX
                            0.5*pre_sigma_CX[f,c],  # sigma_CX
                            0.9*pre_x0_cx[f,c],     # x0_cx  
                          ]
            upper_bound = [ 1.5*pre_a_CX[f,c],      # a_CX
                            1.5*pre_sigma_CX[f,c],  # sigma_CX
                            1.1*pre_x0_cx[f,c],     # x0_cx 
                          ]
            bounds = (lower_bound,upper_bound)
            #                                  0        1      2          
            popt, pcov = curve_fit(lambda x, a_CX , sigma_CX,x0_cx : fit_fun_app(x, a_CX , sigma_CX, x0_cx), pixels,av_act[id_hes][:, f, c],\
                                    p0=p0,bounds = bounds, \
                                    verbose = 0)
            a_CX[f,c]      = popt[0]
            sigma_CX[f,c]  = popt[1]
            x0_cx[f,c]     = popt[2]

            y_fit[:, f, c] = fit_fun_app(pixels, *popt)

            y_fit_cx[:, f, c]  = Gauss(pixels, a_CX[f,c], x0_cx[f,c], sigma_CX[f,c])
            
            # Calculate intensities:
            # Int_abs = Int_exp   * Labsphere power * 4pi * (Exposure_calib)
            # Exposure_calib = Exposure_exp
            # Calibr_labsphere = [photons/(count*cm2*s*Angstrom*st)]
            # [counts*Angstrom/s]*[photons/(count*cm2*s*Angstrom*sr)]*[sr]*[s] = [phot/(cm2*s)]

            Int_raw[f,c]  = np.sum(av_act[id_hes][:pxl_max,f,c],axis=0) # [counts]
                                                         # *pixel_size*D_l/Exposure_exp  [counts*Angstrom/s] 
                                                         # *Calibr_labsphere*4*pi*Exposure*Kamp;
            
            Int_fit[f,c]  = np.sum(y_fit[:, f, c],axis=0)
            Int_cx[f,c]   = np.sum(y_fit_cx[:, f, c],axis=0)

            print('Integral deviation c={0:d},{1:s} : {2:2.3f}'.format(c,spectrometer.system[c],np.abs(Int_raw[f,c]-Int_fit[f,c])/Int_raw[f,c]))


### Plots
ncols = int(np.floor(np.sqrt(ch_dim)))

fig_pas, ax_pas = plt.subplots(ncols, ncols)
for f in range(len(av_id)):
    for c in range(ch_dim):
        m = c // ncols
        n = c % ncols
        ax_pas[m, n].plot(pixels, av_pas[id_hes][:, f, c], 'r-', linewidth=1, label='av. PAS')
        ax_pas[m, n].plot([x0_pas[f,c],x0_pas[f,c]],[0, av_pas[id_hes][int(x0_pas[f,c]), f, c]], 'k--', linewidth=1)

        ax_pas[m, n].plot([pxl1, pxl1],[0, 200],'k--')    
        ax_pas[m, n].plot([pxl2, pxl2],[0, 200],'k--')    

        ax_pas[m, n].legend()
        ax_pas[m, n].set_title(spectrometer.system[c])
        ax_pas[m, n].set_xlabel('pixels')
        ax_pas[m, n].set_ylabel('Int, counts/pixel')

    plt.show()

fig_av, ax_av = plt.subplots(ncols, ncols)
for f in range(len(av_id)):
    for c in range(ch_dim):
        m = c // ncols
        n = c % ncols
        ax_av[m, n].plot(pixels, av_act[id_hes][:, f, c], 'r-', linewidth=1, label='av. ACT')
        ax_av[m, n].plot(pixels, y_fit[:, f, c], 'k-', linewidth=1, label='total')  ## draw fittings

        ax_av[m, n].plot(pixels, y_prefit[:, f, c], 'k--', linewidth=0.5, label='pre-total')  ## draw pre_fittings
        ax_av[m, n].plot(pixels, y_prefit_cx[:, f, c], 'b--', linewidth=0.5, label='pre-CX')
        ax_av[m, n].plot(pixels, y_prefit_E0[:, f, c], 'g--', linewidth=0.5, label='pre-E0')

        ax_av[m, n].plot(pixels, y_fit_cx[:, f, c], 'b-', linewidth=1, label='CX')
        ax_av[m, n].plot(pixels, y_fit_E0[:, f, c], 'g-', linewidth=0.5, label='E0')
        ax_av[m, n].plot(pixels, y_fit_E02[:, f, c], 'm-', linewidth=0.5, label='E0/2')
        ax_av[m, n].plot(pixels, y_fit_E03[:, f, c], 'c-', linewidth=0.5, label='E0/3')

        ax_av[m, n].plot([x0_E0[f,c], x0_E0[f,c]],   [0, 200], 'g--')
        ax_av[m, n].plot([x0_E02[f,c], x0_E02[f,c]], [0, 100], 'm--')
        ax_av[m, n].plot([x0_E03[f,c], x0_E03[f,c]],   [0, 50], 'c--')

        ax_av[m, n].legend()
        ax_av[m, n].set_title(spectrometer.system[c])
        ax_av[m, n].set_xlabel('pixels')
        ax_av[m, n].set_ylabel('Int, counts/pixel')

    plt.show()

fig_frame, ax_frame = plt.subplots(ncols, ncols)
for c in range(ch_dim):
    m = c // ncols
    n = c % ncols

    Y = np.sum(raw_data[id_hes][:,:,c],axis=0)
    Y = Y/np.max(Y)

    ax_frame[m, n].plot(frames, Y, 'r-', linewidth=1)
    ax_frame[m, n].plot(frames[id_act], Y[id_act], 'kx', linewidth=1)

    ax_frame[m, n].legend()
    ax_frame[m, n].set_title(spectrometer.system[c])
    ax_frame[m, n].set_xlabel('frames')
    ax_frame[m, n].set_ylabel('Sum(Int) over pixels, counts')



plt.show()