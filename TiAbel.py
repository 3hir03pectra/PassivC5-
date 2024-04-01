import numpy as np
import matplotlib.pyplot as plt
from pyCXRS.processing import Gauss, bubble_sort
from scipy.optimize import curve_fit, fsolve, root_scalar, bisect
from scipy.interpolate import BSpline, make_interp_spline
from Abel import MatrixL, gauss, Jmean, Jmean2ver, gaussSum, AbeL
from abel.direct import direct_transform
from scipy.interpolate import make_interp_spline
from pyCXRS.processing import make_av_groups

from pyCXRS.io_utils import spe
from pyCXRS.io_utils import log

a = 67 # cm
p = 12 # chords number
k = 12 # radius numberpi
N = 1000

folder='D:\\Spectra_T-15MD\\CXRS_Ti'

fname='plasma_5320_00845.SPE'
log_file_name = "00845.log"
filepath = folder+'\\'+fname
logpath = folder+'\\'+log_file_name
frame = 1
ch = 1
id_frame = frame - 1 # since numbering start with 0
id_ch = ch - 1       # since numbering start with 0

width_pixel = 16
pxl_max = 512

pxl0 = 330 # Where L0 approximately is
pxl_window = 15 # window to find peak position

frames_dark = np.arange(10, 15, 1)

spe_header = spe.read_spe(filepath)
data = spe_header['data']
pxl_dim = spe_header['xdim']
frame_dim = spe_header['NumFrames']
ch_dim = spe_header['ydim']

log_header = log.read_log(logpath)
frame_start = log_header['frame_start']
frame_end = log_header['frame_end']
active_frames = log_header['active_frames']
central_pixel = log_header['central_pixel']

av_id, av_frames = make_av_groups('by', len(active_frames), active_frames)
av_pas_local = np.zeros((pxl_dim, len(av_id), ch_dim))
av_pas  = []

id_act = active_frames - 1  # since numeration starts with 0
x1     = int(central_pixel - width_pixel) - 1  # since numeration starts with 0
x2     = int(central_pixel + width_pixel) - 1  # since numeration starts with 0
f1     = frame_start - 1  # since numeration starts with 0
f2     = frame_end - 1  # since numeration starts with 0
pixels = np.linspace(1, pxl_dim, pxl_dim)
frames = np.linspace(1, frame_dim, frame_dim)

dark = np.zeros(ch_dim)
spectra_pas_local = np.zeros((pxl_dim, id_act.shape[0], ch_dim))

for c in range(ch_dim):
    dark[c] = np.mean(data[:pxl_max, frames_dark, c])
    spectra = data[:, :, c] - dark[c]
    for k in range(id_act.shape[0]):
        buffer_pas = spectra[:, id_act[k]]
        spectra_pas_local[:, k, c] = buffer_pas

for i in range(len(av_id)):
        for c in range(ch_dim):
            av_pas_local[:, i, c] = np.mean(spectra_pas_local[:, av_id[i], c], axis=1)
av_pas = spectra_pas_local

pxl1 = int(pxl0 - pxl_window)
pxl2 = int(pxl0 + pxl_window)
PeakPix1 = 315
PeakPix2 = 355
PeakPixels = np.arange(PeakPix1, PeakPix2, 1)

y_prefit = np.zeros((len(PeakPixels), av_pas.shape[1], av_pas.shape[2]))
y_prefit_pas = np.zeros((len(PeakPixels), av_pas.shape[1], av_pas.shape[2]))
y_prefit_side = np.zeros((len(PeakPixels), av_pas.shape[1], av_pas.shape[2]))
Int_raw = np.zeros((av_pas.shape[1], av_pas.shape[2]))
I = np.zeros((av_pas.shape[1], av_pas.shape[2]))
x0 = np.zeros((av_pas.shape[1], av_pas.shape[2]))

pre_a = np.zeros((av_pas.shape[1], av_pas.shape[2]))
pre_sigma = np.zeros((av_pas.shape[1], av_pas.shape[2]))
pre_x0 = np.zeros((av_pas.shape[1], av_pas.shape[2]))

for c in range(av_pas.shape[2]):

    for f in range(av_pas.shape[1]):
        # Find location of passive peak
        x0[f, c] = np.sum(av_pas[pxl1:pxl2, f, c] * pixels[pxl1:pxl2]) \
                   / np.sum(av_pas[pxl1:pxl2, f, c])

        # Find preliminary location of peak located at x0
        #         a_pas , sigma_pas
        pre_p0 = [1200,    5,     ]
        pre_bounds = (0, [65e3, 100])
        #
        pre_opt, pre_cov = curve_fit(lambda x, a_pas, sigma_pas: Gauss(x, a_pas, sigma_pas, x0[f, c]), PeakPixels, \
                                               av_pas[PeakPix1:PeakPix2, f, c], p0=pre_p0, bounds=pre_bounds)
        pre_a[f, c] = pre_opt[0]
        pre_sigma[f, c] = pre_opt[1]
        pre_x0[f, c] = x0[f, c]
        y_prefit[:, f, c] = Gauss(PeakPixels, pre_a[f, c], pre_sigma[f, c], pre_x0[f, c])
        I[f,c]  = np.sum(y_prefit[:, f, c],axis=0)

        # Calculate intensities:
        # Int_abs = Int_exp   * Labsphere power * 4pi * (Exposure_calib)
        # Exposure_calib = Exposure_exp
        # Calibr_labsphere = [photons/(count*cm2*s*Angstrom*st)]
        # [counts*Angstrom/s]*[photons/(count*cm2*s*Angstrom*sr)]*[sr]*[s] = [phot/(cm2*s)]

xFull = np.array([2, 4.5, 6.8, 9, 11, 12, 13, 14, 16, 17.5, 18.8, 19.6, 21.5, 23.6, 26, 27, 28, 31, 34, 36])
jFull = np.array([1000, 2400, 3500, 4300, 8200, 10100, 11600, 12100, 12100, 11900, 11800, 11500, 11000, 10800, 11000, 11000, 9800, 4100, 2700, 2000])

X_Y_Spline = make_interp_spline(xFull, jFull)
X_ = np.linspace(xFull.min(), xFull.max(), N)
Y_ = X_Y_Spline(X_)

Jnew = Jmean2ver(Y_, X_, int(N/2))
X_new = X_ + 0.9

J = np.array([11500, 11500, 11500, 11200, 10000, 7900, 6000, 4150, 2300, 600, 80, 20, 0])
x = np.array([ 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600])

ChannelNumber = np.linspace(0, 39, 40)
ShotParametr = [0, 0, 0, 72.3, 72.3, 168, 168, -22.4, -22.4, 262.9, 262.9, -116.7, -116.7, 355, 355, -216.5, -216.5, 447.7, 447.7, 534.9, 534.9, 
                -749.2, -749.2, -676.2, -676.2, -60.8, -60.8, -598, -598, -157, -157, -514.7, -514.7, -253, -253, -433.7, -433.7, -343, -343, 0]

IntensityByChannel = np.array([1508, 1475, 1480, 1566, 1499, 1572, 1565, 1605, 1302, 1592, 1634, 1584, 1671, 1612, 1654, 1639, 1619, 1639])
RelCalibration = IntensityByChannel / max(IntensityByChannel)

ch_id = 11
frame = 9

ShotParametr_mean = np.zeros((18))
pre_sigma_mean = np.zeros((18))
I_mean = np.zeros((18))
k = 0
for c in range(3,38,2):
     ShotParametr_mean[k] = ShotParametr[c]
     pre_sigma_mean[k] = (pre_sigma[frame][c] + pre_sigma[frame][c + 1])/2
     I_mean[k] = (I[frame][c] + I[frame][c + 1])/2 / RelCalibration[k]
     k = k + 1

ShotParametr_sorted = bubble_sort(ShotParametr_mean, pre_sigma_mean, I_mean)[0]
Sigma_sorted = bubble_sort(ShotParametr_mean, pre_sigma_mean, I_mean)[1]
I_sorted = bubble_sort(ShotParametr_mean, pre_sigma_mean, I_mean)[2]

Sigma_sorted[-1] = Sigma_sorted[-1] / 1.7
Sigma_sorted[0] = Sigma_sorted[0] / 1.45
Sigma_sorted[1] = Sigma_sorted[1] / 1.15

Sigma_symmetry = Jmean(Sigma_sorted, int(ShotParametr_sorted.shape[0]/2), 0.1)
Intensity_symmetry = Jmean(I_sorted, int(ShotParametr_sorted.shape[0]/2), 10000)

X_ = np.linspace(ShotParametr_sorted.min(), ShotParametr_sorted.max(), N)

Sigma_Spline = make_interp_spline(ShotParametr_sorted, Sigma_symmetry, k=3)
Intensity_Spline = make_interp_spline(ShotParametr_sorted, Intensity_symmetry, k=3)

#Sigma_symmetry = Jmean2ver(Sigma_Spline(X_), X_, int(N/2))[1]
#Intensity_symmetry = Jmean(Intensity_Spline(X_), int(N/2), 10000)

Sigma = Sigma_Spline(X_)
Intensity_Abel = direct_transform(Intensity_Spline(X_)[int(N/2):N], \
                                  dr = a / int(N/2), direction="inverse", correction=True)



Sigma_ras = np.zeros(int(N/2) - 1)
Sigma_ras_part = np.zeros(int(N/2) - 1)

x_gauss = np.linspace(- ShotParametr_sorted.shape[0], ShotParametr_sorted.shape[0], 2 * ShotParametr_sorted.shape[0] + 1)

Gauss_signal = np.zeros((int(N/2) - 1, len(x_gauss)))
Gauss_signal_ras = np.zeros((int(N/2) - 1, len(x_gauss)))
Gauss_summary = np.zeros((int(N/2) - 1, len(x_gauss)))
Gauss_part = np.zeros((int(N/2) - 1, len(x_gauss)))
Gauss_ras = np.zeros((int(N/2) - 1, len(x_gauss)))

Sigma_ras[:] = np.flip(Sigma[int(N/2) + 1:N])
Sigma_ras_part[0] = Sigma_ras[0]
I_ras_part = np.zeros((int(N/2) - 1))

Matrix_L = np.matrix.transpose(MatrixL(int(N/2) - 1, a, int(N/2) - 1))

Intensity_ras = np.flip(Intensity_Spline(X_)[int(N/2) + 1:N])
Intensity_Abel = np.flip(Intensity_Abel[0: int(N/2) - 1])

func = lambda self, sigma, I: gauss(x_gauss, I /sigma, sigma)

K = int(N/2) - 1

for i in range(1, K):
     
     A_gauss_previous = np.zeros(i)
     sigma_gauss_previous = np.zeros(i)

     ML = Matrix_L[i, 0: i]
     IA = Intensity_Abel[0: i]
     SRP = Sigma_ras_part[0: i]

     A_gauss_previous = Matrix_L[i, 0: i] * Intensity_Abel[0: i] / Sigma_ras_part[0: i]

     sigma_gauss_previous = Sigma_ras_part[0: i]

     Gauss_summary[i] =  gauss(x_gauss, sum(Matrix_L[i, :]) *  Intensity_ras[i] / Sigma_ras[i], Sigma_ras[i])
     Gauss_part[i] = gaussSum(x_gauss, A_gauss_previous, sigma_gauss_previous, i)
     Gauss_signal[i] = (Gauss_summary[i] - Gauss_part[i])/ Matrix_L[i, i]
     pre_opt, pre_cov = curve_fit(func, x_gauss, Gauss_signal[i], p0=[Sigma_ras_part[i - 1], A_gauss_previous[i - 1]])
     Sigma_ras_part[i] = pre_opt[0]
     I_ras_part[i] = pre_opt[1]

     Gauss_ras[i] = gauss(x_gauss, I_ras_part[i] / Sigma_ras_part[i], Sigma_ras_part[i])

fig0 = plt.figure()
plt.plot(x_gauss, Gauss_signal[1], label=r"Разность")
plt.plot(x_gauss, Gauss_ras[1], label=r"Восстановленный")
plt.title("Сравнение сигналов")
plt.legend(fontsize = 10, framealpha=1.0)
plt.ylabel('Интенсивность, отн.ед.',)
plt.xlabel('Номер пикселя',)

T_ion = 1.7 * 10 ** -2 * 12 * (Sigma_ras_part * 1.663) ** 2

fig1 = plt.figure()

plt.plot(pixels, av_pas[:, frame, ch_id] - 50, label=r"Эксперимент")
plt.plot(PeakPixels, y_prefit[:, frame, ch_id], label=r"Аппроксимация")
plt.title("Аппроксимация экспериментальных данных")
plt.legend(fontsize = 10, framealpha=1.0)
plt.ylabel('Интенсивность, отн.ед.',)
plt.xlabel('Номер пикселя',)

fig2 = plt.figure()
plt.plot(ShotParametr_sorted, Sigma_sorted, 'o', label=r"Эксперимент")
plt.plot(ShotParametr_sorted, Sigma_symmetry, label=r"Симметризация")
plt.plot(X_, Sigma_Spline(X_), label=r"Аппроксимация")
plt.title("Симметризация данных по уширению")
plt.legend(fontsize = 10, framealpha=1.0)
plt.ylabel('Уширение, пиксель',)
plt.xlabel('Координата, см',)

fig3 = plt.figure()
plt.plot(ShotParametr_sorted, I_sorted , 'o', label=r"Эксперимент")
plt.plot(ShotParametr_sorted, Intensity_symmetry, label=r"Симметризация")
plt.plot(X_, Intensity_Spline(X_), label=r"Аппроксимация")
plt.title("Симметризация данных по яркости")
plt.legend(fontsize = 10, framealpha=1.0)
plt.ylabel('Яркость, отн.ед.',)
plt.xlabel('Координата, см',)

fig4 = plt.figure()
plt.plot(X_[int(N/2) + 1:N], np.flip(Intensity_Abel[0: int(N/2) - 1]))
plt.title("Абелизация профиля яркости")
plt.ylabel('Яркость, отн.ед.',)
plt.xlabel('Координата, см',)

fig5 = plt.figure()
plt.plot(X_[int(N/2) + 1:N], Sigma[int(N/2) + 1:N], label=r"Хордовое")
plt.plot(X_[int(N/2) + 1:N], np.flip(Sigma_ras_part), label=r"Радиальное")
plt.title("Распределение уширений профиля Гаусса")
plt.legend(fontsize = 10, framealpha=1.0)
plt.ylabel('Уширение, отн.ед.',)
plt.xlabel('Координата, см',)

fig6 = plt.figure()
plt.plot(X_[int(N/2) + 1:N], np.flip(T_ion), label=r"")
plt.title("Профиль температуры C5+")
plt.legend(fontsize = 10, framealpha=1.0)
plt.ylabel('Температура, эВ',)
plt.xlabel('Координата, см',)

fig7 = plt.figure()
plt.plot(x_gauss, Gauss_signal[10], label=r"")
plt.plot(x_gauss, Gauss_signal_ras[10], label=r"")
plt.title("Профиль температуры C6+")
plt.legend(fontsize = 10, framealpha=1.0)
plt.ylabel('Температура, эВ',)
plt.xlabel('Координата, см',)

plt.show()