import numpy as np
import pandas as pd
import pickle
from scipy.interpolate import interp1d

def count_scans(names):
    # Counts how many times in $names$ appears <pattern> and return counters for
    # <pattern>='Ne', 'Ni', 'Te', 'Ti', 'Zef', or 'Bt'  
    N_ne=0
    N_ni=0
    N_te=0
    N_ti=0
    N_zef=0
    N_bt=0
    for key in names:
        if (key.find('Ne')!=-1):
            N_ne+=1
        if (key.find('Ni')!=-1):
            N_ni+=1            
        if (key.find('Te')!=-1):
            N_te+=1            
        if (key.find('Ti')!=-1):
            N_ti+=1            
        if (key.find('Zeff')!=-1):
            N_zef+=1            
        if (key.find('Bt')!=-1):
            N_bt+=1            
    return N_ne, N_ni, N_te, N_ti, N_zef, N_bt
def read_scan(prefix,data,Ns_x):
    # Read DataFrame $data$ and 
    # return two Numpy arrays with scan grid $x_scan$ and values $Int_x_scan$
    # $prefix$ defines the pattern of name in DataFrame $data$
    # $Ns_x$ is the counter for number of points in each scan
    
    x_scan=np.zeros((Ns_x))
    Int_x_scan=np.zeros((data[prefix].shape[0]-1,Ns_x))
    for i in range(Ns_x):
        if i==0:
            key=prefix
        else:
            key=prefix+'.'+str(i)
        x_scan[i]=data[key][0]
        Int_x_scan[:,i]=data[key][1:].to_numpy()
    return x_scan, Int_x_scan

def excel2dict(fine_excel,file_to_write=None):
    # Converts data from excel file with data produced by ADAS306 to python Dictionary
    fine_data=pd.read_excel(fine_excel,sheet_name='Scan')
    ref_data=pd.read_excel(fine_excel,sheet_name='Ref')
    wavelength=fine_data['wavelength'][1:].to_numpy()

    Ns_ne, Ns_ni, Ns_te, Ns_ti, Ns_zef, Ns_bt=count_scans(fine_data.columns)
    ne_scan, Int_ne_scan = read_scan('Scan Ne',fine_data,Ns_ne)
    ni_scan, Int_ni_scan = read_scan('Scan Ni',fine_data,Ns_ni)
    Te_scan, Int_Te_scan = read_scan('Scan Te',fine_data,Ns_te)
    Ti_scan, Int_Ti_scan = read_scan('Scan Ti',fine_data,Ns_ti)
    Zef_scan, Int_zef_scan = read_scan('Scan Zeff',fine_data,Ns_zef)
    Bt_scan, Int_bt_scan = read_scan('Scan Bt',fine_data,Ns_bt)

    Int_ref=ref_data['Int'].to_numpy()
    ne_ref=ref_data['ne'][0]
    ni_ref=ref_data['ni'][0]
    Te_ref=ref_data['Te'][0]
    Ti_ref=ref_data['Ti'][0]
    Zef_ref=ref_data['Zeff'][0]
    Bt_ref=ref_data['Bt'][0]


    fine_structure_data={'wavelength':wavelength,
        'ne_scan':ne_scan, 'Int_ne_scan':Int_ne_scan,
        'ni_scan':ni_scan, 'Int_ni_scan':Int_ni_scan,
        'Te_scan':Te_scan, 'Int_Te_scan':Int_Te_scan,
        'Ti_scan':Ti_scan, 'Int_Ti_scan':Int_Ti_scan,
        'Zef_scan':Zef_scan, 'Int_zef_scan':Int_zef_scan,
        'Bt_scan':Bt_scan, 'Int_bt_scan':Int_bt_scan,

        'ne_ref':ne_ref, 'ni_ref':ni_ref, 'Te_ref':Te_ref,
        'Ti_ref':Ti_ref, 'Zef_ref':Zef_ref, 'Bt_ref':Bt_ref,
        'Int_ref':Int_ref}
    if file_to_write:
        print("\033[32m fine_structure.exel2dict:\033[0m Fine structure data saved to: ")
        print(file_to_write)
        with open(file_to_write,'wb') as f:
            pickle.dump(fine_structure_data,f)
    return fine_structure_data

def make_fine_structure(fs_data,ne=None,ni=None,Te=None,Ti=None,Zeff=None,Bt=None):
    # For given plasma parameters return relative intensities of fine structure lines
    # Scan over parameters should be provided in $fs_data$
    # Each scan define how line intensity changes in comparison with reference:
    #       I_new = I_ref * (I_scan_1 / I_ref) * (I_scan_2 / I_ref) * ....
    # [ne] = cm-3
    # [ni] = cm-3
    # [Te] = eV
    # [Ti] = eV
    # [Ze] = -
    # [Bt] = T

    ne=ne or fs_data['ne_ref']
    ni=ni or fs_data['ni_ref']
    Te=Te or fs_data['Te_ref']
    Ti=Ti or fs_data['Ti_ref']
    Zeff=Zeff or fs_data['Zef_ref']
    Bt=Bt or fs_data['Bt_ref']
    wlen=fs_data['wavelength']
    I_ref=fs_data['Int_ref']

    Int_ne=np.zeros(I_ref.shape)
    Int_ni=np.zeros(I_ref.shape)
    Int_Te=np.zeros(I_ref.shape)
    Int_Ti=np.zeros(I_ref.shape)
    Int_Zef=np.zeros(I_ref.shape)
    Int_Bt=np.zeros(I_ref.shape)
    Rel_Int=np.zeros(I_ref.shape)
    for i in range(wlen.shape[0]):
        buffer_fit=interp1d(fs_data['ne_scan'],fs_data['Int_ne_scan'][i,:],kind='cubic',fill_value='extrapolate')
        Int_ne[i]=buffer_fit(ne)
        buffer_fit=interp1d(fs_data['ni_scan'],fs_data['Int_ni_scan'][i,:],kind='cubic',fill_value='extrapolate')
        Int_ni[i]=buffer_fit(ni)
        buffer_fit=interp1d(fs_data['Te_scan'],fs_data['Int_Te_scan'][i,:],kind='cubic',fill_value='extrapolate')
        Int_Te[i]=buffer_fit(Te)
        buffer_fit=interp1d(fs_data['Ti_scan'],fs_data['Int_Ti_scan'][i,:],kind='cubic',fill_value='extrapolate')
        Int_Ti[i]=buffer_fit(Ti)
        # print('---', Ti,'---')
        # print(fs_data['Ti_scan'])
        buffer_fit=interp1d(fs_data['Zef_scan'],fs_data['Int_zef_scan'][i,:],kind='cubic',fill_value='extrapolate')
        Int_Zef[i]=buffer_fit(Zeff)
        buffer_fit=interp1d(fs_data['Bt_scan'],fs_data['Int_bt_scan'][i,:],kind='cubic',fill_value='extrapolate')
        Int_Bt[i]=buffer_fit(Bt)
        Rel_Int[i]=I_ref[i]*(Int_ne[i]/I_ref[i])*(Int_ni[i]/I_ref[i]) \
                        *(Int_Te[i]/I_ref[i])*(Int_Ti[i]/I_ref[i]) \
                        *(Int_Zef[i]/I_ref[i])*(Int_Bt[i]/I_ref[i])
    Rel_Int=Rel_Int/np.sum(Rel_Int)            
    return Rel_Int
def apply_zeeman(wlen,Rel_Int,Bt,theta):
    # Split fine structure {wlen,Rel_Int} by Zeeman effect
    # Formula is taken from R.C. Isler (1994) PPCF-36-171; DOI 10.1088/0741-3335/36/2/001]
    #
    # [Bt] = T
    # [theta] = deg
    theta=np.deg2rad(theta)
    wlen_zeeman=np.zeros(3*wlen.shape[0])
    Rel_Int_zeeman=np.zeros(3*Rel_Int.shape[0])
    k=np.arange(0,wlen.shape[0],1)
    kL=3*(k)   # indexes of left sigma components
    k0=3*(k)+1 # indexes of central pi components
    kR=3*(k)+2 # indexes of right sigma components

    wlen_zeeman[kL]=wlen-4.699e-9*wlen**2*Bt; # wavelength left sigma components
    wlen_zeeman[k0]=wlen-0*Bt;                # wavelength central pi components 
    wlen_zeeman[kR]=wlen+4.699e-9*wlen**2*Bt; # wavelength right sigma components 

    Rel_Int_zeeman[kL]=0.5*(1+np.cos(theta)**2)/2*Rel_Int   # intensities of left sigma components
    Rel_Int_zeeman[k0]=0.5*np.sin(theta)**2*Rel_Int         # intensities of central pi components
    Rel_Int_zeeman[kR]=0.5*(1+np.cos(theta)**2)/2*Rel_Int   # intensities of right sigma component
    return wlen_zeeman, Rel_Int_zeeman