import numpy as np
import json
import matplotlib.pyplot as plt

class Device():
    def __init__(self,device_name,shot):
        self.device_name = device_name
        self.shot = shot
        self.target = None
        self.origin = None
        self.system = None
        self.rel_calibr = None
        self.abs_calibr = None

        # something
        return
    def load_settings(self,path):
        file_settings_name=path+'\\'+self.device_name+'_'+str(str(self.shot))+'.settings'
        with open(file_settings_name,'rb') as f:
            settings = json.load(f)
        self.target = settings['target']
        self.origin = settings['origin']
        self.system = settings['system']
        self.rel_calibr = settings['rel_calib']
        self.abs_calibr = settings['abs_calib'] #[photons/(count*cm2*s*Angstrom*st)]
        self.D_l = settings['D_l'] # [Angstrom/mm]
        self.camera = settings['camera']['name']
        self.camera_type = settings['camera']['type'] 
        self.spixel = settings['camera']['spixel'] # [mm]
        return settings

class AppFun_T10():
    def __init__(self, spectrometer:Device):
        # void for initial development. Data for T-10
        self.a=np.array([[1576.40847, 4406.412813, 11176.41884, 7677.291616, 6701.383634],
                [2651.639437, 7047.477415, 21022.19977, 10975.22509, 11817.56841],
                [4051.943039, 8452.277645, 17352.15108, 8482.324155, 8598.743246],
                [3173.522031, 10814.73998, 12850.0623, 3670.536507, 7782.23633],
                [9909.940949, 22261.19158, 30903.64791, 8974.454847, 18729.43017],
                [9666.303643, 23808.67687, 35060.21315, 13268.12095, 21395.67085],
                [5182.775353, 22584.79291, 27125.82526, 11218.59527, 20781.89141],
                [2495.761294, 13729.87023, 20053.43497, 4766.652633, 16304.04487],
                [1337.444193, 4787.584411, 12118.98415, 5239.073543, 8245.623231]])
        self.b=np.array([[193.3240267, 195.9040873, 200.6081499, 206.0116275, 210.3093398],
                        [191.8616527, 194.4910024, 199.7345037, 205.0029335, 209.1409325],
                        [192.1303748, 195.7828351, 200.9402101, 205.5528808, 209.6259812],
                        [191.9438716, 196.6127011, 202.2078738, 205.6487952, 208.89844],
                        [192.5986927, 197.0011816, 202.2164278, 206.153429, 209.3455494],
                        [193.527705, 197.9094957, 203.1072454, 207.0872129, 210.4504244],
                        [195.0066684, 199.7406849, 204.7733729, 208.3491097, 211.6063689],
                        [197.2459035, 201.3760471, 206.2939394, 210.1086199, 212.7176729],
                        [200.6649068, 203.8600832, 208.7600783, 213.3651867, 217.1955896]])
        self.c=np.array([[2.004111435, 2.783503862, 4.234895427, 3.531221938, 4.376113014],
                        [2.052899415, 3.011380229, 4.696717734, 3.352156478, 4.286690527],
                        [2.594276908, 3.65500208, 4.904448184, 3.119674375, 3.817368622],
                        [2.678879881, 4.000448512, 4.14062695, 2.334488297, 4.063333434],
                        [2.846610644, 3.352163088, 4.014153601, 2.167071418, 4.212379267],
                        [2.967839928, 3.259841833, 3.900557199, 2.164797658, 4.090172405],
                        [2.584102378, 3.726632154, 3.529651987, 2.151014647, 4.385766852],
                        [2.406730376, 3.465688595, 3.725697731, 2.151755334, 4.847735123],
                        [2.052904462, 2.992267259, 4.361669927, 2.918148891, 4.498870392]])
        pass
    def get_appfun(self,pixels):
        app_contour=np.zeros((pixels.shape[0],self.a.shape[0]))
        app_pos=np.zeros(self.a.shape[0])
        for c in range(self.a.shape[0]):
            for i in range(self.a.shape[1]):
                y=self.a[c,i]*np.exp(-(pixels-self.b[c,i])**2/self.c[c,i]**2)
                app_contour[:,c] = app_contour[:,c] + y
            app_contour[:,c]=app_contour[:,c]/np.sum(app_contour[:,c],axis=0)
            app_pos[c]=np.sum(np.multiply(pixels,app_contour[:,c]))
        return app_contour, app_pos
    def make_appfun():
        # TO DO: get apparatus function from measurements
        return    