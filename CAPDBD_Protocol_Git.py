import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sp
import glob
import os
from scipy.interpolate import interp1d
import scipy.integrate as smp

do = []
po_w = []
path_dir_i = r'/Users/scienceman/Desktop/Data/ALL0000/'
path_dir = r'/Users/scienceman/Desktop/Data/'
files_i = sorted(glob.glob(path_dir_i + '/*.CSV'))
# Frequency (1/s)
freq = 3000
# Period (s)
period = 1/ freq
# Monitor Capacitor capacitance (F)
Cm = 10.43E-9
# Actuator Voltage (V)
def channel_one():
    result = []
    for root, dirs, files in os.walk(path_dir):
        for d in files:
            if(d.endswith('CH1.CSV')):
                result.append(os.path.join(root,d))
            else:
                result.append('###############################################')
    return result
# Voltage Across the Capacitor (V)
def channel_two():
    result = []
    for root, dirs, files in os.walk(path_dir):
        for file in files:
            if(file.endswith('CH2.CSV')):
                result.append(os.path.join(root,file))
            else:
                result.append('************************************************')
    return result
######################
for i in channel_one():
    for f in channel_two():
        if i[42:44] == f[42:44]:
            print('________________________')
            print(i[42:47])
            print(f[42:47])
            # importing data from the input signal voltage
            X = T = pd.read_csv(i, delimiter= ',', header= None)
            # Time (s)
            time = np.array(X.iloc[:,3]).astype(float)
            # Start Time of the cycle (s)
            time_min = np.min(time)
            # Beginning Time of the first period (s)
            time_ap = np.around((time_min + (0*period)), decimals = 5)
            # End Time of half of the first period
            time_p = np.around((time_min + 1*period), decimals = 5)
            # End Time of the cycle
            time_max = np.max(time)
            # time grid interp
            ten = np.linspace(time_min, time_max, int(1E4))
            
            # index number of min time
            mini = np.argwhere(time == time_ap)[0][0]
            # index number of end of first period
            onep = np.argwhere(time == time_p)[0][0]
            # Length of Cycle
            cycle = (abs(time_min) + abs(time_max))/period
            # Actuator Voltage (V)
            Va = np.array(X.iloc[:,4]).astype(float)*1000
            # offset Va
            va_offset = abs(np.max(Va[:onep])) - ((abs(np.max(Va[:onep])) + abs(np.min(Va[:onep])))/2)
            # Corrected Actuator Voltage
            new_Va = Va - va_offset
            Va_interp = np.interp(ten, time, new_Va)
            # Max Actuator Voltage (V)
            max_voltage = np.max(new_Va)
            # Importing data for the voltage across the capacitor
            G = pd.read_csv(f, delimiter= ',', header= None)
            # voltage across the capacitor (V)
            Vm = ((np.array(G.iloc[:,4]).astype(float)))
            # offset Vm
            vm_offset = abs(np.max(Vm[:onep])) - ((abs(np.max(Vm[:onep])) + abs(np.min(Vm[:onep])))/2)
            # Corrected Voltage across the capacitor (V)
            new_Vm = Vm - vm_offset
            Vm_interp = np.interp(ten, time, new_Vm)
            # Smoothing (Average Box Method)
            def smooth(y, box_pts):
                box = np.ones(box_pts)/box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth
            # smoothed voltage across the capacitor (V)
            Vms = smooth(Vm_interp,150)
            Ims = Cm * np.gradient(Vms, ten)
            Qm = Cm * Vms
            pa = Va_interp * Ims
            paa = abs(pa)
            pa_area = np.trapz(paa, ten) / (cycle*period)
            po_w.append(max_voltage/1000)
            do.append(pa_area)
            '''
            print('Max Voltage', max_voltage)
            print('Power per period (W)', pa_area)
            plt.scatter(max_voltage/1000,pa_area)
            plt.ylabel('Average Power Over a Period (W)', fontsize=18)
            plt.xlabel('Voltage Across the Actuator (kV)', fontsize= 18)
            #plt.ylim(0,0.07)
            #plt.xlim(0.1, 0.9)
            fig = plt.gcf()
            plt.grid(True)
            fig.set_size_inches(10, 10)
            plt.rcParams.update({'font.size': 9})
            '''
#plt.plot(ten, Va_interp)
#plt.plot(ten, Ims)
#plt.scatter(one_g_nog, two_g_nog, s=30)
#plt.scatter(one_g_nolaser, two_g_nolaser, s=20)
#plt.scatter(one_g_wlaser, two_g_wlaser, s=20)
#plt.scatter(one_nolaser, two_nolaser, s=60)
#plt.scatter(one_wlaser, two_wlaser, s=40)
#plt.ylabel('Average Power Over a Period (W)', fontsize=18)
#plt.xlabel('Voltage Across the Actuator (kV)', fontsize= 18)
#plt.legend(['He + Gold + no laser', 'He + Gold + laser', 'He + laser'], fontsize=18)
#plt.ylim(0,0.07)
#plt.xlim(0.1, 0.9)
#fig = plt.gcf()
#plt.grid(True)
#fig.set_size_inches(10, 10)
#plt.rcParams.update({'font.size': 9})











