himport pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.signal as sp
import glob
import os
from scipy.interpolate import interp1d


path_dir_i = r'/Users/scienceman/Desktop/Data/ALL0000/'
path_dir = r'/Users/scienceman/Desktop/Data/'
files_i = sorted(glob.glob(path_dir_i + '/*.CSV'))
max_p = []
do = []
num = []
pow_area = []
# Frequency (1/s)
freq = 1E4
# Period (s)
period = 1/ freq

def channel_one():
    result = []
    for root, dirs, files in os.walk(path_dir):
        for d in files:
            if(d.endswith('CH1.CSV')):
                result.append(os.path.join(root,d))
            else:
                result.append('###############################################')
    return result

def channel_two():
    result = []
    for root, dirs, files in os.walk(path_dir):
        for file in files:
            if(file.endswith('CH2.CSV')):
                result.append(os.path.join(root,file))
            else:
                result.append('************************************************')
    return result

for i in channel_one():
    for f in channel_two():
        if i[42:44] == f[42:44]:
            print('________________________')
            print(i[42:47])
            print(f[42:47])
            X = T = pd.read_csv(i, delimiter= ',', header= None)
            time = np.array(X.iloc[:,3]).astype(float)
            low_bound = np.around((np.min(time)+ period), decimals = 6)
            high_bound = np.max(time) - period
            index_l = np.argwhere(time == low_bound)[0][0]
            index_h = np.argwhere(time == high_bound)[0][0]
            voltage = np.array(X.iloc[:,4]).astype(float)*1000
            G = pd.read_csv(f, delimiter= ',', header= None)
            ttime = np.array(G.iloc[:,3]).astype(float)
            vch2 = (np.array(G.iloc[:,4]).astype(float))
            current = ((np.array(G.iloc[:,4]).astype(float))/60)
            voltage = np.array(X.iloc[:,4]).astype(float)*1000
            def smooth(y, box_pts):
                box = np.ones(box_pts)/box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth
            current_s= smooth(current, 100)
            # Current-Voltages signals are not symmetric
            index_p = np.argwhere(time == (-0.00015))[0][0] 
            #print(voltage[:500])
            v_offset = abs(np.max(voltage[:index_p])) - ((abs(np.max(voltage[:index_p])) + abs(np.min(voltage[:index_p])))/2)
            c_offset = abs(np.max(current_s[:index_p])) - ((abs(np.max(current_s[:index_p])) + abs(np.min(current_s[:index_p])))/2)
            # symmetric voltage profile
            may_voltage = voltage - v_offset
            # symmetric current profile
            new_current = current - c_offset
            area_current = np.trapz(abs(new_current), ttime)
            # corrected new voltage
            new_voltage = may_voltage - ((new_current)*60)
            # symmetric voltage profile
            new_voltage = voltage - v_offset
            # max voltage
            max_voltage = np.max(new_voltage)
            # Max current
            max_current = np.max(new_current)
            # normalized voltage profile
            voltage_norm = new_voltage / max_voltage
            # normalized current profile
            current_norm = new_current / max_current
            # Total Time (s)
            total_time = (abs(np.max(time)) + abs(np.min(time)))
            # Total number of periods 
       
            total_period = total_time / period
            #print(total_period)
            #voltage_fit = np.sin(2*np.pi*freq*time)
            #phase = 0
            #current_fit = np.sin(2*np.pi*freq*time - phase * (np.pi))
            # time for three periods
            time_int = time[index_l:index_h]
            # Power
            power = new_current * new_voltage
            power_int = new_current[index_l:index_h] * new_voltage[index_l:index_h]
            power_area = np.trapz(abs(power[index_l:index_h]), time[index_l:index_h])/(total_time - 2*period)
            pow_area.append(power_area)
            num.append(i[42:44])
            print('Maximum Voltage (kV)', max_voltage/1000)
            print('Maximum Current (A)', max_current)
            print('Integrated Power (J)', power_area)
            #plt.plot(time, power_area)

            max_p.append(max_voltage/1000)
            do.append(power_area)
            '''
            #plt.scatter(max_voltage/1000, power_area)
            #plt.yscale('log')
            plt.plot(time, new_current)
            #plt.plot(time_int, (power_int))
            #plt.plot(time, current_norm)
            plt.xlabel('Max Voltage (kV)', fontsize=18)
            #plt.xticks(np.arange(0, 10, step=1))
            #plt.legend()
            plt.ylabel('Power Integral / (3x periods) (W)', fontsize=18)
            fig = plt.gcf()
            plt.grid(True)
            fig.set_size_inches(12, 12)
            plt.rcParams.update({'font.size': 9})
            '''