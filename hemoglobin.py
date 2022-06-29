import os
from pandas.core.frame import DataFrame
import pywt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from scipy.signal import savgol_filter, find_peaks
import math

def get_bloodVol_data(data):
  
  IR_data = data['IR'] * -1
  RED_data = data['RED'] * -1
  
  d = []
  for i in range(len(IR_data)):
    row=[]
    row.append(IR_data[i])
    row.append(RED_data[i])
    d.append(row)
  
  return d
  
def get_holes(signal):
  neg_signal = [x*-1 for x in signal]
  holes, props = find_peaks(neg_signal, distance=100)
  return holes

def get_derivative(data):
  
  dx = 1

  dy_IR = np.diff(data['IR'])/dx
  dy_RED = np.diff(data['RED'])/dx

  d = []
  for i in range(len(dy_IR)):
    row=[]
    row.append(dy_IR[i])
    row.append(dy_RED[i])
    d.append(row)
  
  return d

def feature_extractor2(signal):
  
  neg_signal = [x*-1 for x in signal]

  syst_peaks, props = find_peaks(signal, distance=100)
  holes, props = find_peaks(neg_signal, distance=100)
  all_peaks, props = find_peaks(signal)
  all_notches, props = find_peaks(neg_signal)


  set_syst_peaks = set(syst_peaks)
  set_all_peaks = set(all_peaks)
  set_all_notches = set(all_notches)
  set_holes = set(holes)
  
  waves = []
  temp = []
  f = 0
  p = 0

  for i in range(len(signal)):

    if i in set_holes:
      temp.append(i)
      f = 1

      if len(temp) > 1:
        waves.append(temp)
        temp = [i]
    
    elif i in set_all_peaks and f and len(temp) < 4:
      temp.append(i)
      
    elif i in set_all_notches and f and len(temp) < 4:
      temp.append(i)
  # print(waves)
  return waves


def select_good_waves(waves):
  good_waves = []

  for x in waves:
    if len(x) == 5:
      good_waves.append(x)

  return good_waves

def get_wave_times(waves):
  time = []
  for x in waves:
    temp = []
    for i in range(1,len(x)):
      temp.append(x[i]-x[0])
    time.append(temp)
  #for x in time:
  #  print(x)
  return time

def feature_extractor2_1d(signal, holes):

  set_holes = set()
  for x in holes:
    set_holes.add(x)
    
  neg_signal = [x*-1 for x in signal]

  syst_peaks, props = find_peaks(signal, distance=100)
  all_peaks, props = find_peaks(signal)
  all_notches, props = find_peaks(neg_signal)

  
  set_syst_peaks = set(syst_peaks)
  set_all_peaks = set(all_peaks)
  set_all_notches = set(all_notches)
  
  waves = []
  temp = []
  f = 0
  p = 0
  # print("holes: ",set_holes)
  # print("signal: ",signal)
  for i in range(len(signal)):

    if i in set_holes:
      f = 1
      if len(temp) == 0:
        temp.append(i)
      if len(temp) > 1:
        waves.append(temp)
        temp = [i]

    if i in set_syst_peaks and f and len(temp) < 5:
      temp.append(i)
    
    elif i in set_all_peaks and f and len(temp) < 5:
      temp.append(i)
      
    elif i in set_all_notches and f and len(temp) < 5:
      temp.append(i)
  # print("waves: ",waves)
  return waves

def get_wave_times_1d(waves):
  
  # set_holes = set()
  # for x in holes:
  #   set_holes.add(x)

  time = []
  # print("set_holes: ",set_holes)

  for x in waves:
    # h = x[0]
    # while h not in set_holes and h > 0:
    #   h -= 1
    temp = []
    for i in range(1,len(x)):
      temp.append(x[i]-x[0])
    time.append(temp)
  # print("time:")
  # for x in time:
  #  print(x)
  return time

def feature_extractor2_2d(signal, holes):

  set_holes = set()
  for x in holes:
    set_holes.add(x)

  neg_signal = [x*-1 for x in signal]

  syst_peaks, props = find_peaks(signal, distance=100)
  all_peaks, props = find_peaks(signal)
  all_notches, props = find_peaks(neg_signal)

  
  set_syst_peaks = set(syst_peaks)
  set_all_peaks = set(all_peaks)
  set_all_notches = set(all_notches)
  
  waves = []
  temp = []
  f = 0
  p = 0

  # print("holes: ",set_holes)

  for i in range(len(signal)):

    if i in set_holes:
      f = 1
      if len(temp) == 0:
        temp.append(i)
      if len(temp) > 1:
        waves.append(temp)
        temp = [i]

    # if i in set_syst_peaks and f and len(temp) < 5:
    #   temp.append(i)
    
    if i in set_all_peaks and f and len(temp) < 5:
      temp.append(i)
      
    elif i in set_all_notches and f and len(temp) < 5:
      temp.append(i)

  return waves

def get_wave_times_2d(waves):
  # set_holes = set()
  # for x in holes:
  #   set_holes.add(x)
  time = []
  # print(set_holes)
  for x in waves:
    # h = x[0]
    # print("h: ",h)
    # while h not in set_holes and h > 0:
    #   h -= 1
    temp = []
    for i in range(1,len(x)):
      temp.append(x[i]-x[0])
    time.append(temp)
  # for x in time:
  #  print(x)
  return time

class feature_extraction:

  def __init__(self, wave_type, signal_data, features_dataframe, 
               times_dataframe, second_derivative_times_dataframe, 
               first_derivative_times_dataframe, 
               second_derivative_peaks_dataframe):
    self.wave_type = wave_type
    self.signal_data = signal_data

    self.x_pos = features_dataframe['Systolic Peaks'].to_list()
    self.y_pos = features_dataframe['Diastolic Peaks'].to_list()
    self.z_pos = features_dataframe['Dicrotic Notches'].to_list()
    self.t1 = times_dataframe['t1'].to_list()
    self.t2 = times_dataframe['t2'].to_list()
    self.t3 = times_dataframe['t3'].to_list()
    self.tpi = times_dataframe['tpi'].to_list()
    self.ta1 = first_derivative_times_dataframe['ta1'].to_list()
    self.tb1 = first_derivative_times_dataframe['tb1'].to_list()
    self.te1 = first_derivative_times_dataframe['te1'].to_list()
    self.tl1 = first_derivative_times_dataframe['tl1'].to_list()
    self.a2_pos = second_derivative_peaks_dataframe['a2'].to_list()
    self.b2_pos = second_derivative_peaks_dataframe['b2'].to_list()
    self.e2_pos = second_derivative_peaks_dataframe['e2'].to_list()
    self.ta2 = second_derivative_times_dataframe['ta2'].to_list()
    self.tb2 = second_derivative_times_dataframe['tb2'].to_list()
    self.te2 = second_derivative_times_dataframe['te2'].to_list()
    self.tl2 = second_derivative_times_dataframe['tl2'].to_list()

  def aug_index(self,x,y):
    return y/x

  def alt_aug_index(self,x,y):
    return 1-self.aug_index(x,y)

  def z_x_ratio(self,x,z):
    return z/x

  def neg_aug_index(self,x,y):
    return self.aug_index(x,y) - 1

  def sys_diast_time(self,t1,t3):
    return t3-t1

  def half_sys_peak_time(self,t1):
    return t1/2

  def sys_peak_rising_slope(self,t1,x):
    return t1/x

  def diast_peak_fall_slope(self,y,tpi,t3):
    return y/(tpi-t3)

  def t1_tpi_ratio(self,t1,tpi):
    return t1/tpi

  def t2_tpi_ratio(self,t2,tpi):
    return t2/tpi

  def t3_tpi_ratio(self,t3,tpi):
    return t3/tpi

  def del_t_tpi_ratio(self,t1,t3,tpi):
    return self.sys_diast_time(t1,t3)/tpi

  def ta1_tpi_ratio(self,ta1,tpi):
    return ta1/tpi

  def tb1_tpi_ratio(self,tb1,tpi):
    return tb1/tpi

  def te1_tpi_ratio(self,te1,tpi):
    return te1/tpi

  def tl1_tpi_ratio(self,tl1,tpi):
    return tl1/tpi

  def b2_a2_ratio(self,b2,a2):
    return b2/a2

  def e2_a2_ratio(self,e2,a2):
    return e2/a2

  def b2_plus_e2_a2_ratio(self,b2,e2,a2):
    return  (b2 + e2)/a2
  
  def ta2_tpi_ratio(self,ta2,tpi):
    return  ta2/tpi

  def tb2_tpi_ratio(self,tb2,tpi):
    return  tb2/tpi

  def ta1_plus_ta2_tpi_ratio(self,ta1,ta2,tpi):
    return  (ta1 + ta2)/tpi

  def tb1_plus_tb2_tpi_ratio(self,tb1,tb2,tpi):
    return  (tb1 + tb2)/tpi

  def te1_plus_t2_tpi_ratio(self,te1,t2,tpi):
    return (te1 + t2)/tpi

  def tl1_plus_t3_tpi_ratio(self,tl1,t3,tpi):
    return (tl1 + t3)/tpi
  


  def feature_extractor(self):
    
    x=[]
    y=[]
    z=[]
    a2=[]
    b2=[]
    e2=[]
    l2=[]
    #Augmentation Index
    aug_index=[]
    # print("signal_data: ",self.signal_data)
    for i in range(len(self.x_pos)):
      x.append(self.signal_data[self.x_pos[i]])
      y.append(self.signal_data[self.y_pos[i]])
      aug_index.append(self.aug_index(self.signal_data[self.x_pos[i]], self.signal_data[self.y_pos[i]]))
    
    #Alternative Augmentation Index
    alt_aug_index=[]
    for i in range(len(self.x_pos)):
      alt_aug_index.append(self.alt_aug_index(self.signal_data[self.x_pos[i]], self.signal_data[self.y_pos[i]]))
    
    #Ratio of dicrotic notch and systolic peak
    notch_sys_ratio=[]
    for i in range(len(self.x_pos)):
      z.append(self.signal_data[self.z_pos[i]])
      notch_sys_ratio.append(self.z_x_ratio(self.signal_data[self.x_pos[i]], self.signal_data[self.z_pos[i]]))
    
    #Negative Relative Augmentation Index
    neg_aug_index=[]
    for i in range(len(self.x_pos)):
      neg_aug_index.append(self.neg_aug_index(
          self.signal_data[self.x_pos[i]], 
          self.signal_data[self.y_pos[i]])
      )

    #Time between Systolic and Diastolic Peak
    sys_diast_time=[]
    for i in range(len(self.t1)):
      sys_diast_time.append(self.sys_diast_time(self.t1[i],self.t3[i]))
    
    #Time between half systolic peak points
    half_sys_peak_time=[]
    for i in range(len(self.t1)):
      half_sys_peak_time.append(self.half_sys_peak_time(self.t1[i]))
    
    #Systolic peak rising slope
    sys_slope=[]
    for i in range(len(self.t1)):
      sys_slope.append(self.sys_peak_rising_slope(
          self.t1[i], 
          self.signal_data[self.x_pos[i]])
      )  

    #Diastolic peak falling time
    diast_slope=[]
    for i in range(len(self.t1)):
      diast_slope.append(self.diast_peak_fall_slope(
          self.signal_data[self.y_pos[i]], 
          self.tpi[i], 
          self.t3[i])
      )  

    #Ratio of systolic peak time (t1) and pulse interval time (tpi)
    t1_tpi_ratio=[]
    for i in range(len(self.t1)):
      t1_tpi_ratio.append(self.t1_tpi_ratio(
          self.t1[i], 
          self.tpi[i])
      )  
    
    #Ratio of dicrotic notch time (t2) and pulse interval time (tpi)
    t2_tpi_ratio=[]
    for i in range(len(self.t2)):
      t2_tpi_ratio.append(self.t2_tpi_ratio(
          self.t2[i], 
          self.tpi[i])
      ) 

    #Ratio of diastolic peak time (t3) and pulse interval time (tpi)
    t3_tpi_ratio=[]
    for i in range(len(self.t3)):
      t3_tpi_ratio.append(self.t3_tpi_ratio(
          self.t3[i], 
          self.tpi[i])
      )  

    #Ratio of time between systolic and diastolic peak (delta T) and pulse interval time (tpi)
    del_t_tpi_ratio=[]
    for i in range(len(self.t3)):
      del_t_tpi_ratio.append(self.del_t_tpi_ratio(
          self.t1[i], 
          self.t3[i],
          self.tpi[i])
      )  

    #Ratio of time interval of a1 (ta1) and pulse interval time (tpi)
    ta1_tpi_ratio=[]
    for i in range(len(self.tpi)):
      ta1_tpi_ratio.append(self.ta1_tpi_ratio(
          self.ta1[i], 
          self.tpi[i])
      ) 

    #Ratio of time interval of b1 (tb1) and pulse interval time (tpi)
    tb1_tpi_ratio=[]
    for i in range(len(self.tpi)):
      tb1_tpi_ratio.append(self.tb1_tpi_ratio(
          self.tb1[i], 
          self.tpi[i])
      )  

    #Ratio of time interval of e1 (te1) and pulse interval time (tpi)
    te1_tpi_ratio=[]
    for i in range(len(self.t3)):
      te1_tpi_ratio.append(self.te1_tpi_ratio(
          self.te1[i], 
          self.tpi[i])
      )  

    #Ratio of time interval of l1 (tl1) and pulse interval time (tpi)
    tl1_tpi_ratio=[]
    for i in range(len(self.t3)):
      tl1_tpi_ratio.append(self.tl1_tpi_ratio(
          self.tl1[i], 
          self.tpi[i])
      )  

    #Ratio of b2 and a2
    b2_a2_ratio=[]
    for i in range(len(self.t3)):
      b2.append(self.signal_data[self.b2_pos[i]])
      a2.append(self.signal_data[self.a2_pos[i]])
      b2_a2_ratio.append(self.b2_a2_ratio(self.signal_data[self.b2_pos[i]], 
                                          self.signal_data[self.a2_pos[i]]))

    #Ratio of e2 and a2
    e2_a2_ratio=[]
    for i in range(len(self.t3)):
      e2.append(self.signal_data[self.e2_pos[i]])      
      e2_a2_ratio.append(self.e2_a2_ratio(self.signal_data[self.e2_pos[i]], 
                                          self.signal_data[self.a2_pos[i]]))  

    #Ratio of (b2 + e2) and a2
    b2_plus_e2_a2_ratio=[]
    for i in range(len(self.t3)):
      b2_plus_e2_a2_ratio.append(self.b2_plus_e2_a2_ratio(
          self.signal_data[self.b2_pos[i]],
          self.signal_data[self.e2_pos[i]],
          self.signal_data[self.a2_pos[i]])
      )  

    #Ratio between time interval of a2 (ta2) and pulse interval (tpi)
    ta2_tpi_ratio=[]
    for i in range(len(self.t3)):
      ta2_tpi_ratio.append(self.ta2_tpi_ratio(
          self.ta2[i], 
          self.tpi[i])
      )

    #Ratio between time interval of b2 (tb2) and pulse interval (tpi)
    tb2_tpi_ratio=[]
    for i in range(len(self.t3)):
      tb2_tpi_ratio.append(self.tb2_tpi_ratio(
          b2[i], 
          self.tpi[i])
      )

    #Ratio (ta1 + ta2) and pulse interval (tpi)
    ta1_plus_ta2_tpi_ratio=[]
    for i in range(len(self.t3)):
      ta1_plus_ta2_tpi_ratio.append(self.ta1_plus_ta2_tpi_ratio(
          self.ta1[i], 
          self.ta2[i], 
          self.tpi[i]))

    #Ratio (tb1 + tb2) and pulse interval (tpi)
    tb1_plus_tb2_tpi_ratio=[]
    for i in range(len(self.t3)):
      tb1_plus_tb2_tpi_ratio.append(self.tb1_plus_tb2_tpi_ratio(
          self.tb1[i], 
          self.tb2[i], 
          self.tpi[i])
      )

    #Ratio (te1 + t2) and pulse interval (tpi)
    te1_plus_t2_tpi_ratio=[]
    for i in range(len(self.t3)):
      te1_plus_t2_tpi_ratio.append(self.te1_plus_t2_tpi_ratio(self.te1[i], 
                                                              self.t2[i], 
                                                              self.tpi[i])
      )

    #Ratio (tl1 + t3) and pulse interval (tpi)
    tl1_plus_t3_tpi_ratio=[]
    for i in range(len(self.t3)):
      tl1_plus_t3_tpi_ratio.append(self.tl1_plus_t3_tpi_ratio(
          self.tl1[i], 
          self.t3[i], 
          self.tpi[i])
      )


    features_dataframe_main = pd.DataFrame()

    features_dataframe_main["(f3) Systolic Peak"] = x
    features_dataframe_main["(f4) Diastolic Peak"] = y
    features_dataframe_main["(f5) Dicrotic Notch"] = z
    features_dataframe_main["(f6) Pusle interval"] =self.tpi
    features_dataframe_main["(f7) Augmentation Index"] =(aug_index)
    features_dataframe_main["(f8) Alternative Augmentation Index"] =(alt_aug_index)
    features_dataframe_main["(f9) Ratio of dicrotic notch and systolic peak"] =(notch_sys_ratio)
    features_dataframe_main["(f10) Negative Relative Augmentation Index"] =(neg_aug_index)
    features_dataframe_main["(f11) Systolic peak time"] =(self.t1)
    features_dataframe_main["(f12) Dicortic notch time"] =(self.t2)
    features_dataframe_main["(f13) Diastolic peak time"] =(self.t3)
    features_dataframe_main["(f14) Time between Systolic and Diastolic Peak"] =(sys_diast_time)
    features_dataframe_main["(f15) Time between half systolic peak points"] =(half_sys_peak_time)
    features_dataframe_main["(f17) Systolic peak rising slope"] =(sys_slope)
    features_dataframe_main["(f18) Diastolic peak falling time"] =(diast_slope)
    features_dataframe_main["(f19) Ratio of systolic peak time (t1) and pulse interval time (tpi)"] =(t1_tpi_ratio)
    features_dataframe_main["(f20) Ratio of dicrotic notch time (t2) and pulse interval time (tpi)"] =(t2_tpi_ratio)
    features_dataframe_main["(f21) Ratio of diastolic peak time (t3) and pulse interval time (tpi)"] =(t3_tpi_ratio)
    features_dataframe_main["(f22) Ratio of time between systolic and diastolic peak (delta T) and pulse interval time (tpi)"] =(del_t_tpi_ratio)
    features_dataframe_main["(f23) Interval time from point l1 to point a1 for 1st derivative of PPG"] =(self.ta1)
    features_dataframe_main["(f24) Interval time from point l1 to point b1"] =(self.tb1)
    features_dataframe_main["(f25) Interval time from point l1 to point e1"] =(self.te1)
    features_dataframe_main["(f26) Interval time from point l1 to point l1"] =(self.tl1)
    features_dataframe_main["(f27) Ratio of time interval of a1 (ta1) and pulse interval time (tpi)"] =(ta1_tpi_ratio)
    features_dataframe_main["(f28) Ratio of time interval of b1 (tb1) and pulse interval time (tpi)"] =(tb1_tpi_ratio)
    features_dataframe_main["(f29) Ratio of time interval of e1 (te1) and pulse interval time (tpi)"] =(te1_tpi_ratio)
    features_dataframe_main["(f30) Ratio of time interval of l1 (tl1) and pulse interval time (tpi)"] =(tl1_tpi_ratio)
    features_dataframe_main["(f31) Ratio of b2 and a2"] =(b2_a2_ratio)
    features_dataframe_main["(f32) Ratio of e2 and a2"] =(e2_a2_ratio)
    features_dataframe_main["(f33) Ratio of (b2 + e2) and a2"] =(b2_plus_e2_a2_ratio)
    features_dataframe_main["(f34) Interval time from point l2 to next point a1 for 1st derivative of PPG"] =(self.ta2)
    features_dataframe_main["(f35) Interval time from point l2 to point b2"] =(self.tb2)
    features_dataframe_main["(f36) Ratio between time interval of a2 (ta2) and pulse interval (tpi)"] =(ta2_tpi_ratio)
    features_dataframe_main["(f37) Ratio between time interval of b2 (tb2) and pulse interval (tpi)"] =(tb2_tpi_ratio)
    features_dataframe_main["(f38) Ratio (ta1 + ta2) and pulse interval (tpi)"] =(ta1_plus_ta2_tpi_ratio)
    features_dataframe_main["(f39) Ratio (tb1 + tb2) and pulse interval (tpi)"] =(tb1_plus_tb2_tpi_ratio)
    features_dataframe_main["(f40) Ratio (te1 + t2) and pulse interval (tpi)"] =(te1_plus_t2_tpi_ratio)
    features_dataframe_main["(f41) Ratio (tl1 + t3) and pulse interval (tpi)"] =(tl1_plus_t3_tpi_ratio)
    # features_dataframe_main["(f) ta2"] = self.second_derivative_times_dataframe['ta2']
    # features_dataframe_main["(f) tb2"] = self.second_derivative_times_dataframe['tb2']
    # features_dataframe_main["(f) te2"] = self.second_derivative_times_dataframe['te2']
    # features_dataframe_main["(f) tl2"] = self.second_derivative_times_dataframe['tl2']
    # return features_dataframe_main

    

    if not os.path.isfile("features_"+self.wave_type+".csv"):
      features_dataframe_main.to_csv("features_"+self.wave_type+".csv", index = False, mode = "w", header='column_names')
    else:
      features_dataframe_main.to_csv("features_"+self.wave_type+".csv", index = False, mode = "a", header=False)
    # print(features_dataframe_main)
    # Features = pd.DataFrame(np.array(second_derivative_times_dataframe), columns=['ta2', 'tb2', 'te2', 'tl2'])
    # Features.to_csv("Features.csv", index=False, mode="w")

def filter(data):
  data['IR'] = savgol_filter(data['IR'],41,3)
  data['RED'] = savgol_filter(data['RED'],41,3)

  data['IR'] = savgol_filter(data['IR'],41,3)
  data['RED'] = savgol_filter(data['RED'],41,3)

  # data['IR'] = savgol_filter(data['IR'],41,3)
  # data['RED'] = savgol_filter(data['RED'],41,3)
  return data

def filter1d(d1_blood_vol_df):
  d1_blood_vol_df['IR'] = savgol_filter(d1_blood_vol_df['IR'],31,3)
  d1_blood_vol_df['RED'] = savgol_filter(d1_blood_vol_df['RED'],31,3)
  return d1_blood_vol_df

def run(blood_vol_df):
    # blood_vol_df = pd.read_csv("blood_vol_data.csv")

    blood_vol_IR_list = blood_vol_df['IR'].values.tolist()
    blood_vol_RED_list = blood_vol_df['RED'].values.tolist()

    blood_vol_IR_holes_list = get_holes(blood_vol_df['IR'].values.tolist())
    blood_vol_RED_holes_list = get_holes(blood_vol_df['RED'].values.tolist())

    blood_vol_IR_holes_df = pd.DataFrame(np.array(blood_vol_IR_holes_list), columns=['Holes'])
    # blood_vol_IR_holes_df.to_csv("blood_vol_IR_holes.csv", index=False, mode="w")
    # print("Got IR holes")

    blood_vol_RED_holes_df = pd.DataFrame(np.array(blood_vol_RED_holes_list), columns=['Holes'])
    # blood_vol_RED_holes_df.to_csv("blood_vol_RED_holes.csv", index=False, mode="w")
    # print("Got RED holes")



    d1_blood_vol_list = get_derivative(blood_vol_df)

    d1_blood_vol_df = pd.DataFrame(np.array(d1_blood_vol_list), columns=['IR', 'RED'])

    d1_blood_vol_df = filter1d(d1_blood_vol_df)
    # print("Got first derivative")

    # d1_blood_vol_df.to_csv("d1_blood_vol_data.csv", index=False, mode="w")


    # d1_blood_vol_df = pd.read_csv("d1_blood_vol_data.csv")


    d2_blood_vol_list = get_derivative(d1_blood_vol_df)

    d2_blood_vol_df = pd.DataFrame(np.array(d2_blood_vol_list), columns=['IR', 'RED'])

    d2_blood_vol_df = filter1d(d2_blood_vol_df)
    # print("Got second derivative")

    # d2_blood_vol_df.to_csv("d2_blood_vol_data.csv", index=False, mode="w")




    blood_vol_IR_features_list = feature_extractor2(blood_vol_IR_list)
    # print(blood_vol_IR_features_list)
    blood_vol_IR_features_list = select_good_waves(blood_vol_IR_features_list)
    print(blood_vol_IR_features_list)
    if not blood_vol_IR_features_list or len(blood_vol_IR_features_list[0]) < 5:
      print("1Bad data, moving to next wave (;")
      return ''
    blood_vol_IR_features_df = pd.DataFrame(np.array(blood_vol_IR_features_list), columns=['Holes','Systolic Peaks', 'Dicrotic Notches', 'Diastolic Peaks', 'Holes\''])
    # print("Got blood vol IR features")

    blood_vol_RED_features_list = feature_extractor2(blood_vol_RED_list)
    blood_vol_RED_features_list = select_good_waves(blood_vol_RED_features_list)
    if not blood_vol_RED_features_list or len(blood_vol_RED_features_list[0])  < 5:
      print("1Bad data, moving to next wave (;")
      return ''
    blood_vol_RED_features_df = pd.DataFrame(np.array(blood_vol_RED_features_list), columns=['Holes','Systolic Peaks', 'Dicrotic Notches', 'Diastolic Peaks', 'Holes\''])
    # print("Got blood vol RED features")

    # blood_vol_IR_time_waves_list = pd.read_csv("blood_vol_IR_features.csv").to_numpy().tolist()
    # blood_vol_RED_time_waves_list = pd.read_csv("blood_vol_RED_features.csv").to_numpy().tolist()

    blood_vol_IR_times_list = get_wave_times(blood_vol_IR_features_list)

    blood_vol_IR_times_df = pd.DataFrame(np.array(blood_vol_IR_times_list), columns=['t1', 't2', 't3', 'tpi'])
    # print("Got blood vol IR times")

    blood_vol_RED_times_list = get_wave_times(blood_vol_RED_features_list)
    blood_vol_RED_times_df = pd.DataFrame(np.array(blood_vol_RED_times_list), columns=['t1', 't2', 't3', 'tpi'])
    # print("Got blood vol RED times")

    # blood_vol_IR_times_df.to_csv("blood_vol_IR_wave_times.csv", index=False, mode="w")
    # blood_vol_RED_times_df.to_csv("blood_vol_RED_wave_times.csv", index=False, mode="w")


    # d1_blood_vol_df = pd.read_csv("d1_blood_vol_data.csv")

    # blood_vol_IR_holes_df = pd.read_csv("blood_vol_IR_holes.csv")
    # blood_vol_RED_holes_df = pd.read_csv("blood_vol_RED_holes.csv")

    d1_IR_features_list = feature_extractor2_1d(d1_blood_vol_df['IR'], blood_vol_IR_holes_df['Holes'])
    d1_RED_features_list = feature_extractor2_1d(d1_blood_vol_df['RED'], blood_vol_RED_holes_df['Holes'])
    if not d1_IR_features_list or len(d1_IR_features_list[0])  < 5:
      print("2Bad data, moving to next wave (;")
      return ''
    if not d1_RED_features_list or len(d1_RED_features_list[0])  < 5:
      print("2Bad data, moving to next wave (;")
      return ''

    d1_IR_features_df = pd.DataFrame(np.array(d1_IR_features_list), columns=['hole','a1', 'b1', 'e1', 'l1'])
    # d1_IR_features_df.to_csv("d1_IR_features.csv", index=False, mode="w")
    # print('Got d1 IR features')
    d1_RED_features_df = pd.DataFrame(np.array(d1_IR_features_list), columns=['hole','a1', 'b1', 'e1', 'l1'])
    # d1_RED_features_df.to_csv("d1_RED_features.csv", index=False, mode="w")
    # print('Got d1 RED features')



    d1_IR_times_list = get_wave_times_1d(d1_IR_features_list)
    # print("**************************************************")
    d1_RED_times_list = get_wave_times_1d(d1_RED_features_list)

    d1_IR_times_df = pd.DataFrame(np.array(d1_IR_times_list), columns=['ta1', 'tb1', 'te1', 'tl1'])
    # d1_IR_times_df.to_csv("d1_IR_wave_times.csv", index=False, mode="w")
    # print('Got d1 IR wave times')
    d1_RED_times_df = pd.DataFrame(np.array(d1_RED_times_list), columns=['ta1', 'tb1', 'te1', 'tl1'])
    # d1_RED_times_df.to_csv("d1_RED_wave_times.csv", index=False, mode="w")
    # print('Got d1 RED wave times')




    d2_IR_features_list = feature_extractor2_2d(d2_blood_vol_df['IR'], blood_vol_IR_holes_df['Holes'])
    d2_RED_features_list = feature_extractor2_2d(d2_blood_vol_df['RED'], blood_vol_RED_holes_df['Holes'])
    if not d2_IR_features_list or len(d2_IR_features_list[0])  < 5:
      print("3Bad data, moving to next wave (;")
      return ''
    if not d2_RED_features_list or len(d2_RED_features_list[0])  < 5:
      print("3Bad data, moving to next wave (;")
      return ''
    d2_IR_features_df = pd.DataFrame(np.array(d2_IR_features_list), columns=['hole','a2', 'b2', 'e2', 'l2'])
    # d2_IR_features_df.to_csv("d2_IR_features.csv", index=False, mode="w")
    # print('Got d2 IR features')
    d2_RED_features_df = pd.DataFrame(np.array(d2_RED_features_list), columns=['hole','a2', 'b2', 'e2', 'l2'])
    # d2_RED_features_df.to_csv("d2_RED_features.csv", index=False, mode="w")
    # print('Got d2 RED features')



    d2_IR_times_list = get_wave_times_2d(d2_IR_features_list)
    # print("**************************************************")
    d2_RED_times_list = get_wave_times_2d(d2_RED_features_list)

    d2_IR_times_df = pd.DataFrame(np.array(d2_IR_times_list), columns=['ta2', 'tb2', 'te2', 'tl2'])
    # d2_IR_times_df.to_csv("d2_IR_wave_times.csv", index=False, mode="w")
    # print('Got d1 IR wave times')
    d2_RED_times_df = pd.DataFrame(np.array(d2_RED_times_list), columns=['ta2', 'tb2', 'te2', 'tl2'])
    # d2_RED_times_df.to_csv("d2_RED_wave_times.csv", index=False, mode="w")
    # print('Got d2 RED wave times')


    # print("blood_vol_df[IR]", blood_vol_df['IR'])
    IR_features_extraction=feature_extraction('IR', blood_vol_df['IR'], blood_vol_IR_features_df, blood_vol_IR_times_df, d2_IR_times_df, d1_IR_times_df, d2_IR_features_df)
    df = IR_features_extraction.feature_extractor()
    # print("IR DONE")

    RED_features_extraction=feature_extraction('RED', blood_vol_df['RED'], blood_vol_RED_features_df, blood_vol_RED_times_df, d2_RED_times_df, d1_RED_times_df, d2_RED_features_df)
    df = RED_features_extraction.feature_extractor()
    # print("RED DONE")
    # df.to_csv('features1.csv', mode='a', index=False, header=False)
    # print("Got Features")

        
# def run(blood_vol_df):
    # blood_vol_IR_list = blood_vol_df['IR'].values.tolist()
    # blood_vol_RED_list = blood_vol_df['RED'].values.tolist()

    # blood_vol_IR_holes_list = get_holes(blood_vol_df['IR'].values.tolist())
    # blood_vol_RED_holes_list = get_holes(blood_vol_df['RED'].values.tolist())

    # blood_vol_IR_holes_df = pd.DataFrame(np.array(blood_vol_IR_holes_list), columns=['Holes'])


    # blood_vol_RED_holes_df = pd.DataFrame(np.array(blood_vol_RED_holes_list), columns=['Holes'])

    # d1_blood_vol_list = get_derivative(blood_vol_df)

    # d1_blood_vol_df = pd.DataFrame(np.array(d1_blood_vol_list), columns=['IR', 'RED'])

    # d1_blood_vol_df = filter1d(d1_blood_vol_df)


    # d2_blood_vol_list = get_derivative(d1_blood_vol_df)

    # d2_blood_vol_df = pd.DataFrame(np.array(d2_blood_vol_list), columns=['IR', 'RED'])

    # d2_blood_vol_df = filter1d(d2_blood_vol_df)



    # blood_vol_IR_features_list = feature_extractor2(blood_vol_IR_list)

    # blood_vol_IR_features_list = select_good_waves(blood_vol_IR_features_list)

    # if len(blood_vol_IR_features_list) == 0:
    #     print("Bad data, moving to next wave (;")
    #     return ''
    # blood_vol_IR_features_df = pd.DataFrame(np.array(blood_vol_IR_features_list), columns=['Holes','Systolic Peaks', 'Dicrotic Notches', 'Diastolic Peaks', 'Holes\''])


    # blood_vol_RED_features_list = feature_extractor2(blood_vol_RED_list)
    # blood_vol_RED_features_list = select_good_waves(blood_vol_RED_features_list)
    # if len(blood_vol_RED_features_list) == 0:
    #     print("Bad data, moving to next wave (;")
    #     return ''
    # blood_vol_RED_features_df = pd.DataFrame(np.array(blood_vol_RED_features_list), columns=['Holes','Systolic Peaks', 'Dicrotic Notches', 'Diastolic Peaks', 'Holes\''])


    # blood_vol_IR_times_list = get_wave_times(blood_vol_IR_features_list)

    # blood_vol_IR_times_df = pd.DataFrame(np.array(blood_vol_IR_times_list), columns=['t1', 't2', 't3', 'tpi'])


    # blood_vol_RED_times_list = get_wave_times(blood_vol_RED_features_list)
    # blood_vol_RED_times_df = pd.DataFrame(np.array(blood_vol_RED_times_list), columns=['t1', 't2', 't3', 'tpi'])
    # # print("Got blood vol RED times")

    # # blood_vol_IR_times_df.to_csv("blood_vol_IR_wave_times.csv", index=False, mode="w")
    # # blood_vol_RED_times_df.to_csv("blood_vol_RED_wave_times.csv", index=False, mode="w")


    # # d1_blood_vol_df = pd.read_csv("d1_blood_vol_data.csv")

    # # blood_vol_IR_holes_df = pd.read_csv("blood_vol_IR_holes.csv")
    # # blood_vol_RED_holes_df = pd.read_csv("blood_vol_RED_holes.csv")

    # d1_IR_features_list = feature_extractor2_1d(d1_blood_vol_df['IR'], blood_vol_IR_holes_df['Holes'])
    # d1_RED_features_list = feature_extractor2_1d(d1_blood_vol_df['RED'], blood_vol_RED_holes_df['Holes'])
    # if len(d1_IR_features_list) == 0:
    #     print("Bad data, moving to next wave (;")
    #     return ''
    # if len(d1_RED_features_list) == 0:
    #     print("Bad data, moving to next wave (;")
    #     return ''

    # d1_IR_features_df = pd.DataFrame(np.array(d1_IR_features_list), columns=['hole','a1', 'b1', 'e1', 'l1'])
    # # d1_IR_features_df.to_csv("d1_IR_features.csv", index=False, mode="w")
    # # print('Got d1 IR features')
    # d1_RED_features_df = pd.DataFrame(np.array(d1_IR_features_list), columns=['hole','a1', 'b1', 'e1', 'l1'])
    # # d1_RED_features_df.to_csv("d1_RED_features.csv", index=False, mode="w")
    # # print('Got d1 RED features')



    # d1_IR_times_list = get_wave_times_1d(d1_IR_features_list)
    # # print("**************************************************")
    # d1_RED_times_list = get_wave_times_1d(d1_RED_features_list)

    # d1_IR_times_df = pd.DataFrame(np.array(d1_IR_times_list), columns=['ta1', 'tb1', 'te1', 'tl1'])
    # # d1_IR_times_df.to_csv("d1_IR_wave_times.csv", index=False, mode="w")
    # # print('Got d1 IR wave times')
    # d1_RED_times_df = pd.DataFrame(np.array(d1_RED_times_list), columns=['ta1', 'tb1', 'te1', 'tl1'])
    # # d1_RED_times_df.to_csv("d1_RED_wave_times.csv", index=False, mode="w")
    # # print('Got d1 RED wave times')




    # d2_IR_features_list = feature_extractor2_2d(d2_blood_vol_df['IR'], blood_vol_IR_holes_df['Holes'])
    # d2_RED_features_list = feature_extractor2_2d(d2_blood_vol_df['RED'], blood_vol_RED_holes_df['Holes'])
    # if len(d2_IR_features_list) == 0:
    #     print("Bad data, moving to next wave (;")
    #     return ''
    # if len(d2_RED_features_list) == 0:
    #     print("Bad data, moving to next wave (;")
    #     return ''
    # d2_IR_features_df = pd.DataFrame(np.array(d2_IR_features_list), columns=['hole','a2', 'b2', 'e2', 'l2'])
    # # d2_IR_features_df.to_csv("d2_IR_features.csv", index=False, mode="w")
    # # print('Got d2 IR features')
    # d2_RED_features_df = pd.DataFrame(np.array(d2_RED_features_list), columns=['hole','a2', 'b2', 'e2', 'l2'])
    # # d2_RED_features_df.to_csv("d2_RED_features.csv", index=False, mode="w")
    # # print('Got d2 RED features')
    # # print("blood_vol_df[IR]", blood_vol_df['IR'])
    # d2_IR_times_list = get_wave_times_2d(d2_IR_features_list)
    # # print("**************************************************")
    # d2_RED_times_list = get_wave_times_2d(d2_RED_features_list)

    # d2_IR_times_df = pd.DataFrame(np.array(d2_IR_times_list), columns=['ta2', 'tb2', 'te2', 'tl2'])
    # # d2_IR_times_df.to_csv("d2_IR_wave_times.csv", index=False, mode="w")
    # # print('Got d1 IR wave times')
    # d2_RED_times_df = pd.DataFrame(np.array(d2_RED_times_list), columns=['ta2', 'tb2', 'te2', 'tl2'])
    # # d2_RED_times_df.to_csv("d2_RED_wave_times.csv", index=False, mode="w")
    # # print('Got d2 RED wave times')

    # IR_features_extraction=feature_extraction('IR', blood_vol_df['IR'], blood_vol_IR_features_df, blood_vol_IR_times_df, d2_IR_times_df, d1_IR_times_df, d2_IR_features_df)
    # IR_features_extraction.feature_extractor()
    # # print("IR DONE")

    # RED_features_extraction=feature_extraction('RED', blood_vol_df['RED'], blood_vol_RED_features_df, blood_vol_RED_times_df, d2_RED_times_df, d1_RED_times_df, d2_RED_features_df)
    # RED_features_extraction.feature_extractor()
    # print("RED DONE")
    # df.to_csv('features1.csv', mode='a', index=False, header=False)
    # print("Got Features")

def start(full_baseline_corrected_list):

    # full_baseline_corrected_list = correct_baseline(full_data)

    full_baseline_corrected_df = pd.DataFrame(np.array(full_baseline_corrected_list), columns=['IR', 'RED'])
    full_baseline_corrected_df.to_csv("baseline_corrected_data.csv", index=False, mode="w")

    full_baseline_corrected_df = pd.read_csv("baseline_corrected_data.csv")
    full_blood_vol_list = get_bloodVol_data(full_baseline_corrected_df)
    full_blood_vol_df = pd.DataFrame(np.array(full_blood_vol_list), columns=['IR', 'RED'])
    # full_blood_vol_df.to_csv("blood_vol_data.csv", index=False, mode="w")

    full_blood_vol_IR_list = full_blood_vol_df['IR'].values.tolist()
    full_blood_vol_RED_list = full_blood_vol_df['RED'].values.tolist()



    full_blood_vol_IR_holes_list = get_holes(full_blood_vol_IR_list)
    full_blood_vol_RED_holes_list = get_holes(full_blood_vol_RED_list)
    # print(full_blood_vol_IR_holes_list)
    i = 1
    # f = 0
    while i < len(full_blood_vol_IR_holes_list):

        left_limit_IR = full_blood_vol_IR_holes_list[i-1]
        right_limit_IR = full_blood_vol_IR_holes_list[i]
        # print(left_limit_IR)
        # print(right_limit_IR)

        left_limit_IR = max(0,left_limit_IR-20)
        right_limit_IR = min(len(full_blood_vol_IR_list)-1, right_limit_IR+20)
        # print('i: ',i)
        # print("left_limit_IR: ",left_limit_IR)
        # print("right_limit_IR",right_limit_IR)
        blood_vol_df = full_blood_vol_df.iloc[left_limit_IR:right_limit_IR].reset_index()
        i += 1
        run(blood_vol_df)