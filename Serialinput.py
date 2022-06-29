
import serial
import pywt
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, welch, periodogram, butter, filtfilt, iirnotch
from scipy.interpolate import UnivariateSpline

import serial.tools.list_ports as ports

com_ports = list(ports.comports())  # create a list of com ['COM1','COM2']

for i in com_ports:
    print(i.device)  # returns 'COMx'

# This block dedicated to contain function block that support main program

def butter_lowpass(cutoff, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter (order, normal_cutoff, btype='low', analog=False)
    return b,a

def butter_highpass(cutoff, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter (order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, sample_rate, order=2):
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter (order, [low, high], btype='band')
    return b, a

def  calc_baseline (signal):         #function to calculate and remove baseline
    """
    Calculate the baseline of signal.

    Args:
        signal (numpy 1d array): signal whose baseline should be calculated


    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")
    s = signal - baseline[: len(signal)]
    return (s ,baseline)

def correct_baseline(data):
  d = []
  # IR_data, IR_base_line = calc_baseline(data['IR'])
  RED_data, RED_base_line = calc_baseline(data)
  #
  # IR_data = IR_data.values.tolist()
  # RED_data = RED_data.tolist()

  # IR_data = down_scaling(IR_data)
  # RED_data = down_scaling(RED_data)
  # print(IR_data)
  # scaling(IR_data)
  for i in range(len(RED_data)):
    # row=[]
    d.append(RED_data[i])
    # row.append(RED_data[i])
    # d.append(row)
  # print(np.array(d))
  return d

def process_data(data):
    try:
        param = list (map (int, data.split (",")))
        return param
    except:
        return data

def plotdata(data):
    plt.ion()
    fig = plt.figure (figsize=(10, 10))
    for d in data:
        plt.plot (d)
    plt.show ()
    plt.pause(1.5)
    plt.close(fig)

def filter_signal(data, cutoff, sample_rate, order=2, filtertype='lowpass', return_top = False):
    if filtertype.lower () == 'lowpass':
        b, a = butter_lowpass (cutoff, sample_rate, order=order)
    elif filtertype.lower () == 'highpass':
        b, a = butter_highpass (cutoff, sample_rate, order=order)
    elif filtertype.lower () == 'bandpass':
        assert type (cutoff) == tuple or list or np.array, 'if bandpass filter is specified, \
    cutoff needs to be array or tuple specifying lower and upper bound: [lower, upper].'
        b, a = butter_bandpass (cutoff[0], cutoff[1], sample_rate, order=order)
    elif filtertype.lower () == 'notch':
        b, a = iirnotch (cutoff, Q=0.005, fs=sample_rate)
    else:
        raise ValueError ('filtertype: %s is unknown, available are: \
    lowpass, highpass, bandpass, and notch' % filtertype)

    filtered_data = filtfilt (b, a, data)

    if return_top:
        return np.clip (filtered_data, a_min=0, a_max=None)
    else:
        return filtered_data

def calc_hr(data, sampling_rate): # Here the data is peak values from ppg signal
    hrInterval = []
    datalen = len(data)
    for i in range(1, datalen):
        rr_interval = data[i] - data[i-1]
        rr_time = (rr_interval / sampling_rate) * 1000
        hrInterval.append(rr_time)

    hr = 60000 / np.mean(hrInterval)
    return hr

def calc_spo2(reddata, irdata):
    A = 100
    B = 2.7
    ppg_Red_std = np.std(reddata)
    ppg_IR_std = np.std(irdata)
    ppg_Red = np.mean(reddata)
    ppg_IR = np.mean(irdata)

    spo2 = (A - B*(ppg_Red_std / ppg_Red )/(ppg_IR_std / ppg_IR))
    return spo2

def calc_rr(data, sampling_rate): # Data is actually the peaks indexes
    # Calc respiration rate parameters
    cutoff_freq = [0.1, 0.4]
    method = 'welch'
    filter_breathing = True

    # Finding the RR-interval to be passed in to the actual main function.
    hrInterval = []
    datalen = len(data)
    for i in range(1, datalen):
        rr_interval = data[i] - data[i-1]
        rr_time = (rr_interval / sampling_rate) * 1000
        hrInterval.append(rr_time)

    # Main function
    x = np.linspace(0, len(hrInterval), len(hrInterval))
    x_new = np.linspace(0, len(hrInterval), np.sum(hrInterval, dtype=np.int32))
    interp = UnivariateSpline(x, hrInterval, k=3)
    breathing = interp(x_new)

    # Filtering
    if filter_breathing:
        breathing = filter_signal(breathing, cutoff=cutoff_freq, sample_rate = 1000.0, filtertype='bandpass')

    if method.lower () == 'fft':
        datalen = len (breathing)
        frq = np.fft.fftfreq (datalen, d=((1 / 1000.0)))
        frq = frq[range (int (datalen / 2))]
        Y = np.fft.fft (breathing) / datalen
        Y = Y[range (int (datalen / 2))]
        psd = np.power (np.abs (Y), 2)
    elif method.lower () == 'welch':
        if len (breathing) < 30000:
            frq, psd = welch (breathing, fs=1000, nperseg=len (breathing))
        else:
            frq, psd = welch (breathing, fs=1000, nperseg= np.clip(len(breathing)//10, a_min=30000, a_max=None))
    elif method.lower () == 'periodogram':
        frq, psd = periodogram (breathing, fs=1000.0, nfft=30000)

    else:
        raise ValueError ('Breathing rate extraction method not understood! Must be \'welch\' or \'fft\'!')

    breathingRate = frq[np.argmax(psd)]
    return 60 * breathingRate

def calc_sqi(data):
    pass


serialPort = serial.Serial(port = "COM3", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

serialString = ""                           # Used to hold data coming over UART
red_buffer = []
ir_buffer = []

updateBuffer = 200
bufferLength = 400

n = 0
prvspo2 = 0

while 1:

    # Wait until there is data waiting in the serial buffer
    if serialPort.in_waiting > 0:

        # Read data out of the buffer until a carraige return / new line is found
        serialString = serialPort.readline()

        # Print the contents of the serial data
        input_from_brd = serialString.decode ('Ascii')
        processed_output = process_data(input_from_brd)

        if isinstance (processed_output, list):

            red = processed_output[4]
            ir = processed_output[3]

            if n < bufferLength:
                red_buffer.append (red)
                ir_buffer.append (ir)
                n += 1

            else:
                n = 0

                smooth_red = savgol_filter(red_buffer,27,3)
                smooth_ir = savgol_filter (ir_buffer,27, 3)
                smooth_red = savgol_filter (smooth_red, 27, 3)
                smooth_ir = savgol_filter (smooth_ir, 27, 3)

                # smooth_red = pd.Series (red_buffer).rolling (5).mean ()
                # smooth_ir = pd.Series (ir_buffer).rolling (5).mean ()

                # Baseline correction
                basecorectedRed = correct_baseline (smooth_red)
                basecorectedIr = correct_baseline (smooth_ir)
                redPeaks, _ = find_peaks(np.array(basecorectedRed), distance=60)
                irPeaks, _ = find_peaks (np.array(basecorectedIr), distance=60)
                npredPeaks = np.array(redPeaks)
                npirPeaks = np.array(irPeaks)

                redpeaksindex = []
                for t in redPeaks:
                    redpeaksindex.append(basecorectedRed[t])
                redpeaksindex = np.array(redpeaksindex)

                irpeaksindex = []
                for t in irPeaks:
                    irpeaksindex.append(basecorectedIr[t])
                irpeaksindex = np.array(irpeaksindex)

                redhr = calc_hr(npredPeaks, 100)
                irhr = calc_hr(npirPeaks, 100)
                spo2 = calc_spo2((basecorectedRed), basecorectedIr)
                redRr = calc_rr(npredPeaks, 100)
                irRr = calc_rr(npirPeaks, 100)

                # Spo2 Averaging
                if prvspo2 == 0:
                    prvspo2 = spo2
                else:
                    if abs(prvspo2 - spo2) > 2:
                        spo2 = prvspo2 - 1 if spo2 < prvspo2 else prvspo2 + 1
                        prvspo2 = spo2

                spo2 = spo2 if spo2 <= 100 else 100

                print("RED HR: ", round(redhr))
                print("IR HR: ", round(irhr))
                print("SPO2: ", round(spo2))
                print("Red RR: ", round(redRr))
                print("Ir RR: ", round(irRr))
                print()

                plt.ion()
                fig = plt.figure()
                plt.plot(basecorectedRed)
                plt.plot(npredPeaks, redpeaksindex, 'ro')
                plt.plot (basecorectedIr)
                plt.plot (npirPeaks, irpeaksindex, 'bo')
                plt.show()
                plt.pause(1.5)
                plt.close(fig)

                # plotdata ([basecorectedRed, basecorectedIr])

                red_buffer = red_buffer[-updateBuffer:]
                ir_buffer = ir_buffer[-updateBuffer:]

        # Tell the device connected over the serial port that we recevied the data!
        # The b at the beginning is used to indicate bytes!
        # serialPort.write(b"Thank you for sending data \r\n")

        # -------------------------------------------------------------------------------------------------------#
        #                          INDHA KOTTA THAANDI NAANU VARAMATE NIUM VARAKODATHU                           #
        # -------------------------------------------------------------------------------------------------------#
