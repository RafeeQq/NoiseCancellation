#importing packages to be used : numpy for the operations , matplotlib for plotting & sounddevice for producing the audio.
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from scipy.fftpack import fft

# Defining each note & matching it to its frequency.
C =130.81
D =146.83
E =164.81
F =174.61
G =196
A =220
B =246.93

#Creating 12*1024 samples from 0 to 3 which is the length of the song (3 seconds).
t = np.linspace(0,3,12*1024)

#Create a list with the notes to be played .
temp = np.array([A/2,0,A/2,E,D,C,B/2,B/2,B/2,D,C,B/2,A/2,A/2,0,C*2,B,C*2,B,C*2, A/2,A/2]) 

#creating a corresponding list with the length of the notes (setting the duration).
duration =np.array([0.1,0.05,0.1,0.1,0.2,0.2,0.2,0.1,0.1,0.2,0.1,0.1,0.2,0.1,0.05,0.1,0.1,0.1,0.1,0.1, 0.2,0.1]) * 0.7

# using a loop to set the length of each note in the list to its duration and adding it to the other octave for each note using its corresponding frequency..
sum1 = 0
i = 0 
T = 0
for x in temp:
    #defining the signal
    sum1 += (np.sin(2*np.pi*x*t)+np.sin(2*np.pi*x*2*t))*((t>=T) & (t<=T + duration[i]))
   
    T += duration[i] + 0.05 
    i += 1

#plotting the produced sum to show the whole wave in time domain .
#using sounddevice to play the audio.

N = 3 * 1024
f = np.linspace(0, 512, int(N / 2))

x_f = fft(sum1)
x_f = 2 / N * np.abs(x_f[0 : np.int(N / 2)])

fn1, fn2 = np.random.randint(0, 512, 2)
#fn2 = np.random.randint(0, 512, 2)

n = np.sin(2 * np.pi * fn1 * t) + np.sin(2 * np.pi * fn2 * t)

max1 = int(np.max(x_f)) + 1
ff1 = []


sum2 = sum1 + n

x2_f = fft(sum2)
x2_f = 2 / N * np.abs(x2_f[0 : np.int(N / 2)])


i = 0
while i < len(f):
    ff = x2_f[i]
    if ff > max1:
        ff1.append(int(f[i]))        
    i += 1

sum3 = sum2 - (np.sin(2 * np.pi * ff1[0] * t) + np.sin(2 * np.pi * ff1[1] * t))

x3_f = fft(sum3)
x3_f = 2 / N * np.abs(x3_f[0 : np.int(N / 2)])

sd.play(sum1, 3*1024)

plt.subplot(2, 2, 1)
plt.plot(t,sum1)

plt.subplot(2, 2, 2)
plt.plot(f,x_f)


plt.subplot(2, 2, 3)
plt.plot(t,sum2)

plt.subplot(2, 2, 4)
plt.plot(f,x2_f)