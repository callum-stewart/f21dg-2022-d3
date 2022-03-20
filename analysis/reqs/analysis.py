#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import emd
from scipy import signal


# In[2]:


sample_rate = 1000
seconds = 10
num_samples = sample_rate*seconds

time_vect = np.linspace(0, seconds, num_samples)

freq = 5

# Change extent of deformation from sinusoidal shape [-1 to 1]
nonlinearity_deg = 0.25

# Change left-right skew of deformation [-pi to pi]
nonlinearity_phi = -np.pi/4

# Compute the signal

# Create a non-linear oscillation
x = emd.simulate.abreu2010(freq, nonlinearity_deg, nonlinearity_phi, sample_rate, seconds)

x += np.cos(2 * np.pi * 1 * time_vect)        # Add a simple 1Hz sinusoid
x -= np.sin(2 * np.pi * 2.2e-1 * time_vect)   # Add part of a very slow cycle as a trend


# Visualise the time-series for analysis
plt.figure(figsize=(12, 4))
plt.plot(x)


# In[3]:


imf = emd.sift.sift(x)
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')


# In[4]:


freq_range = (0.1, 10, 80, 'log')
f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
emd.plotting.plot_imfs(imf, scale_y=True, cmap=True)


# In[18]:


rng = np.random.default_rng()
fs = 10e3

N = 1e5

amp = 2 * np.sqrt(2)

noise_power = 0.01 * fs / 2

time = np.arange(N) / float(fs)

mod = 500*np.cos(2*np.pi*0.25*time)

carrier = amp * np.sin(2*np.pi*3e3*time + mod)

noise = rng.normal(scale=np.sqrt(noise_power),
                   size=time.shape)

noise *= np.exp(-time/5)

x = carrier + noise
imf = emd.sift.sift(x)
IP, IF, IA = emd.spectra.frequency_transform(imf, 10000, 'hilbert')
freq_range = (0.1, 5000, 100)
f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False, sample_rate=10000)
print(f)
emd.plotting.plot_imfs(imf, scale_y=True, cmap=True, sample_rate = 10000)


# In[21]:



#print(f)
fig = plt.figure(figsize=(10, 6))
emd.plotting.plot_hilberthuang(hht, time, f,
                               fig=fig)


# In[6]:



f, t, Zxx = signal.stft(x, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=amp, shading='gouraud')

plt.title('STFT Magnitude')

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

plt.show()


# In[ ]:




