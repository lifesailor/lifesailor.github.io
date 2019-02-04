---
title: "Signal Processing(2) - Fourier Transform"
categories:
  - signal processing
tags:
  - signal porcessing

---

회사에서 진동 데이터로 장비 고장을 예지하는 업무를 하고 있습니다. 신호 처리는 진동 데이터를 분석하기 위한 초석이 됩니다. 생소한 분야지만 하나씩 정리하면서 스스로 배움을 얻고자 합니다. [Coursera](https://www.coursera.org/learn/dsp)와  [Udemy](https://www.udemy.com/signal-processing/)의 신호처리 강의를 참고했습니다. 

<br/>

## 1. Two major use of fourier transform

- Spectral 분석
  - 몇몇 신호는 주파수 도메인에서 더 잘 이해할 수 있다.
- 신호처리의 수단
  - Convolution Theroem을 활용해 Filtering과 Autocorrelation과 같은 신호 처리를 주파수 도메인에서 한다.

<br/>

## 2. Foundation of Fourier Transform

- Euler Formula
  - $e^{ik} = cosk + i sink$
    - $me^{ik} = m(cosk + isink)$에서 m은 amplitude, k는 phase이다.

```python
# define k
k = 1/3 * np.pi

# Euler's notation
euler = np.exp(1j*k)

# plot dot
plt.plot(np.cos(k), np.sin(k),'ro')
#plt.plot(np.real(euler), np.imag(euler), 'ro') - 위와 결과가 같다.

# draw unit circle for reference
x = np.linspace(-np.pi,np.pi,num=100)
plt.plot(np.cos(x),np.sin(x))

# some plotting touch-ups
plt.axis('square')
plt.grid(True)
plt.xlabel('Real axis'), plt.ylabel('Imaginary axis')
plt.show()
```

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/signal-processing/euler-formula.png">
</p>

<br/>

- complex sine wave
  - $e^{ik} = cosk + isink$에서 $k=2\pi ft + \theta$
  - 실수부에서는 cos 함수, 허수부에서는 sin 함수를 모두 가지고 있음

```python
# complex sine waves
# general simulation parameters
srate = 500; # sampling rate in Hz
time  = np.arange(0.,2.,1./srate) # time in seconds

# sine wave parameters
freq = 3;    # frequency in Hz
ampl = 2;    # amplitude in a.u.
phas = np.pi/3; # phase in radians

# generate the complex sine wave
csw = ampl * np.exp( 1j* (2*np.pi * freq * time + phas) );

# plot the results
plt.plot(time,np.real(csw),label='real')
plt.plot(time,np.imag(csw),label='imag')
plt.xlabel('Time (sec.)')
plt.ylabel('Amplitude')
plt.title('Complex sine wave projections')
plt.legend()
plt.show()
```

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/signal-processing/complex-sine-wave.png">
</p>

<br/>

아래와 같이 3D에서 보면 complex sine wave 를 한 눈에 볼 수 있음

```python
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(time,np.real(csw),np.imag(csw))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Real part')
ax.set_zlabel('Imag part')
ax.set_title('Complex sine wave in all its 3D glory')
plt.show()
```



<p align="center">
    <img src="https://lifesailor.github.io/assets/images/signal-processing/complex-sine-wave-2.png">
</p>

<br/>

- complex dot product
  - real dot product: 같은 크기의 두 벡터의 유사성을 나타내는 지표이다.
    - 다른 주기의 sine wave간의 내적은 0이다.
    - 한 신호에서 $\pi$/2 만큼 움직이면 내적은 0이 된다.
    - 즉, real dot product 값은 phase마다 달라진다. 즉, 특정 신호와 어떤 Phase에서 내적을 했는가에 따라서 값이 달라진다. 하지만 fourier transform의 목적은 phase에 상관 없이 각 진동수마다 얼마의 에너지를 가지고 있는지 판단하는 것이다. 따라서 real dot product는 적절하지 않다.
  - complex dot product
    - complex dot product의 크기는 $real^{2} + imag^{2}$ 이다.
    - complex dot product는 dot product 의 단점을 보완한다.
    - complex dot product는 phase마다 신호와의 내적이 일정하다. 즉, 각 주파수마다의 에너지를 나타낼 수 있다.



이를 확인하기 위해서 다음을 보자. 먼저 아래는 실수 내적을 한 결과이다. phase를 변경하면 진동수 영역을 나타내는 두 번째 사진이 달라진다.

```python
# phase: phase를 변경하면 
theta = 2*np.pi/4;

# simulation parameters
srate = 1000;
time  = np.arange(-1.,1.,1./srate)

# signal
sinew  = np.sin(2*np.pi * 5*time + theta)
gauss  = np.exp( (-time**2) / .1);
signal = np.multiply(sinew, gauss)

# sine wave frequencies
sinefrex = np.arange(2.,10.,.5);

# plot signal
plt.plot(time,signal)
plt.xlabel('Time (sec.)') 
plt.ylabel('Amplitude (a.u.)')
plt.title('Signal')
plt.show()

# initialize dot products vector
dps = np.zeros(len(sinefrex));

# loop over sine waves
for fi in range(1,len(dps)):
    
    # create sine wave
    sinew = np.sin(2 * np.pi * sinefrex[fi] * time)
    
    # compute dot product
    dps[fi] = np.dot(sinew, signal) / len(time)

# and plot
plt.stem(sinefrex,dps)
plt.xlabel('Sine wave frequency (Hz)'), plt.ylabel('Dot product')
plt.title('Dot products with sine waves')
plt.show()
```

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/signal-processing/complex-dot-product-1.png">
</p>

반면에 위의 식에서 내적을 하는 부분만 complex sine wave와의 dot product로 변경하면 phase가 변해도 진동수 대역의 결과는 일정하다.

```python
# initialize dot products vector
dps = np.zeros(len(sinefrex));

# loop over sine waves
for fi in range(1,len(dps)):
    
    # create sine wave
    sinew = np.exp( 1j*2*np.pi*sinefrex[fi]*time )
    
    # compute dot product
    dps[fi] = np.abs( np.dot( sinew,signal ) / len(time) )

# and plot
plt.stem(sinefrex,dps)
plt.xlabel('Sine wave frequency (Hz)'), plt.ylabel('Dot product')
plt.title('Dot products with sine waves')
plt.show()
```

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/signal-processing/complex-dot-product-2.png">
</p>



## 3. Discrete Time Fourier Transform

- Discrete Time Fourier Transform은 다음과 같은 로직을 거쳐서 수행된다.
  - Loop over time points
    - Create complex sine wave with the same number of time points of the signal and a frequency defined by point - 1
    - Multiply the sine wave by complex Fourier coefficient
  - end
- Nyquist Frequency
  - 측정 가능한 가장 큰 진동수 - 1/2 sampling rate 이상의 신호
- Positive and negative frequency
  - 진동수 변환을 하면 Positive Frequency와 Negative Frequency가 둘 다 나타난다.
  - 둘은 Nyquist Frequency에 대해서 대칭이다.
  - 만약에 Negative Frequency를 고려하지 않으면 Fourier Transform을 하고 나서 진폭이 반이 된다. 따라서 이를 고려해야 한다.
- Amplitude vs Power Spectrum
  - $Power = Amplitude^{2}​$
- Passeval's Theorem
  - Conservation of enegy in time domain and in frequency domain are the same.

<br/>

다음은 DTFT를 수행하는 코드입니다.

```python
# create the signal
srate  = 1000 # hz
time   = np.arange(0,2.,1/srate)  # time vector in seconds
pnts   = len(time) # number of time points
signal = 2.5 * np.sin(2*np.pi*4*time ) + 1.5 * np.sin( 2*np.pi*6.5*time )

# prepare the Fourier transform
fourTime = np.array(np.arange(0,pnts))/pnts
fCoefs   = np.zeros(len(signal),dtype=complex)

for fi in range(0,pnts):
    
    # create complex sine wave
    csw = np.exp( -1j*2*np.pi*fi*fourTime )
    
    # compute dot product between sine wave and signal
    fCoefs[fi] = np.sum( np.multiply(signal,csw) )

# extract amplitudes
ampls = np.abs(fCoefs) / pnts
ampls[range(1,len(ampls))] = 2*ampls[range(1,len(ampls))]

# compute frequencies vector
hz = np.linspace(0,srate/2,num=math.floor(pnts/2)+1)

plt.stem(hz,ampls[range(0,len(hz))])
plt.xlim([0,10])
plt.show()
```

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/signal-processing/DTFT.png">
</p>

<br/>

## 4. The Discrete inverse Fourier Transform

- Discrete Time Fourier Transform은 다음과 같은 로직을 거쳐서 수행된다.
  - Loop over frequencies
    - Create complex sine wave with the same number of time points of the signal and a frequency defined by point - 1
    - Multiply the sine wave by complex Fourier coefficient
    - Sum the modulated sine wave together
  - end
  - Divide the result by N

다음은 위의 DTFT를 수행하는 신호에 역변환을 하는 코드입니다.

```python
# initialize time-domain reconstruction
reconSignal = np.zeros(len(signal));
for fi in range(0,pnts):
    
    # create coefficient-modulated complex sine wave
    csw = fCoefs[fi] * np.exp( 1j*2*np.pi*fi*fourTime )
    
    # sum them together
    reconSignal = reconSignal + csw


# divide by N
reconSignal = reconSignal/pnts

plt.plot(time,signal,label='original')
plt.plot(time,np.real(reconSignal),'r.',label='reconstructed')
plt.legend()
plt.show() 
```

<br/>

## 5. Fast Fourier Transform

- FFT는 위의 DFTF처럼 for 문을 대신에 matrix 연산을 한다.	
  - 하지만 계산이 빨라지는 것이지 Output은 바뀌지 않는다.
- FFT on matrices
  - 여러 채널의 신호를 한 번에 Fourier Transform 할 수 있다.
- DTFT, FFT 모두 Fourier Transform 과정에서 Phase 및 허수 값을 버리면 역변환을 할 수 없다.

```python
srate = 1000
time  = np.arange(0,2,1/srate)
npnts = len(time)

# signal
signal = 2*np.sin(2*np.pi*6*time)

# Fourier spectrum
signalX = scipy.fftpack.fft(signal) 
hz = np.linspace(0,srate,npnts)

# amplitude
ampl = 2 * np.abs(signalX[0:len(hz)]) / npnts

plt.stem(hz,ampl)
plt.xlim(0,10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (a.u.)')
plt.show()
```

<p align="center">
    <img src="https://lifesailor.github.io/assets/images/signal-processing/FFT.png">
</p>

