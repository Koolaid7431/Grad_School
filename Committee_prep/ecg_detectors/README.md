

# ECG Detectors
A collection of 7 ECG hearbeat detection algorithms implemented in Python. Developed in conjunction with a new ECG database: http://researchdata.gla.ac.uk/716/.

[github repository link](https://github.com/luishowell/ecg-detectors)

## Class Initalisation
Before the detectors can be used the class must first be initalised with the sampling rate of the ECG recording
```python
detectors = Detectors(fs)
```
## Detectors

### Hamilton
Implementation of P.S. Hamilton, “Open Source ECG Analysis Software Documentation”, E.P.Limited, 2002.\
Usage:
```python
r_peaks = detectors.hamilton_detector(unfiltered_ecg)
```

### Christov
Implementation of Ivaylo I. Christov, “Real time electrocardiogram QRS detection using combined adaptive threshold”, BioMedical Engineering OnLine 2004, vol. 3:28, 2004.\
Usage:
```python
r_peaks = detectors.christov_detector(unfiltered_ecg)
```

### Engelse and Zeelenberg
Implementation of W. Engelse and C. Zeelenberg, “A single scan algorithm for QRS detection and feature extraction”, IEEE Comp. in Cardiology, vol. 6, pp. 37-42, 1979 with modifications A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred, “Real Time Electrocardiogram Segmentation for Finger Based ECG Biometrics”, BIOSIGNALS 2012, pp. 49-54, 2012.\
Usage:
```python
r_peaks = detectors.engzee_detector(unfiltered_ecg)
```


### Pan and Tompkins
Implementation of Jiapu Pan and Willis J. Tompkins. “A Real-Time QRS Detection Algorithm”. In: IEEE Transactions on Biomedical Engineering BME-32.3 (1985), pp. 230–236.\
Usage:
```python
r_peaks = detectors.pan_tompkins_detector(unfiltered_ecg)
```

### Stationary Wavelet Transform
Implementation based on Vignesh Kalidas and Lakshman Tamil. “Real-time QRS detector using Stationary Wavelet Transform for Automated ECG Analysis”. In: 2017 IEEE 17th International Conference on Bioinformatics and Bioengineering (BIBE). Uses the Pan and Tompkins thresolding method.\
Usage:
```python
r_peaks = detectors.swt_detector(unfiltered_ecg)
```

### Two Moving Average
Implementation of Elgendi, Mohamed & Jonkman, Mirjam & De Boer, Friso. (2010). "Frequency Bands Effects on QRS Detection" The 3rd International Conference on Bio-inspired Systems and Signal Processing (BIOSIGNALS2010). 428-431.
Usage:
```python
r_peaks = detectors.two_average_detector(unfiltered_ecg)
```

### Matched Filter
FIR matched filter using template of QRS complex. Template provided for 250Hz and 360Hz. Uses the Pan and Tompkins thresolding method.
Usage:
```python
r_peaks = detectors.matched_filter_detector(unfiltered_ecg)
```
