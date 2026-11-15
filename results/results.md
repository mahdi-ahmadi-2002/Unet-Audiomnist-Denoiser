Test set: 3000 utterances from 6 held-out speakers, synthetic noise, 16000 Hz.

| System | SI-SDR (dB) | SI-SDR improvement (dB) | SNR (dB) | STOI | Parameters | Real-time factor (CPU) |
|---|---:|---:|---:|---:|---:|---:|
| Noisy input | 6.00 | +0.00 | 5.71 | 0.883 | - | - |
| Spectral subtraction | 10.92 | +4.93 | 11.04 | 0.894 | - | 0.003 |
| Wiener filter | 11.33 | +5.33 | 11.56 | 0.886 | - | 0.013 |
| U-Net (complex mask) | 16.91 | +10.92 | 17.19 | 0.947 | 9.07 M | 0.102 |
| U-Net (magnitude mask) | 16.35 | +10.36 | 16.62 | 0.947 | 9.07 M | 0.097 |

**SI-SDR improvement (dB) by noise type**

| | Spectral subtraction | Wiener filter | U-Net (complex mask) | U-Net (magnitude mask) |
|---|---:|---:|---:|---:|
| babble | +0.02 | -0.02 | +3.85 | +3.94 |
| pink | +4.47 | +5.35 | +9.70 | +9.47 |
| brown | +3.93 | +6.89 | +15.86 | +14.99 |
| hum | +8.74 | +6.22 | +13.75 | +11.97 |
| white | +7.30 | +8.01 | +11.25 | +11.25 |

**SI-SDR improvement (dB) by input SNR (dB)**

| | Spectral subtraction | Wiener filter | U-Net (complex mask) | U-Net (magnitude mask) |
|---|---:|---:|---:|---:|
| -5 to 0 | +6.09 | +7.69 | +14.36 | +13.72 |
| 0 to 5 | +5.41 | +6.44 | +12.76 | +12.15 |
| 5 to 10 | +5.01 | +5.41 | +10.77 | +10.27 |
| 10 to 15 | +4.54 | +4.39 | +9.39 | +8.77 |
| 15 to 20 | +3.61 | +2.77 | +7.36 | +6.92 |

**U-Net (complex mask): SI-SDR improvement by gender**

| Group | Speakers | Utterances | Mean (dB) | Std (dB) |
|---|---:|---:|---:|---:|
| female | 2 | 1000 | +12.24 | 5.82 |
| male | 4 | 2000 | +10.25 | 5.41 |

**U-Net (complex mask): SI-SDR improvement by native speaker**

| Group | Speakers | Utterances | Mean (dB) | Std (dB) |
|---|---:|---:|---:|---:|
| no | 6 | 3000 | +10.92 | 5.63 |

**U-Net (complex mask): SI-SDR improvement by accent**

| Group | Speakers | Utterances | Mean (dB) | Std (dB) |
|---|---:|---:|---:|---:|
| chinese | 1 | 500 | +10.74 | 5.89 |
| german | 4 | 2000 | +10.93 | 5.64 |
| south korean | 1 | 500 | +11.04 | 5.32 |

**U-Net (complex mask): SI-SDR improvement by age**

| Group | Speakers | Utterances | Mean (dB) | Std (dB) |
|---|---:|---:|---:|---:|
| 25-34 | 4 | 2000 | +10.93 | 5.64 |
| 35 and over | 1 | 500 | +11.04 | 5.32 |
| under 25 | 1 | 500 | +10.74 | 5.89 |
