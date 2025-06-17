# A multivariate cloud workload prediction method integrating convolutional nonlinear spiking neural model with bidirectional long short-term memory

This is the pytorch implementation of the paper

### Abstract

Multivariate workload prediction in cloud computing environments is a critical research problem.  Effectively capturing inter-variable correlations and temporal patterns in multivariate time series is key to addressing this challenge. To address this issue, this paper proposes a convolutional model based on a Nonlinear Spiking Neural P System (ConvNSNP), which enhances the ability to process nonlinear data compared to conventional convolutional models. Building upon this, a hybrid forecasting model is developed by integrating ConvNSNP with a Bidirectional Long Short-Term Memory (BiLSTM) network. ConvNSNP is first employed to extract temporal and cross-variable dependencies from the multivariate time series, followed by BiLSTM to further strengthen long-term temporal modeling. Comprehensive experiments are conducted on three public cloud workload traces from Alibaba and Google. The proposed model is compared with a range of established deep learning approaches, including CNN, RNN, LSTM, TCN, and hybrid models such as LSTNet, CNN-GRU, and CNN-LSTM. Experimental results on three public datasets demonstrate that our proposed model achieves up to 9.9\% improvement in RMSE and 11.6\% improvement in MAE compared with the most effective baseline methods. The model also achieves favorable performance in terms of MAPE, further validating its effectiveness in multivariate workload prediction.

### DataSet

Google 2011、Google 2019、Alibaba 2020

### Code Structure

> ConvNSNP-BiLSTM
>
>  -data : Alibaba and Google datasets
>
>  -models : save the main model of ConvNSNP-BiLSTM and other compared methods.
>
>  -utils : the augmentation tools and DataLoader
>
> -main : the main file to train the model and test the model

### Environmental requirements

```
brotlipy                 0.7.0
certifi                  2022.12.7
cffi                     1.15.0
charset-normalizer       2.0.4
contourpy                1.1.1
cryptography             38.0.4
cycler                   0.12.1
einops                   0.8.0
flit_core                3.6.0
fonttools                4.53.1
idna                     3.4
importlib_resources      6.4.5
joblib                   1.4.2
kiwisolver               1.4.7
matplotlib               3.7.5
mkl-fft                  1.3.1
mkl-random               1.2.2
mkl-service              2.4.0
numpy                    1.23.5
nvidia-cuda-runtime-cu12 12.8.90
nvidia-pyindex           1.0.9
nvidia-tensorrt          99.0.0
packaging                24.1
pandas                   2.0.3
Pillow                   9.3.0
pip                      22.3.1
ptflops                  0.7.3
pycparser                2.21
pyOpenSSL                22.0.0
pyparsing                3.1.4
PySocks                  1.7.1
python-dateutil          2.9.0.post0
pytz                     2024.2
requests                 2.28.1
scikit-learn             1.3.2
scipy                    1.10.1
setuptools               65.6.3
simpy                    4.1.1
six                      1.16.0
tensorrt                 10.9.0.34
tensorrt-cu12            10.9.0.34
tensorrt_cu12_bindings   10.9.0.34
tensorrt_cu12_libs       10.9.0.34
threadpoolctl            3.5.0
torch                    1.13.1
torchaudio               0.13.1
torchvision              0.14.1
tqdm                     4.66.5
typing_extensions        4.4.0
tzdata                   2024.1
urllib3                  1.26.14
wheel                    0.38.4
zipp                     3.20.2
```

### Usage

```
python main.py -dataset (gc11/gc19 or ali20) -lr (0.0001/0.001)
```

### Result on Alibaba 2020 dataset

[![img](https://github.com/HelloDVA/ConvNSNP-BiLSTM/Alibaba2020.png)](https://github.com/HelloDVA/ConvNSNP-BiLSTM/Alibaba2020.png)

### Result on Google 2011 dataset

[![img](https://github.com/HelloDVA/ConvNSNP-BiLSTM/Google2011.png)](https://github.com/HelloDVA/ConvNSNP-BiLSTM/Google2011.png)

### Result on Google 2019 dataset

[![img](https://github.com/HelloDVA/ConvNSNP-BiLSTM/Google2019.png)](https://github.com/HelloDVA/ConvNSNP-BiLSTM/Google2019.png)

