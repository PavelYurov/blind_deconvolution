# Путеводитель по алгоритмам

## Blind Deconvolution (`blind_deconvolution/`)

Методы сгруппированы по типу источника кода - `our_company` и `third_party_company`

### Собственные реализации (`blind_deconvolution/our_company/`)

Имплементации на основе оригинальных научных публикаций, оптимизированные под фреймворк.

#### Основные алгоритмы (Core)
Методы, показавшие наиболее стабильные и качественные результаты восстановления.
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`htp`](blind_deconvolution/our_company/bayesian/htp) | Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors | 2013 | [Источник](<../../../references/htp/(2013) Blind Deconvolution Using Alternating Maximum a Posteriori Estimation with Heavy-Tailed Priors.pdf>) |
| [`fbdhsgp`](blind_deconvolution/our_company/bayesian/fbdhsgp) | Fast Bayesian blind deconvolution with Huber super Gaussian priors | 2017 | [Источники](../../../references/fbdhsgp) |
| [`bid_hbsp`](blind_deconvolution/our_company/bayesian/bid_hbsp) | Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior | 2024 | [Источник](<../../../references/bid_hbsp/(2024) Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior.pdf>) |


#### Дополнительные подходы (`blind_deconvolution/our_company/experimental_approaches`)
Реализации альтернативных подходов и исследовательские наработки.
| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`ard`](blind_deconvolution/our_company/experimental_approaches/bayesian/ard) | Blind Deconvolution with Model Discrepancies | 2017 | [Источник](<../../../references/ard/(2017) Blind Deconvolution with Model Discrepancies.pdf>) |
| [`сgmrf`](blind_deconvolution/our_company/experimental_approaches/bayesian/cgmrf) | Bayesian image restoration using compound Gauss-Markov random fields | 2003 | [Источники](<../../../references/cgmrf>) |
| [`eml`](blind_deconvolution/our_company/experimental_approaches/bayesian/eml) | Efficient Marginal Likelihood Optimization in Blind Deconvolution | 2011 | [Источники](<../../../references/eml>) |
| [`hsp`](blind_deconvolution/our_company/experimental_approaches/bayesian/hsp) | Bayesian Blind Image Deconvolution using a Hyperbolic-Secant Prior | 2024 | [Источники](<../../../references/hsp>) |
| [`rcs`](blind_deconvolution/our_company/experimental_approaches/bayesian/rcs) | Removing Camera Shake from a Single Photograph | 2006 | [Источник](<../../../references/rcs/(2006) Removing Camera Shake from a Single Photograph.pdf>) |
| [`vbskb_bid_sp`](blind_deconvolution/our_company/experimental_approaches/bayesian/vbskb_bid_sp) | Variational Bayesian Sparse Kernel-Based Blind Image Deconvolution With Student's-t Priors | 2009 | [Источники](<../../../references/vbskb_bid_sp>) |
| [`vdbke`](blind_deconvolution/our_company/experimental_approaches/bayesian/vdbke) |  Variational Dirichlet Blur Kernel Estimation | 2015 | [Источник](<../../../references/vdbke/(2015) Variational Dirichlet Blur Kernel Estimation.pdf>) |
| [`bdgsp`](blind_deconvolution/our_company/experimental_approaches/bayesian/bdgsp) | Bayesian Blind Deconvolution with General Sparse Image Priors | 2012 | [Источник](<../../../references/bdgsp/(2012) Bayesian Blind Deconvolution with General Sparse Image Priors.pdf>) |
| [`fractional_order`](blind_deconvolution/our_company/experimental_approaches/fractional_order) | Blind Image Deconvolution - When Patch-wise Minimal Pixels Prior Meets Fractional-Order Method | 2025 | [Источник](<../../../references/fractional_order/(2025) Blind Image Deconvolution - When Patch-wise Minimal Pixels Prior Meets Fractional-Order Method.pdf>) |







### Внешние обёртки и источники (`blind_deconvolution/third_party_company/`)

Модуль для алгоритмов, заимствованных из внешних репозиториев и адаптированных под фреймворк.

## Non-Blind Deconvolution (`nonblind_deconvolution/`)

## PSF Estimation (`kernel_estimation/`)

Папка зарезервирована под алгоритмы оценки ядра (PSF/kernel estimation); сейчас в репозитории пустая.

## Модификации (`mod_denoise/`)
Методы, использованнные в качестве модификации методов слепой деконволюции для оценки и подавления влияния шума и улучшения качества восстановления.

| Код | Метод / Название статьи | Год | Литература |
| :--- | :--- | :---: | :--- |
| [`act`](mod_denoise/act_denoise.py) | Compressive Sensing Image Restoration Using Adaptive Curvelet Thresholding and Nonlocal Sparse Regularization | 2016 | [Источник](<../../../references/act/(2016) Compressive Sensing Image Restoration Using Adaptive Curvelet Thresholding and Nonlocal Sparse Regularization.pdf>) |
| [`pyatykh` / `pca`](mod_denoise/pyatykh_noise_reconstruction.py) | Image Noise Level Estimation by Principal Component Analysis | 2014 | [Источник](<../../../references/pyatykh/(2014) Image Noise Level Estimation by Principal Component Analysis.pdf>) |
| [`screenot`](mod_denoise/screenot.py) | ScreeNOT Exact MSE-optimal singular value thresholding in correlated noise | 2023 | [Источник](<../../../references/screenot/(2023) ScreeNOT Exact MSE-optimal singular value thresholding in correlated noise.pdf>) |
| [`chen`](mod_denoise/chen_noise_estimate.py) | An Efficient Statistical Method for Image Noise Level Estimation | 2015 | [Источник](<../../../references/chen/(2015) An Efficient Statistical Method for Image Noise Level Estimation.pdf>) |
| [`vst`](mod_denoise/vst.py) | Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise | 2013 | [Источник](<../../../references/vst/(2013) Optimal inversion of the generalized Anscombe transformation for Poisson-Gaussian noise.pdf>) |


## Super Resolution (`kernel_estimation/`)


