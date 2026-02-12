# Теоретические основы

```{admonition} Раздел в разработке
:class: warning

Данный раздел находится в активной разработке и будет заполнен в следующих релизах.
```

## Планируемые материалы

### Слепая деконволюция: введение

- Математическая модель: \( g = h \ast f + n \)
- Определение и свойства PSF (Point Spread Function)
- Виды искажений: смаз движения, расфокусировка, атмосферные турбулентности

### Классические методы

- Richardson-Lucy (метод максимального правдоподобия)
- Метод Expectation-Maximization (EM)
- MAP (Maximum a Posteriori) с регуляризацией

### Вариационные методы

- VABID (Variational Approach to Blind Image Deconvolution)
- VAPIBE (Variational Approach to Point Image Blind Estimation)
- Методы Total Variation (TV) регуляризации

### Байесовские методы

- VBBID-TV (Variational Bayesian Blind Image Deconvolution with Total Variation)
- BBD-DEIP (Bayesian Blind Deconvolution)
- SB-BID-PE (Sparse Bayesian Blind Image Deconvolution)

### Разреженные методы

- VBC-BID (Variational Bayesian Compressive Blind Image Deconvolution)
- PBTVGR (Proximal methods)

### Метрики качества

- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- SML (Sum of Modified Laplacian)
- Мера сложности смаза
- Мера сложности шума

### Оптимизация гиперпараметров

- Байесовская оптимизация (TPE, Gaussian Processes)
- Многокритериальная оптимизация (фронт Парето, NSGA-II)

---

*Для получения дополнительной информации обратитесь к исходным публикациям, указанным в docstrings реализаций.*
