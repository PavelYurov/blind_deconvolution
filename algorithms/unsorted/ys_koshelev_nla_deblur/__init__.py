from algorithms.base import DeconvolutionAlgorithm
import numpy as np
from typing import Any, Dict
from .source.deblur import blind_deconv
from .source.latent_step import synth_kernel, im_normalize
from .source.utils import whiten_background


class YsKoshelevNlaDeblur(DeconvolutionAlgorithm):
    """
    Обёртка над реализацией алгоритма из папки source.

    Структура исходников и сигнатуры функций в source/ остаются без изменений,
    обёртка лишь подготавливает данные и параметры и вызывает `blind_deconv`.
    """

    def __init__(
        self,
        *,
        lambda_: float = 4e-3,
        gamma: float = 2560.0,
        sigma: float = 1.0,
        beta_max: float = 2.0**3,
        mu_max: float = 1e5,
        wei_grad: float = 4e-3,
        lambda_tv: float = 0.003,
        lambda_l0: float = 1e-3,
        kernel_size: int = 25,
        num_iter: int = 50,
        params: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__("ys_koshelev_nla_deblur")

        # Параметры по умолчанию взяты из apply.py
        self.params: Dict[str, Any] = {
            "lambda_": lambda_,
            "gamma": gamma,
            "sigma": sigma,
            "beta_max": beta_max,
            "mu_max": mu_max,
            "wei_grad": wei_grad,
            "lambda_tv": lambda_tv,
            "lambda_l0": lambda_l0,
            "kernel_size": kernel_size,
            "num_iter": num_iter,
        }

        if params is not None:
            self.change_param(params)

    def change_param(self, param: Dict[str, Any]):
        """
        Обновление гиперпараметров алгоритма.
        Ожидается словарь с ключами как в self.params.
        """
        if not isinstance(param, dict):
            return
        for key, value in param.items():
            if key in self.params:
                self.params[key] = value

    def process(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Запуск слепой деконволюции.

        Аргументы:
            image: размытое изображение (numpy массив, шкала 0..255 или 0..1).

        Возвращает:
            восстановленное изображение и оценённое ядро смаза.
        """
        from time import time

        if image.ndim != 2:
            # исходный код работает с одним каналом, поэтому при необходимости берём яркость
            image = image.astype(np.float32)
            image = image.mean(axis=2)

        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # в исходном apply.py используется обрезка полей изображения
        img_cropped = img[10:-15, 10:-5]

        p = self.params

        k0 = synth_kernel(int(p["kernel_size"]))

        t1 = time()
        latent, kernel = blind_deconv(
            img_cropped,
            k0,
            int(p["num_iter"]),
            float(p["lambda_"]),
            float(p["sigma"]),
            float(p["gamma"]),
            float(p["beta_max"]),
            float(p["mu_max"]),
            float(p["wei_grad"]),
            float(p["lambda_tv"]),
            float(p["lambda_l0"]),
        )
        t2 = time()
        self.timer = t2 - t1

        latent = whiten_background(latent)
        latent = im_normalize(latent, 0, 1)

        return latent.astype(np.float32), kernel.astype(np.float32)

    def get_param(self):
        """
        Возвращает список (имя, значение) для всех гиперпараметров,
        как ожидает базовая инфраструктура проекта.
        """
        return list(self.params.items())


__all__ = ["YsKoshelevNlaDeblur"]
