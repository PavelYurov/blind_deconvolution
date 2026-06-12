"""
Denoise Algorithm Module

Pure image denoising wrapper algorithm with automatic noise estimation
and support for 7 different denoiser backends.
"""

from .Denoise import DenoiseWrapper

__all__ = ['DenoiseWrapper']
