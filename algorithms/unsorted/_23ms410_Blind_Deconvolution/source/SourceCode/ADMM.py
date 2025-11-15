from PIL import Image
from ctypes import *
import ctypes.util  
from numpy.ctypeslib import ndpointer 
import numpy as np
from numpy.random import *

def WF(img,width,height,lib,_DOUBLE_PP,tp):
  size1 = width
  size2 = height
  rs_wf = np.zeros((size2,size1))
  Z = 1
  lib.WeightFunction.argtypes = [_DOUBLE_PP, _DOUBLE_PP,c_int32, c_int32, c_int32]
  lib.WeightFunction.restype = None
  img_mpp = (img.__array_interface__['data'][0] + np.arange(img.shape[0])*img.strides[0]).astype(tp)
  rs_wf_mpp = (rs_wf.__array_interface__['data'][0] + np.arange(rs_wf.shape[0])*rs_wf.strides[0]).astype(tp)
  size1 = ctypes.c_int(size1)
  size2 = ctypes.c_int(size2)
  Z = ctypes.c_int(Z)
  lib.WeightFunction(img_mpp,rs_wf_mpp,size2,size1,Z)
  
  return rs_wf
 
def conv2D(img1,img2,height,width,len_psf,lib,_DOUBLE_PP,tp):
  size1 = height
  size2 = width
  size3 = len_psf
  rs_conv = np.zeros((size1,size1))
  lib.conv2d.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP,c_int32, c_int32, c_int32]
  lib.conv2d.restype = None
  img1_mpp = (img1.__array_interface__['data'][0] + np.arange(img1.shape[0])*img1.strides[0]).astype(tp)
  img2_mpp = (img2.__array_interface__['data'][0] + np.arange(img2.shape[0])*img2.strides[0]).astype(tp)
  rs_conv_mpp = (rs_conv.__array_interface__['data'][0] + np.arange(rs_conv.shape[0])*rs_conv.strides[0]).astype(tp)
  size1 = ctypes.c_int(size1)
  size2 = ctypes.c_int(size2)
  size3 = ctypes.c_int(size3)
  lib.conv2d(rs_conv_mpp,img1_mpp,img2_mpp,size1,size2,size3)
  
  return rs_conv
 
def CG(img_y,img_c,width,height,len_psf,lib,_DOUBLE_PP,tp):
  size_y_height = height
  size_y_width = width
  size_rs = len_psf
  size_rs2 = size_rs*size_rs
  rs_CG = np.zeros((size_rs,size_rs))
  itr = 30
  lib.Conjugate.argtypes = [_DOUBLE_PP, _DOUBLE_PP, _DOUBLE_PP, c_int32, c_int32, c_int32, c_int32]
  lib.Conjugate.restype = None
  img_y_mpp = (img_y.__array_interface__['data'][0] + np.arange(img_y.shape[0])*img_y.strides[0]).astype(tp)
  img_c_mpp = (img_c.__array_interface__['data'][0] + np.arange(img_c.shape[0])*img_c.strides[0]).astype(tp)
  rs_CG_mpp = (rs_CG.__array_interface__['data'][0] + np.arange(rs_CG.shape[0])*rs_CG.strides[0]).astype(tp)
  size_y_height = ctypes.c_int(size_y_height)
  size_y_width = ctypes.c_int(size_y_width)
  size_rs = ctypes.c_int(size_rs)
  size_rs2 = ctypes.c_int(size_rs2)
  itr = ctypes.c_int(itr)
  lib.Conjugate(img_y_mpp,img_c_mpp,rs_CG_mpp,size_y_width,size_y_height,size_rs2,itr)
  
  return rs_CG
 
def PSF_MAP(psf,len_m,len_n,len_psf,height,width,lib,_DOUBLE_PP,tp):
  int_len_m = len_m
  int_len_n = len_n
  int_len_psf = len_psf
  int_height = height
  int_width = width
  rs_psf_map = np.zeros((len_n,len_m))
  lib.psf_map.argtypes = [_DOUBLE_PP, _DOUBLE_PP,c_int32, c_int32, c_int32, c_int32, c_int32]
  lib.psf_map.restype = None
  psf_mpp = (psf.__array_interface__['data'][0] + np.arange(psf.shape[0])*psf.strides[0]).astype(tp)
  rs_psf_map_mpp = (rs_psf_map.__array_interface__['data'][0] + np.arange(rs_psf_map.shape[0])*rs_psf_map.strides[0]).astype(tp)
  int_len_m = ctypes.c_int(int_len_m)
  int_len_n = ctypes.c_int(int_len_n)
  int_len_psf = ctypes.c_int(int_len_psf)
  int_height = ctypes.c_int(int_height)
  int_width = ctypes.c_int(int_width)
  lib.psf_map(psf_mpp,rs_psf_map_mpp,int_len_m,int_len_n,int_len_psf,int_height,int_width)
  
  return rs_psf_map

def soft_threshold(x,L,p):
  t = L/p
   
  positive_indexes = x >= t
  negative_indexes = x <= t
  zero_indexes = abs(x) <= t
  
  print(positive_indexes)
  print(negative_indexes)
   
  y = np.zeros(x.shape)    
  y[positive_indexes] = x[positive_indexes] - t
  y[negative_indexes] = x[negative_indexes] + t
  y[zero_indexes] = 0.0
  
  return y

def ADMM(old_r,old_y,g,A,B,u,lambd,rho):
  inv_matrix = np.linalg.inv(np.dot(A.T, A) * u + np.dot(B.T, B) * rho)
  new_f = np.dot(inv_matrix,(np.dot(A.T,g) * u + np.dot(B.T,old_r) * rho - (np.dot(old_y.T,B)).T))
  x = np.dot(B,new_f) + old_y / rho
  new_r = soft_threshold(x,lambd,rho)
  new_y = old_y + (np.dot(B,new_f) - new_r) * rho
  
  return new_f, new_r, new_y
  
def main():
  lib = np.ctypeslib.load_library("ADMM_FUNCTION.so",".")
  _DOUBLE_PP = ndpointer(dtype=np.uintp, ndim=1, flags='C')
  tp = np.uintp
  
  lambd = 1.0
  rho = 1.0
  u = 100.0
  len_height = 64
  len_width = 64
  itr = 100
  Laplacian_Filter = np.array([[1.0,1.0,1.0],[1.0,-8.0,1.0],[1.0,1.0,1.0]])
  count = 0
  
  img_g = open('shepp-logan_64_15_2.img','rb')
  g = np.fromfile(img_g,dtype=float)
  g = g
  g_2 = np.reshape(g,(len_height,len_width))
  
  img_psf_0 = open('psf_2_15.img','rb')
  psf_0 = np.fromfile(img_psf_0,dtype=float)
  psf_0 = np.reshape(psf_0,(15,15))
  psf_0 = psf_0
  
  f_0 = np.zeros(len_height*len_width)
  A = PSF_MAP(psf_0,len(f_0),len(g),len(psf_0),len_height,len_width,lib,_DOUBLE_PP,tp)
  
  f_0 = np.dot(A.T,g)
  
  r_0 = np.zeros(len(f_0))
  
  y_0 = np.zeros(len(f_0))
  
  old_f = f_0
  old_f_2 = np.reshape(old_f,(len_height,len_width))
  old_y = y_0
  psf = psf_0
  W = WF(old_f_2,len_width,len_height,lib,_DOUBLE_PP,tp)
  
  A = PSF_MAP(psf,len(old_f),len(g),len(psf),len_height,len_width,lib,_DOUBLE_PP,tp)
  B = PSF_MAP(Laplacian_Filter,len(old_f),len(old_f),len(Laplacian_Filter),len_height,len_width,lib,_DOUBLE_PP,tp)
  print(B)
  W = np.reshape(W,(len(f_0),1))
  print(W)
  B = W*B
  print(B)
  
  r_0 = np.dot(B,f_0)
  
  old_r = r_0
  
  for i in range(itr):
    (new_f, new_r, new_y) = ADMM(old_r,old_y,g,A,B,u,lambd,rho)
    
    old_f = new_f
    old_f_2 = np.reshape(old_f,(len_height,len_width))
    old_r = new_r
    old_y = new_y
    
    Lap_g = conv2D(g_2,Laplacian_Filter,len_height,len_width,len(Laplacian_Filter),lib,_DOUBLE_PP,tp)
    Lap_f = conv2D(old_f_2,Laplacian_Filter,len_height,len_width,len(Laplacian_Filter),lib,_DOUBLE_PP,tp)
    psf = CG(Lap_g,Lap_f,len_width,len_height,len(psf_0),lib,_DOUBLE_PP,tp)
    W = WF(old_f_2,len_width,len_height,lib,_DOUBLE_PP,tp)
    A = PSF_MAP(psf,len(old_f),len(g),len(psf),len_height,len_width,lib,_DOUBLE_PP,tp)
    B = PSF_MAP(Laplacian_Filter,len(old_f),len(old_f),len(Laplacian_Filter),len_height,len_width,lib,_DOUBLE_PP,tp)
    W = np.reshape(W,(len(f_0),1))
    B = W*B
    
    filename = 'test_{0}.raw'.format(i)
    old_f_2.tofile(filename)
    
    count = count + 1
    print(count)

  print(old_f_2)
  
main()





