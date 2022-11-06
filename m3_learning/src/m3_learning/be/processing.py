"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import numpy as np

def convert_amp_phase(data, type_data='stacked'):
  magnitude = np.abs(data)
  phase = np.angle(data)
  return magnitude, phase