"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import numpy as np

def convert_amp_phase(data):
  """Utility function to extract the manitude and phase from complex data

  Args:
      data (np.complex): raw complex data from BE spectroscopies

  Returns:
      np.array: returns the magnitude and the phase
  """
  magnitude = np.abs(data)
  phase = np.angle(data)
  return magnitude, phase