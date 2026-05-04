# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

# NOTE:
# This file is adapted from the original N-BEATS implementation (Oreshkin et al., 2020).
# Only minor non-functional adjustments have been made for integration with FANBEATS.

import numpy as np
import torch as t

def default_device() -> t.device:

    return t.device('cuda' if t.cuda.is_available() else 'cpu')

def to_tensor(array: np.ndarray) -> t.Tensor:
 
    return t.tensor(array, dtype=t.float32).to(default_device())

def divide_no_nan(a, b):

    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result