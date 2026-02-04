from dataclasses import dataclass
from typing import Optional

import numpy as np


# * Model Data
@dataclass
class ModelData:
    index: Optional[float] = np.nan
    status_index: Optional[float] = 0
