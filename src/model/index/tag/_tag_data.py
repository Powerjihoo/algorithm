from dataclasses import dataclass
from typing import Optional
import numpy as np
from types_ import Array

@dataclass
class TagData:
    """
    Data class representing information about a tag.

    Attributes:
        value (Optional[float]): Current value of the tag.
        pred (Optional[float]): Predicted value of the tag.
        status_pred (Optional[int]): Status of the predicted value.
        slope (Optional[float]): Slope of the tag.
        index (Optional[float]): Index calculated for the tag.
        status_index (Optional[int]): Status of the calculated index.
        ignore_status (Optional[bool]): Ignore status of the tag.
    """

    value: Optional[float] = np.nan
    pred: Optional[float] = np.nan
    status_pred: Optional[int] = 0
    slope: Optional[float] = np.nan
    index: Optional[float] = np.nan
    status_index: Optional[int] = 0
    ignore_status: Optional[bool] = False

    def update_lastdata(self, value:float, pred:float, status_pred:int, ignore_status:bool) -> None:
        """
        Update the tag data with new values and status for the prediction.

        Args:
            value (float): Current value of the tag.
            pred (float): Predicted value of the tag.
            status_pred (int): Status of the predicted value.
            ignore_status (bool): Ignore status of the tag

        Returns:
            None
        """
        self.value = value
        self.pred = pred
        self.status_pred = status_pred
        self.ignore_status = ignore_status
        
