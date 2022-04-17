import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from .metrics import SegMetrics


def SampleMetrics(
    pred: Union[np.ndarray, str], label: Union[np.ndarray, str], num_classes: int
) -> None:
    sm = SegMetrics(pred, label, num_classes)
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(sm.pred)
    plt.title("Pred")
    plt.subplot(122)
    plt.imshow(sm.label)
    plt.title("Label")
    plt.show()
    print("======= Metrics =======")
    for k, v in sm.METRICS.items():
        print("◆ {0}\n{1}".format(k, v))
