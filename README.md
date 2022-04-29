# segMetrics
Metrics about segmentation based on Ptyhon.

## Install
```
pip install ssegmetrics
```

## Use
```python
from ssegmetrics import SampleMetrics

pred = "datas/pred.png"
label = "datas/label.png"
num_classes = 2

SampleMetrics(pred, label, num_classes)
```
