import cv2
import os.path as osp
import numpy as np
from PIL import Image
from typing import Union, Dict


class SegMetrics(object):
    def __init__(
        self,
        pred: Union[np.ndarray, str],
        label: Union[np.ndarray, str],
        num_classes: int,
    ) -> None:
        """Metrics about segmentation

        Args:
            pred (Union[np.ndarray, str]): The output's ndarray or path.
            label (Union[np.ndarray, str]): The label's ndarray or path.
            num_classes (int): The number of classes.
        """
        if num_classes <= 0:
            raise ValueError(
                "The `num_classes` must greater than 0, not be {}.".format(num_classes)
            )
        self.num_classes = num_classes
        self._pred = self._init_img(pred)
        self._label = self._init_img(label)
        self._init_colormap()
        self.eps = 2.2204e-16
        self.ConfusionMatrix = self.calc_confusion_matrix()

    def _init_img(self, img: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(img, str):
            if osp.exists(img):
                img = np.asarray(Image.open(img).convert("L"))
            else:
                raise FileNotFoundError("{} is not find.".format(img))
        if len(img.shape) == 3:
            C = img.shape[-1]
            if C == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif C == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        img = img.astype("int64")
        classes = np.unique(img)
        if len(classes) and np.max(classes) == 255 and np.min(classes) == 0:
            img = np.clip(img, 0, 1)
            classes = np.unique(img)
        if np.max(classes) >= self.num_classes:
            raise ValueError("The image's index out of `num_classes`.")
        return img

    def _init_colormap(self) -> None:
        self.color_map = []
        self.color_map.append([0, 0, 0])
        np.random.seed(666)
        for _ in range(1, self.num_classes):
            self.color_map.append(np.random.randint(0, high=255, size=(1, 3)))

    def _draw_img(self, img: np.ndarray) -> np.ndarray:
        W, H = img.shape
        pseudo_color = np.zeros((W, H, 3), dtype="uint8")
        for color, clas in zip(self.color_map, np.arange(self.num_classes)):
            pseudo_color[img == clas] = np.array(color)
        return pseudo_color

    @property
    def pred(self) -> np.ndarray:
        return self._draw_img(self._pred)

    @property
    def label(self) -> np.ndarray:
        return self._draw_img(self._label)

    def calc_confusion_matrix(self) -> np.ndarray:
        """ConfusionMatrix
        P\L    P      N
        P      TP     FP
        N      FN     TN
        """
        label_f = self._label.flatten()
        pred_f = self._pred.flatten()
        mask = (label_f >= 0) & (label_f < self.num_classes)
        label = self.num_classes * label_f[mask] + pred_f[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    @property
    def OA(self) -> float:
        oa = np.diag(self.ConfusionMatrix).sum() / (
            self.ConfusionMatrix.sum() + self.eps
        )
        return oa

    @property
    def Precision(self) -> np.ndarray:
        precision = np.diag(self.ConfusionMatrix) / (
            self.ConfusionMatrix.sum(axis=0) + self.eps
        )
        return precision

    @property
    def Recall(self) -> np.ndarray:
        recall = np.diag(self.ConfusionMatrix) / (
            self.ConfusionMatrix.sum(axis=1) + self.eps
        )
        return recall

    @property
    def FalseAlarm(self) -> np.ndarray:
        false_alarm = np.ones_like(self.Precision) - self.Precision
        return false_alarm

    @property
    def MissingAlarm(self) -> np.ndarray:
        missing_alarm = np.ones_like(self.Recall) - self.Recall
        return missing_alarm

    @property
    def F1(self) -> np.ndarray:
        f1 = (
            2 * self.Precision * self.Recall / (self.Precision + self.Recall + self.eps)
        )
        return f1

    @property
    def _intersection(self) -> np.ndarray:
        return np.diag(self.ConfusionMatrix)

    def _union(self, sub_diag: bool = True) -> np.ndarray:
        union = np.sum(self.ConfusionMatrix, axis=1) + np.sum(
            self.ConfusionMatrix, axis=0
        )
        if sub_diag:
            union -= np.diag(self.ConfusionMatrix)
        return union

    @property
    def IoU(self) -> np.ndarray:
        iou = self._intersection / (self._union() + self.eps)
        return iou

    @property
    def mIoU(self) -> float:
        miou = np.nanmean(self.IoU)
        return miou

    @property
    def FWIoU(self) -> float:
        freq = np.sum(self.ConfusionMatrix, axis=1) / (
            np.sum(self.ConfusionMatrix) + self.eps
        )
        iu = np.diag(self.ConfusionMatrix) / (self._union() + self.eps)
        fwiou = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwiou

    @property
    def Dice(self) -> np.ndarray:
        dice = (2 * self._intersection) / (self._union(False) + self.eps)
        return dice

    @property
    def mDice(self) -> float:
        mdice = np.nanmean(self.Dice)
        return mdice

    @property
    def Kappa(self) -> np.ndarray:
        pe_rows = np.sum(self.ConfusionMatrix, axis=0)
        pe_cols = np.sum(self.ConfusionMatrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / (float(sum_total**2) + self.eps)
        po = np.trace(self.ConfusionMatrix) / (float(sum_total) + self.eps)
        kappa = (po - pe) / (1 - pe + self.eps)
        return kappa

    @property
    def METRICS(self) -> Dict:
        return {
            "ConfusionMatrix": self.ConfusionMatrix,
            "OA": self.OA,
            "Precision": self.Precision,
            "Recall": self.Recall,
            "FalseAlarm": self.FalseAlarm,
            "MissingAlarm": self.MissingAlarm,
            "F1": self.F1,
            "IoU": self.IoU,
            "mIoU": self.mIoU,
            "FWIoU": self.FWIoU,
            "Dice": self.Dice,
            "mDice": self.mDice,
            "Kappa": self.Kappa,
        }
