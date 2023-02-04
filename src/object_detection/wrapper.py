
import numpy as np
import torch

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size, cv2, non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device


class ObjectDetectionWrapper(object):

    def __init__(self, weights, device, threshold=0.4, iou_threshold=0.45, max_det=300, imgsz=640):

        self.weights = weights
        self.device = select_device(device)
        self.prob_threshold = threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.max_det = max_det
        self.agnostic_nms = False  # class-agnostic NMS

        # load model
        self.model = DetectMultiBackend(weights=self.weights, device=self.device, dnn=False, data=None, fp16=False)
        self.stride, self.pt = self.model.stride, self.model.pt
        self.classes = self.model.names
        self.input_shape = [check_img_size(imgsz=self.imgsz, s=self.stride)] * 2  # convert size to [size, size]
        self.model.warmup(imgsz=(1, 3, *self.input_shape))  # warmup

    def letterbox(
        self,
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleFill=False,
        scaleup=True,
        stride=32,
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def detect(self, image):

        im = self.letterbox(image, new_shape=self.imgsz, stride=self.stride, auto=self.pt)[0]

        # preprocessing
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        # from detect.py
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im)
        pred = non_max_suppression(
            pred,
            conf_thres=self.prob_threshold,
            iou_thres=self.iou_threshold,
            classes=None,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
        )

        # # Process predictions
        # for det in pred:  # per image
        det = pred[0]

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], image.shape).round()

            xyxys = det[:, 0:4].numpy().astype(int)
            scores = det[:, 4].numpy().astype(float)
            classes = det[:, 5].numpy().astype(int)

            return xyxys, scores, classes

        return np.array([]), np.array([]), np.array([])
