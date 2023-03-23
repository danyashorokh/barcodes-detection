
import yolov5


class ObjectDetector(object):

    def __init__(self, weights, device, threshold=0.4, iou_threshold=0.45, max_det=300):

        self.weights = weights
        self.device = device
        self.prob_threshold = threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        # load model
        self.model = yolov5.load(self.weights)
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', self.weights, device=self.device)

        self.classes = self.model.names

        # set model parameters
        self.model.conf = self.prob_threshold  # NMS confidence threshold
        self.model.iou = self.iou_threshold  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = self.max_det  # maximum number of detections per image

    def detect(self, image, size=1280, show=False):

        results = self.model(image, size=size)

        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        classes = predictions[:, 5]

        if show:
            results.show()

        return boxes, scores, classes
