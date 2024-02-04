from ultralytics.utils import FRAME_SAMPLE
from ultralytics.models.yolo.classify.predict import ClassificationPredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.obb.predict import OBBPredictor
from ultralytics.models.yolo.pose.predict import PosePredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor

# classification
args = dict(model='yolov8n-cls.pt', source=FRAME_SAMPLE)
predictor = ClassificationPredictor(overrides=args)
predictor.predict_cli()

# detection
args = dict(model='yolov8n.pt', source=FRAME_SAMPLE)
predictor = DetectionPredictor(overrides=args)
predictor.predict_cli()

# oriented bounding box
args = dict(model='yolov8n-obb.pt', source=FRAME_SAMPLE)
predictor = OBBPredictor(overrides=args)
predictor.predict_cli()

# pose
args = dict(model='yolov8n-pose.pt', source=FRAME_SAMPLE)
predictor = PosePredictor(overrides=args)
predictor.predict_cli()

# segmentation
args = dict(model='yolov8n-seg.pt', source=FRAME_SAMPLE)
predictor = SegmentationPredictor(overrides=args)
predictor.predict_cli()