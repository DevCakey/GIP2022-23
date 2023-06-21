import os
import time
from typing import Sequence

import time
from Adafruit_IO import Client,Feed,RequestError
import cv2
import fire
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from motpy import Detection, ModelPreset, MultiObjectTracker, NpImage
from motpy.core import setup_logger
from motpy.detector import BaseObjectDetector
from motpy.testing_viz import draw_detection, draw_track
from motpy.utils import ensure_packages_installed

from coco_labels import get_class_ids

ensure_packages_installed(['torch', 'torchvision', 'cv2'])

logger = setup_logger(__name__, 'DEBUG', is_main=True)


class CocoObjectDetector(BaseObjectDetector):
    """ A wrapper of torchvision example object detector trained on COCO dataset """

    def __init__(self,
                 class_ids: Sequence[int],
                 confidence_threshold: float = 0.5,
                 architecture: str = 'ssdlite320',
                 device: str = 'cpu'):

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.class_ids = class_ids
        assert len(self.class_ids) > 0, f'select more than one class_ids'

        if architecture == 'ssdlite320':
            self.model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
        elif architecture == 'fasterrcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise NotImplementedError(f'unknown architecture: {architecture}')

        self.model = self.model.eval().to(device)

        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def _predict(self, image):
        image = self.input_transform(image).to(self.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(image)

        outs = [pred[0][attr].detach().cpu().numpy() for attr in ['boxes', 'scores', 'labels']]

        sel = np.logical_and(
            np.isin(outs[2], self.class_ids),  # only selected class_ids
            outs[1] >= self.confidence_threshold)  # above confidence threshold

        return [outs[idx][sel].astype(to_type) for idx, to_type in enumerate([float, int, float])]

    def process_image(self, image: NpImage) -> Sequence[Detection]:
        t0 = time.time()
        boxes, scores, class_ids = self._predict(image)
        elapsed = (time.time() - t0) * 1000.
        logger.debug(f'inference time: {elapsed:.3f} ms')
        return [Detection(box=b, score=s, class_id=l) for b, s, l in zip(boxes, scores, class_ids)]


def get_webcam():
    cap = cv2.VideoCapture(0)
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    return cap, video_fps

def run(
        video_downscale: float = 1.,
        architecture: str = 'ssdlite320',
        confidence_threshold: float = 0.5,
        tracker_min_iou: float = 0.25,
        show_detections: bool = False,
        track_text_verbose: int = 0,
        device: str = 'cpu',
        viz_wait_ms: int = 1):
    # setup detector, video reader and object tracker
    class_ids=get_class_ids(['person'])
    print("Using class ids:")
    print(class_ids)
    detector = CocoObjectDetector(class_ids=class_ids, confidence_threshold=confidence_threshold, architecture=architecture, device=device)
    cap, cap_fps = get_webcam()
    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': 5},
        model_spec={'order_pos': 1, 'dim_pos': 2,
                    'order_size': 0, 'dim_size': 2,
                    'q_var_pos': 5000., 'r_var_pos': 0.1},
        matching_fn_kwargs={'min_iou': tracker_min_iou,
                            'multi_match_min_iou': 0.93})
    
    IOKey = "aio_VQOC25asRdP6V7dFxNYjyyYyxwO7"
    IOUserName = "KemelEmiel"

    ada = Client(IOUserName,IOKey)

    try: 
        analog = ada.feeds('gip')
    except RequestError: 
        feed = Feed(name='gip')
        analog = ada.create_feed(feed)
    count = 0
    globalCount = 0
    trackList = []
    secondTrackList = []
    contains = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, fx=video_downscale, fy=video_downscale, dsize=None, interpolation=cv2.INTER_AREA)

        # detect objects in the frame
        detections = detector.process_image(frame)

        # track detected objects
        detectedobjects = tracker.step(detections=detections)
        if len(detectedobjects) > 0:
            print(f"det: {len(detectedobjects)}  count: {count}")
            if len(detectedobjects) > count:
                index = len(detectedobjects)-count-1
                addition = detectedobjects[index][0]
                trackList.append(addition)
                print(f"TrackList = {trackList}")
                count = len(detectedobjects)
        elif count > len(detectedobjects):
            print("removed")
            for i in range(len(trackList)):
                x = trackList[i]
                print(detectedobjects)
                print(f"x: {x}")
                for z in range(len(detectedobjects)):
                    contains = False
                    y = detectedobjects[z][0]
                    if x == y:
                        print("contains")
                        contains = True
            if contains == False:
                trackList.pop(i)
                lostIndex = i
            count  = len(detectedobjects)
            if secondTrackList[lostIndex][1][0]< 20:
                globalCount += 1
                print(f"GlobalCount = {globalCount}")
                ada.send_data('gip', globalCount)
            elif secondTrackList[lostIndex][1][0]> 500:
                print(f"GlobalCount = {globalCount}")
                if globalCount == 0:
                    ada.send_data('gip', globalCount)
                else:
                    globalCount -= 1

        secondTrackList = detectedobjects
        active_tracks = tracker.active_tracks(min_steps_alive=3)
        # visualize and show detections and tracks
        if show_detections:
            for det in detections:
                draw_detection(frame, det)

        for track in active_tracks:
            draw_track(frame, track, thickness=2, text_at_bottom=True, text_verbose=track_text_verbose)

        cv2.imshow('frame', frame)
        c = cv2.waitKey(viz_wait_ms)
        if c == ord('q'):
            break


if __name__ == '__main__':
    fire.Fire(run)