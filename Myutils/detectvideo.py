
import time
import threading
from queue import Queue
import cv2
from imutils.video import FPS
import numpy as np
from object_detection.utils import visualization_utils as viz_utils


def detectvideo_tovideo(inputvideo, mydetector, outputvideo):
    t3 = time.time()

    cap = cv2.VideoCapture('filesrc location='+inputvideo+' ! decodebin name=dec !   videoconvert  !  appsink', cv2.CAP_GSTREAMER)
    cap.get(cv2.CAP_PROP_FPS)
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    print("Frame Width:" + str(frame_width))
    print("Frame Height:" + str(frame_height))

    fps = FPS().start()
    out = cv2.VideoWriter(outputvideo ,cv2.VideoWriter_fourcc('M' ,'P' ,'4' ,'V'), 30, (frame_width ,frame_height))


    # read and insert frames in this queue
    que = Queue(maxsize=150)
    #detect objects, boxes and insert in this queue
    que1 = Queue(maxsize=150)

    # Flags to make video is processed till the end
    all_frames_read = False
    all_frames_object_detected = False
    currentframe = 0

    def read_frames():
        nonlocal fps
        nonlocal all_frames_read
        nonlocal que
        nonlocal cap
        nonlocal currentframe

        while cap.isOpened() and currentframe <100000:
            ret, image_np = cap.read()
            if len((np.array(image_np)).shape) == 0:
                break

            que.put(image_np, block=True)
            currentframe += 1

        all_frames_read = True

    def detect_obj():
        nonlocal out
        nonlocal all_frames_read
        nonlocal all_frames_object_detected
        nonlocal que
        nonlocal fps
        nonlocal que1

        while True:
            if que.empty() and all_frames_read:
                break
            try:
                image_np = que.get(block=True, timeout=0.1)
            except:
                continue

            input_tensor = np.expand_dims(image_np, 0)
            detections = mydetector.detect_fn(input_tensor)
            que1.put((image_np, detections), block=True)

        all_frames_object_detected = True

    def write_obj():

        nonlocal fps
        nonlocal out
        nonlocal que1
        nonlocal all_frames_object_detected

        while True:
            if que1.empty() and all_frames_object_detected:
                break
            try:
                image_np, detections = que1.get(block=True, timeout=0.1)
            except:
                continue

            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.int32),
                detections['detection_scores'][0].numpy(),
                mydetector.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.40,
                agnostic_mode=False)
            out.write(image_np_with_detections)

    th1 = threading.Thread(target=read_frames)
    th2 = threading.Thread(target=detect_obj)
    th3 = threading.Thread(target=write_obj)

    th1.start()
    th2.start()
    th3.start()

    th1.join()
    th2.join()
    th3.join()

    # stop the timer and display FPS information
    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    cap.release()
    out.release()

    t4 = time.time()
    print('Total Frames: ' + str(currentframe))
    print('Total Execution Time: ' + str(t4 -t3))

