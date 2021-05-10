import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import streamlit as st
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import altair as alt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def video_detect(video_coming, threshold, prop):
    PROGRESS_BAR = st.progress(0)
    HEADER_DISPLAY1 = st.subheader("")
    HEADER_DISPLAY2 = st.text("")
    FRAME_WINDOW = st.image([], channels='BGR')
    DASHBOARD = st.subheader("")

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = prop['size']
    saved_model_loaded = tf.saved_model.load(prop['weights'], tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_coming))
    except:
        vid = cv2.VideoCapture(video_coming)

    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, 1)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
    
        # frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=prop['iou'],
            score_threshold=prop['score']
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        if prop['count']:
            # count objects found
            counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, prop['info'], counted_classes, allowed_classes=allowed_classes)
        else:
            image = utils.draw_bbox(frame, pred_bbox, prop['info'], allowed_classes=allowed_classes)
        
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)

        if len(counted_classes) == 0:
            HEADER_DISPLAY1.subheader("Prediction : " + "No cow detected")
        else:
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            HEADER_DISPLAY1.subheader("Count of {}s : {}".format(key, value))
            HEADER_DISPLAY2.text("Processing : {} FPS".format("{:.2f}".format(fps)))

        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        FRAME_WINDOW.image(result)
        if frame_num < 101:
            PROGRESS_BAR.progress(frame_num)
        histogramprep(pred_bbox,DASHBOARD)


def histogramprep(data,DASHBOARD,CHART):
    boxes, scores, classes, num_objects = data
    score_arr = []
    classes_arr = []
    count = 0
    for i in scores:
        if (i > 0):
            count += 1
            classes_arr.append(classes[0] + count)
            score_arr.append(i * 100)
    DASHBOARD.subheader('Dashboard')

    chart_data = pd.DataFrame()
    chart_data['Cows'] = classes_arr
    chart_data['Confidence Score %'] = score_arr

    st.altair_chart(alt.Chart(chart_data)
        .mark_bar(
        interpolate='step-after',
    ).encode(
        x=alt.X("Cows:Q", scale=alt.Scale(nice=True)),
        y=alt.Y("Confidence Score %:Q"),
        tooltip=['Cows', 'Confidence Score %']
    ).configure_mark(
        opacity=0.5,
        color='red'
    ), use_container_width=True)