import altair as alt
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from detection import detect_image


@st.cache(show_spinner=True)
def read_img(img):
    img_array = np.array(img)
    image = cv2.cvtColor(img_array, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]  # BGR -> RGB
    return image


def histogramprep(data):
    boxes, scores, classes, num_objects = data
    score_arr = []
    classes_arr = []
    count = 0
    for i in scores:
        if (i > 0):
            count += 1
            classes_arr.append(classes[0] + count)
            score_arr.append(i * 100)
    st.subheader('Dashboard')

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


def processimages(uploaded_file, prop):
    image = Image.open(uploaded_file)
    image = read_img(image)
    col1, col2 = st.beta_columns(2)
    original = Image.open(uploaded_file)
    col1.subheader("Actual")
    st.image(original, use_column_width=True)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.01)
    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    pred_image, counted_classes, data = detect_image.detector([np.array(original.convert('RGB'))], confidence_threshold, prop)
    if len(counted_classes) == 0:
        st.subheader("Prediction : " + "No cow detected")
    else:
        for key, value in counted_classes.items():
            print("Number of {}s: {}".format(key, value))
        st.subheader("Count of {}s : {}".format(key, value))
    # Display the final image
    st.image(cv2.cvtColor(pred_image.astype(np.uint8), cv2.COLOR_BGR2RGB), use_column_width=True)
    histogramprep(data)
