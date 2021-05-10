import streamlit as st
import tempfile
from core.helper import processimages
from detection.detect_video import video_detect

prop = dict(
    framework='tf',
    weights='yolov4-weights/',
    output='./output/',
    video='Dataset2_1.mp4',
    size=416,
    info=False,
    iou=0.45,
    score=0.50,
    count=True
)

st.sidebar.markdown("## San José State University")
st.sidebar.markdown("#### Department of Applied Data Science")
st.sidebar.markdown("#### MSDA Project II, Spring 2021")
st.sidebar.markdown("----------------------------------------")
st.sidebar.markdown("## Smart Farm Survelliance App")
st.sidebar.markdown("----------------------------------------")
# Appends an empty slot to the app. We'll use this later.
my_slot1 = st.empty()
# Appends another empty slot.
my_slot2 = st.empty()

# features
co1, co2, co3, co4 = st.beta_columns(4)
press_button1 = co1.checkbox('Cattle Detection & Counting')
press_button2 = co2.checkbox('Livestock Behaviour Monitoring')
press_button3 = co3.checkbox('Fire & Smoke Detection')
press_button4 = co4.checkbox('Object Detection')

if press_button1:
    st.sidebar.markdown("** Model :** Yolov4")
    st.sidebar.markdown("** Data Source :** Drone")
    imageFileTypeList=['png', 'jpeg', 'jpg']
    videoFileTypeList=['mp4']
    uploaded_file = st.file_uploader("Choose an image/video ...", type=imageFileTypeList.append(videoFileTypeList))
    if uploaded_file is not None:
        if uploaded_file.type.split('/')[1] in imageFileTypeList :
            processimages(uploaded_file, prop)
        else:
            confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.01)
            tmpfile = tempfile.NamedTemporaryFile(delete=True)
            tmpfile.write(uploaded_file.read())
            video_detect(tmpfile.name, confidence_threshold, prop)
            tmpfile.close()



elif press_button2:
    st.sidebar.empty()
    st.sidebar.markdown("** Model :** Yolov4")
    st.sidebar.markdown("** Data Source :** Camera")
elif press_button3:
    st.sidebar.markdown("** Model :** Fine Tuned VGG16")
    st.sidebar.markdown("** Data Source :** Satellite")
elif press_button4:
    st.sidebar.markdown("** Model :** Faster RCNN")
    st.sidebar.markdown("** Data Source :** Satellite")
else:
    # right panel
    st.sidebar.markdown("### Team 2")
    st.sidebar.markdown("* Kai Kwan Poon")
    st.sidebar.markdown("* Kiran Brar")
    st.sidebar.markdown("* Neha Singh")
    st.sidebar.markdown("* Praveena Manikonda")
    my_slot1.image('data/images/FarmImage.png', width=700)
    my_slot2.info("**Select feature: **")
