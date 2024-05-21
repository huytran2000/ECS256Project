import streamlit as st
from PIL import Image
import numpy as np
import torch
# import cv2
import io
# import os

import rcnn_utils

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_model():

    custom_yolov7_model = torch.hub.load(
        "WongKinYiu/yolov7", 'custom', './yolo/best.pt')

    return custom_yolov7_model


def get_prediction(img_bytes, model):

    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, size=640)

    return results


def analyse_image(img_file, model):

    if img_file is not None:

        img = Image.open(img_file)
        st.write("### Original image")
        st.write(img)
        # print(file_buffer)
        # print(uploaded_file.getvalue())
        st.write("### Running Yolov7")

        bytes_data = img_file.getvalue()
        # print(type(bytes_data))
        # print(bytes_data[:100])
        # img_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        result = get_prediction(bytes_data, model)
        result.imgs[0] = result.imgs[0].copy()
        result.render()

        st.image(result.imgs[0])
        print("Yolo Done.")

        # for img in result.imgs:
        #     RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     im_arr = cv2.imencode('.jpg', RGB_img)[1]
        # # st.image(im_arr.tobytes())

        # result_list = list((result.pandas().xyxy[0])["name"])

        st.write("### Running Faster RCNN")

        out_img = rcnn_utils.analyze_image(img)  # img byte form
        st.image(out_img)
        print("Faster RCNN Done")

    else:
        st.write("No image was detected!")
        # result_list = []

    return  # result_list


if __name__ == "__main__":
    model = load_model()

    st.write("## Welcome to Swimmer Detection app")
    # supports more img types?
    uploaded_file = st.file_uploader("Upload image here.", ['jpg'])

    analyse_image(uploaded_file, model)
