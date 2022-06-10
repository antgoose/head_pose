import os
import sys
import streamlit as st
from PIL import Image
from src.demo import head_pose


sys.path.append("../")

root_dir = os.path.dirname(__file__)
#weights = root_dir + "/model_float16_quant.tflite"
#class_names = root_dir + "/class_names.txt"


def main():
    st.header("Head pose demo")
    st.write("Upload the image for head pose detection:")

    file = st.file_uploader("Choose an image...")

    if file is not None:
        img = Image.open(file)
        st.image(img, "your image")
        result = head_pose(img, root_dir)
        st.image(result, "resulting image after detecting")


if __name__ == "__main__":
    main()