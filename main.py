import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import func as f


def set_page_config():
    front_image = Image.open("images/aksara-jawa.png")
    st.set_page_config(page_title="Aksara Jawa Image Classification")
    st.image(front_image, use_column_width=True)
    st.write(
        "<h2 style='text-align: center;'>Aksara Jawa Image Classification Demo!</h6>",
        unsafe_allow_html=True,
    )


def sidebar_config():
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)
    up_image = st.sidebar.file_uploader("Upload image:", type=["png", "jpg"])
    return stroke_width, stroke_color, bg_color, realtime_update, up_image


def create_canvas(stroke_width, stroke_color, bg_color, realtime_update):
    return st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=700,
        width=700,
        drawing_mode="freedraw",
        key="canvas",
    )


def display_results(up_image, canvas_result):
    st.divider()
    if up_image is not None:
        st.image(up_image)
        response = f.predict_func(up_image)
        st.code(response)
    elif canvas_result.image_data is not None:
        response = f.predict_func(canvas_result.image_data, from_numpy=True)
        st.code(response)


def main():
    set_page_config()
    stroke_width, stroke_color, bg_color, realtime_update, up_image = sidebar_config()
    canvas_result = create_canvas(stroke_width, stroke_color, bg_color, realtime_update)
    display_results(up_image, canvas_result)


if __name__ == "__main__":
    main()
