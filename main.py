import streamlit as st
from streamlit_drawable_canvas import st_canvas
import predict

st.title('Handwriting Recogniction')

# Specify canvas parameters in application
drawing_mode = 'freedraw'
stroke_width = '28'
stroke_color = "#fff"
bg_color = '#000'
# realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
with st.sidebar:
    canvas_result = st_canvas(
        # fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=True,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key="full_app",
    )
  

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    with st.sidebar:
        st.image(canvas_result.image_data)
    processed_image = predict.preprocess_image(canvas_result.image_data)
    st.subheader('Probability of Each Digit')
    tab1, tab2 = st.tabs(["📈 Bar Chart", "🗃 Data"])
    prediction = predict.make_prediction(processed_image)
    tab1.bar_chart(prediction)
    tab2.write(prediction)
    predict.visualize_layers(processed_image)
  
