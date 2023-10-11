from absl import app
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
import streamlit as st

# Researchers
researchers = [
    "Abdul Wahid Rukua, S.Mat",
    "Yopi Andry Lesnussa, S.Si, M.Si",
    "Dorteus Lodewyik Rahakbauw, S.Si, M.Si"
]

# Geser language discussion
geser_language_discussion = """
... (potongan kode sebelumnya)
"""

st.set_page_config(page_title="Sigma Ai | Aplikasi Penerjemah Bahasa Geser", page_icon="🤖")
st.title("Demo NMT Indonesia|Geser")

interpreter = tf.lite.Interpreter('model_terbaru.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def translate(user_input):
    input_shape = input_details[0]['shape']
    input_data = np.array([user_input], dtype=np.str)
    return input_data

# Initialize translation result
translated_result = ""

# Form to add your items
with st.form("my_form"):
    user_input = st.text_area("Masukan Kata...", max_chars=200)

# Display the translation result while typing
    if user_input:
        input_data = translate(user_input)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        translated_result = output_data[0][0]
        st.write("Hasil Terjemahan")
        st.info(translated_result)

# Display researchers' information
st.header("Researchers:")
for researcher in researchers:
    st.write("- " + researcher)

# Display Geser language discussion with left-right alignment
st.header("Bahasa Geser:")
st.text(geser_language_discussion)
        
