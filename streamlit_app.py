from absl import app
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
import streamlit as st

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

# Form to add your items
with st.form("my_form"):
    user_input = st.text_area("Masukan Kata...", max_chars=200)

    # Run the model
    if st.form_submit_button("Translate"):
        st.write("Hasil Terjemahan")
        input_data = translate(user_input)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        st.info(output_data)
