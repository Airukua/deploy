import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from tensorflow.lite.python import interpreter
import streamlit as st

st.set_page_config(page_title="Sigma Ai | Aplikasi Penerjemah Bahasa Geser", page_icon="🤖")
st.title("Demo NMT Indonesia|Geser")

interpreter = tf.lite.Interpreter('converted_model.tflite')
interpreter.allocate_tensors()

    
# Form to add your items
with st.form("my_form"):
    user_input = st.text_area("Masukan Kata...", max_chars=200)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = [user_input]
    input_data = np.array(input_data, dtype=str)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_string = output_data[0].decode('utf-8')
    submitted = st.form_submit_button("Translate")

    if submitted:
        st.write("Hasil Terjemahan")
        st.info(output_string)
