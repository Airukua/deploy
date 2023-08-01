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
Bahasa Seram(Geser) adalah salah satu bahasa yang dituturkan oleh masyarakat di Kepulauan Geser yang merupakan salah satu daerah di Kabupaten Seram Bagian Timur (SBT), Provinsi Maluku. Menurut Ethnologue (edisi ke-18, 2015) : Bahasa geser berasal dari rumpun bahasa Austronesia, yang di kelompokan sebagai rumpun bahasa Melayu-Polinesia Inti, kemudian di kelompokan lagi dalam rumpun bahasa Maluku Tengah atau lebih spesifik lagi ke dalam rumpun bahasa Maluku Tengah Timur. Pada tahun 1989 jumlah penutur dari bahasa geser adalah 36.500 penutur.
Bahasa geser di golongkan oleh Kantor Bahasa Provinsi Maluku sebagai bahasa yang terancam punah, ini di karenakan bahasa tersebut tidak di pakai sebagai sarana komunikasi utama. Bahasa geser tidak digunakan di ranah pemerintahan, sekolah, atau bahkan keluarga. Menurut pakar bahasa, keadaan ini akan membuat Bahasa Geser mengalami perubahan sedikit demi sedikit hingga akhirnya punah (Khairiyah, 2017).
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

# Display researchers' information
st.header("Researchers:")
for researcher in researchers:
    st.write("- " + researcher)

# Display Geser language discussion with left-right alignment
st.header("Bahasa Geser:")
st.text(geser_language_discussion)
