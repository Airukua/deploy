#simpan model NMT
tf.saved_model.save(export, 'dynamic_translator',
                    signatures={'serving_default': export.translate})
#load model 
loaded = tf.saved_model.load('masukan direktori model', options=tf.saved_model.LoadOptions(experimental_io_device='/job:localhost'))
translate_func = loaded.signatures['serving_default']

# konversi model
converter = tf.lite.TFLiteConverter.from_saved_model('masukan direktori model')
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# simpan model yang telah dikonversi
with open("model_terbaru.tflite", "wb") as f:
    f.write(tflite_model)
