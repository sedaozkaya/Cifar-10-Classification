import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CIFAR-10 sınıf adları
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cifar10_gelismis_model.h5")

model = load_model()

st.title("CIFAR-10 Görsel Sınıflandırıcı")
st.write("Bir görsel yükleyin (32x32 RGB) ve sınıflandırma sonuçlarını görün.")

uploaded_file = st.file_uploader("Bir .jpg veya .png dosyası yükleyin", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((32, 32))
    st.image(image, caption='Yüklenen Görsel', use_column_width=False)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]  # tek örnek olduğu için [0] alındı
    top_indices = prediction.argsort()[-3:][::-1]  # en yüksek 3 tahmin (indis)

    st.write("🏆 Tahminler:")
    for rank, i in enumerate(top_indices, start=1):
        class_name = class_names[i]
        confidence = prediction[i] * 100
        st.write(f"{rank}. **{class_name}** (%{confidence:.1f})")
