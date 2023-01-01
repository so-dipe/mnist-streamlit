import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

model = tf.keras.models.load_model('mnist.v.0.2.h5')


def make_prediction(image):
  pred = model.predict(image.reshape(1, 28, 28, 1))[0]
  print(pred)
  return pd.Series(pred).rename('digit')


# def pred_bar(pred):
#   fig, ax = plt.subplots()
#   pred.plot(kind='bar')
#   return fig


def visualize_layers(image):
    image = image.reshape((1, ) + image.shape)
    image /= 255.0
    successive_outputs = [layer.output for layer in model.layers]
    visualization_model = tf.keras.models.Model(inputs=model.input,
                                              outputs=successive_outputs)
    successive_feature_maps = visualization_model.predict(image)
    layer_names = [layer.name for layer in model.layers]
    # figs = []
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]
            # size = feature_map.shape[1]
            fig, axs = plt.subplots(1, n_features, figsize=(15, 5))
            plt.grid(False)
            plt.axis(False)
            st.subheader(layer_name)
            for i in range(0, n_features):
                x = feature_map[0, :, :, i]
                axs[i].imshow(x, cmap='gray')
                axs[i].axis(False)
                st.pyplot(fig)
                plt.title(layer_name)
      # figs.append(fig)
    return None


def preprocess_image(image):
  return tf.image.resize(image, [28, 28],
                         method=tf.image.ResizeMethod.BILINEAR,
                         preserve_aspect_ratio=False,
                         antialias=False).numpy()[:, :, 0].reshape(28, 28, 1)
