#importações libs
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model


class Gradcam:
    
    def __init__(self, model:str, size:tuple):
        """
        Args:
            model (string): model path
            size (tuple): image size (int,int)
        """
        self.model = load_model(model)
        self.size = size
        for layer in self.model.layers:
            if isinstance(layer, layers.Conv2D):
                self.conv_layer = layer.name
  
    def display(self, img_path:str, pred_index=None,alpha=0.4):
        """Merge heatmap and the original image into one image

        Args:
            img_path (string): Imagr path
            pred_index (int, optional): Index of the desired activation class. Defaults to None.
            alpha (float, optional): transparency index. Defaults to 0.4.
        """
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(self.conv_layer).output, self.model.output]
        )
        img = keras.preprocessing.image.load_img(img_path, target_size=self.size)
        array = keras.preprocessing.image.img_to_array(img)
        array = preprocess_input(np.expand_dims(array, axis=0))
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        img = keras.preprocessing.image.load_img(img_path)
        img = keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        imgplot = plt.imshow(superimposed_img)
        plt.show()