# lấy pre-train model
import tensorflow as tf
import cv2
import imutils

img = cv2.imread('/home/minhpv/Downloads/115931252_3220428544686144_5731497845297843513_n.jpg')
img = cv2.resize(img, (300,100), interpolation = cv2.INTER_AREA)
print(img.shape)
img = tf.expand_dims(img, 0)
print(img.shape)
saved_model = tf.keras.models.load_model("/home/minhpv/Desktop/tools/classify/checkpoints/weights.999-0.00.hdf5")
rs = saved_model.predict(img, steps=1)
print(rs)