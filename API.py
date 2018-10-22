# Copyright 2018 Gang Wei wg0502@bu.edu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


class use_API():

  file_name = "/Users/gangwei/Desktop/daisy-flower-1532449822.jpg"
  model_file = "/tmp/output_graph.pb"
  label_file = "/tmp/output_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "Placeholder"
  output_layer = "final_result"

  def __init__(self):
    pass

  def load_image_path(self,image_path):
    use_API.file_name = image_path

  def load_model_path(self,model_path):
    use_API.model_file = model_path

  def load_label_path(self,model_path):
    use_API.label_file = model_path

  def load_graph(self,model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

  def read_tensor_from_image_file(self,file_name,
                                  input_height=299,
                                  input_width=299,
                                  input_mean=0,
                                  input_std=255):
    if not os.path.exists(file_name):
      print("Wrong file path, please enter a correct one")
      exit()
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
      image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

  def load_labels(self,label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

  def go(self):
    graph = use_API.load_graph(self,use_API.model_file)
    t = use_API.read_tensor_from_image_file(self,use_API.file_name,
      input_height=use_API.input_height,
      input_width=use_API.input_width,
      input_mean=use_API.input_mean,
      input_std=use_API.input_std)

    input_name = "import/" + use_API.input_layer
    output_name = "import/" + use_API.output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
        input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = use_API.load_labels(self,use_API.label_file)
    for i in top_k:
      print(labels[i], results[i])