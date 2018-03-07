#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import tensorflow as tf
import cv2

import rospy
import rospkg

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_classifier:

    def __init__(self):

        self.bridge = CvBridge()
        
        self.rgb_topic = rospy.get_param("/label_image/rgb_topic", "/camera/color/image_raw")
              
        self.model_file = rospy.get_param("/label_image/model_file", rospack.get_path('quadrobot_vision')+'/model/output_graph.pb')
        self.label_file = rospy.get_param("/label_image/label_file", rospack.get_path('quadrobot_vision')+'/model/output_labels.txt')
        self.input_height = rospy.get_param("/label_image/input_height", "224")
        self.input_width = rospy.get_param("/label_image/input_width", "224")
      
        self.input_layer = "input"
        self.output_layer = "final_result"
        
        
        self.image_sub = rospy.Subscriber(self.rgb_topic, Image,self.callback)
        self.image_pub = rospy.Publisher("/image_pub",Image)
    
    def callback(self,data):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        
        graph = self.load_graph(self.model_file)
        t = self.read_tensor_from_cv(cv_image)
        
        input_name = "import/" + self.input_layer
        output_name = "import/" + self.output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);
    
        with tf.Session(graph=graph) as sess:
            results = sess.run(output_operation.outputs[0],
                        {input_operation.outputs[0]: t})
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = self.load_labels(self.label_file)
        top_label = top_k[0]
        for i in top_k:
          print(labels[i], results[i])
        print()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv_image, 'Type='+labels[top_label],(10,50), font,1 ,(0,0,0),2,cv2.LINE_AA)
        cv2.putText(cv_image,'Score='+str(results[top_label]),(10,100), font, 1,(0,0,0),2,cv2.LINE_AA)
        
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)

        try:
          self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
          print(e)
        

    def load_graph(self, model):
      graph = tf.Graph()
      graph_def = tf.GraphDef()

      with open(model, "rb") as f:
        graph_def.ParseFromString(f.read())
      with graph.as_default():
        tf.import_graph_def(graph_def)

      return graph

    def read_tensor_from_cv(self, img, input_height=224,input_width=224,input_mean=128, input_std=128):
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        float_caster = tf.cast(rgb_img, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def read_tensor_from_image_file(self, file_name, input_height=299, input_width=299,
				    input_mean=0, input_std=255):
      input_name = "file_reader"
      output_name = "normalized"
      file_reader = tf.read_file(file_name, input_name)
      if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels = 3,
                                           name='png_reader')
      elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
      elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
      else:
        image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                            name='jpeg_reader')
      float_caster = tf.cast(image_reader, tf.float32)
      dims_expander = tf.expand_dims(float_caster, 0);
      resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
      normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
      sess = tf.Session()
      result = sess.run(normalized)

      return result

    def load_labels(self, label_file):
      label = []
      proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
      for l in proto_as_ascii_lines:
        label.append(l.rstrip())
      return label
  

    
if __name__ == "__main__":
  rospy.init_node('label_image', anonymous=True)
  rospack = rospkg.RosPack()   
  sess=tf.Session()
  ic = image_classifier()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
    
  cv2.destroyAllWindows()    
    
