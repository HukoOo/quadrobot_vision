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
        
        # get parameters
        self.rgb_topic = rospy.get_param("/label_image/rgb_topic", "/camera/color/image_raw")
              
        self.model_file = rospy.get_param("/label_image/model_file", rospack.get_path('quadrobot_vision')+'/model/output_graph.pb')
        self.label_file = rospy.get_param("/label_image/label_file", rospack.get_path('quadrobot_vision')+'/model/output_labels.txt')
        self.input_height = rospy.get_param("/label_image/input_height", "224")
        self.input_width = rospy.get_param("/label_image/input_width", "224")
              
        # init sub,pub topics
        self.image_sub = rospy.Subscriber(self.rgb_topic, Image,self.callback)
        self.image_pub = rospy.Publisher("/image_pub",Image)

        # init mobilenet graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=self.detection_graph, config=config) as self.sess:
                self.windowNotSet = True

    def callback(self,data):
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)
        
        t = self.read_tensor_from_cv(cv_image)

        input_operation = self.detection_graph.get_operation_by_name("input");
        output_operation = self.detection_graph.get_operation_by_name("final_result");
    
        results = self.sess.run(output_operation.outputs[0],
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
        
        cv2.namedWindow("Image window",0)
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)

        try:
          self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
          print(e)
        

    def read_tensor_from_cv(self, img, input_height=224,input_width=224,input_mean=128, input_std=128):  
      rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      resized_img = cv2.resize(rgb_img, (input_height, input_width))
      norm_img = cv2.normalize(resized_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
      image_tensor = np.expand_dims(norm_img, axis=0)

      return image_tensor



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
    
