#!/usr/bin/env python

import rospy
import cv2

from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Vector3
from cv_bridge import CvBridge, CvBridgeError

"""
    The following class subscribes to
    the Turtlebot's camera input and
    carries some computer vision using
    haar like feature to detect faces.
"""

class Face_Detector:

    # Constructor
    def __init__(self):

        # ROS/OpenCV bridge
        self.bridge = CvBridge()

        # Image transportation
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.image_pub = rospy.Publisher("/face_centroid")

        # Config variables
        screenmaxx              = ""
        center_offset           = ""
        face_tracking           = ""
        haar_file_face          = ""
        input_image_topic       = ""
        output_image_topic      = ""
        display_original_image  = ""
        display_tracking_image  = ""

        # Load config file
        try:
            rospy.get_param("input_image_topic", input_image_topic)
            rospy.get_param("output_image_topic", output_image_topic)
            rospy.get_param("haar_file_face", haar_file_face)
            rospy.get_param("face_tracking", face_tracking)
            rospy.get_param("display_original_image", display_original_image)
            rospy.get_param("display_tracking_image", display_tracking_image)
            rospy.get_param("center_offset", center_offset)
            rospy.get_param("screenmaxx", screenmaxx)

            rospy.loginfo("Configuration file loaded !")

        except Exception as e:
            rospy.logwarn("Error during config loading: %s", e)

    # Callback (process image and detects face)
    def callback(self, image_raw):

        # Convert image to OpenCV MAT type
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_raw, "bgr8")

        except CvBridgeError as e:
            rospy.logwarn("Error occured during image conversion: %s", e)

        # Show image
        cv2.namedWindow("Original Image")
        cv2.imshow("Original Image", cv_image)
        cv2.waitKey(5)
