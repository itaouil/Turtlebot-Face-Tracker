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

        # Face classifier
        self.face_classifier = cv2.CascadeClassifier(haar_file_face)

    # Callback (process image and detects face)
    def callback(self, image_raw):

        # Convert image to OpenCV MAT type
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_raw, "bgr8")

        except CvBridgeError as e:
            rospy.logwarn("Error occured during image conversion: %s", e

        # Display original image
        if display_original_image:
            cv2.imshow("Original", image_raw)

        # Get those faces in the image
        self.detectAndDraw(cv_image)

    # Detect faces and draw them
    def detectAndDraw(self, cv_image):

        # Convert cv_image to greyscale
        image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

        # Draw rectangles over faces
        for (x,y,w,h) in faces:
            cv2.rectangle(cv_image, (x,y), (x+w,y+h), (255,0,0), 2)

        # Display tracking image
        if display_tracking_image:
            cv2.imshow("Tracking Image", cv_image)
