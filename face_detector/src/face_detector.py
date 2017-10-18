#!/usr/bin/env python

import rospy
import cv2

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
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

        # OpenCV bridge
        self.cv_bridge = CvBridge()

        # Camera image subscription
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.process)
