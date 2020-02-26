#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import UInt16MultiArray
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

cur_frame = None
kernel = np.ones((10,10), np.uint8)
bridge = CvBridge()

def image_cb(data):
    global cur_frame
    cur_frame = bridge.imgmsg_to_cv2(data, "bgr8")

def main():
    global cur_frame
    rospy.init_node('target_detector', anonymous=True)
    rospy.Subscriber('img_topic', Image, image_cb)
    pub = rospy.Publisher('target_loc', UInt16MultiArray, queue_size=10)

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        if cur_frame is not None:
            hsv_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2HSV)
            lower_range = np.array([100,100,100])
            upper_range = np.array([140,255,255])

            mask = cv2.inRange(hsv_frame, lower_range, upper_range)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(c)
                center = (x+(w//2), y+(h//2))

                cur_frame = cv2.rectangle(cur_frame, (x,y), (x+w,y+h), (0,0,255), 2)
                cur_frame = cv2.rectangle(cur_frame, (center[0]-1, center[1]-1), (center[0]+1, center[1]+1), (0,255,255), 2)
                rospy.loginfo(center)
                center_msg = UInt16MultiArray(data=[center[0], center[1]])
                pub.publish(center_msg)
            
            cv2.imshow("cur frame", cur_frame)
            cv2.imshow('mask', mask)

            cv2.waitKey(3)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()