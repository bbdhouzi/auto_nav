#!/usr/bin/env python

import rospy
# from nav_msgs.msg import Odometry
# from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
# from tf.transormations import euler_from_quaternion
# from geometry_msgs.msg import Twist
# import math
# import cmath
import numpy as np
import time

laser_range = np.array([])
# occdata = []
# front_angle = 30
# front_angles = range(-front_angle,front_angle+1,1)

front_angle = 5
front_angles = range(-front_angle,front_angle+1,1)

def get_laserscan(msg):
	global laser_range
	laser_range = np.array([msg.ranges])

def mover():
	global laser_range
	rospy.init_node('mover', anonymous=True)
	rospy.Subscriber('scan', LaserScan, get_laserscan)

	rate = rospy.Rate(5)
	time.sleep(1)
	while not rospy.is_shutdown():
		rospy.loginfo('hi')
		lr2 = laser_range[0]
		# lr20
		# rospy.loginfo(lr2)
		if lr2[0] <= 1.0:
			rospy.loginfo('1 meter')
		rate.sleep()

if __name__ == '__main__':
	try:
		mover()
	except rospy.ROSInterruptException:
		pass