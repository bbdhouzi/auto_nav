#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import math
import cmath
import numpy as np
import time
import tf2_ros

laser_range = np.array([])
occdata = []
yaw = 0.0
rotate_speed = 0.1
linear_speed = 0.01
stop_distance = 0.25
occ_bins = [-1, 0, 100, 101]
front_angle = 30
front_angles = range(-front_angle,front_angle+1,1)

occ_grid = []
current_pos = ()
checked_positions = []


def get_odom_dir(msg):
	global yaw

	orientation_quat =  msg.pose.pose.orientation
	orientation_list = [orientation_quat.x, orientation_quat.y, orientation_quat.z, orientation_quat.w]
	(roll, pitch, yaw) = euler_from_quaternion(orientation_list)


def get_laserscan(msg):
	global laser_range

	# create numpy array
	laser_range = np.array([msg.ranges])


def get_occupancy(msg, tf_buf):
	global occdata
	global current_pos
	global occ_grid

	# create numpy array
	occdata = np.array([msg.data])
	# compute histogram to identify percent of bins with -1
	occ_counts = np.histogram(occdata,occ_bins)
	# calculate total number of bins
	total_bins = msg.info.width * msg.info.height
	# log the info
	rospy.loginfo('Unmapped: %i Unoccupied: %i Occupied: %i Total: %i', occ_counts[0][0], occ_counts[0][1], occ_counts[0][2], total_bins)

	trans2 = tf_buf.lookup_transform('map', 'base_link', rospy.Time(0))
	cur_pos_pre = trans2.transform.translation
	map_res = msg.info.resolution
	map_origin = msg.info.origin.position
	current_pos = ((cur_pos_pre.x - map_origin.x)/map_res, (cur_pos_pre.y - map_origin.y)/map_res)
	rospy.loginfo(current_pos)

	oc2 = occdata + 1
	oc3 = (oc2>1).choose(oc2,2)
	occ_grid = np.uint8(oc3.reshape(msg.info.width,msg.info.height, order='F'))

def rotatebot(rot_angle):
	global yaw

	# create Twist object
	twist = Twist()
	# set up Publisher to cmd_vel topic
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
	# set the update rate to 1 Hz
	rate = rospy.Rate(1)

	# get current yaw angle
	current_yaw = np.copy(yaw)
	# log the info
	rospy.loginfo(['Current: ' + str(math.degrees(current_yaw))])
	# we are going to use complex numbers to avoid problems when the angles go from
	# 360 to 0, or from -180 to 180
	c_yaw = complex(math.cos(current_yaw),math.sin(current_yaw))
	# calculate desired yaw
	target_yaw = current_yaw + math.radians(rot_angle)
	# convert to complex notation
	c_target_yaw = complex(math.cos(target_yaw),math.sin(target_yaw))
	rospy.loginfo(['Desired: ' + str(math.degrees(cmath.phase(c_target_yaw)))])
	# divide the two complex numbers to get the change in direction
	c_change = c_target_yaw / c_yaw
	# get the sign of the imaginary component to figure out which way we have to turn
	c_change_dir = np.sign(c_change.imag)
	# set linear speed to zero so the TurtleBot rotates on the spot
	twist.linear.x = 0.0
	# set the direction to rotate
	twist.angular.z = c_change_dir * rotate_speed
	# start rotation
	pub.publish(twist)

	# we will use the c_dir_diff variable to see if we can stop rotating
	c_dir_diff = c_change_dir
	# rospy.loginfo(['c_change_dir: ' + str(c_change_dir) + ' c_dir_diff: ' + str(c_dir_diff)])
	# if the rotation direction was 1.0, then we will want to stop when the c_dir_diff
	# becomes -1.0, and vice versa
	while(c_change_dir * c_dir_diff > 0):
		# get current yaw angle
		current_yaw = np.copy(yaw)
		# get the current yaw in complex form
		c_yaw = complex(math.cos(current_yaw),math.sin(current_yaw))
		rospy.loginfo('While Yaw: %f Target Yaw: %f', math.degrees(current_yaw), math.degrees(target_yaw))
		# get difference in angle between current and target
		c_change = c_target_yaw / c_yaw
		# get the sign to see if we can stop
		c_dir_diff = np.sign(c_change.imag)
		# rospy.loginfo(['c_change_dir: ' + str(c_change_dir) + ' c_dir_diff: ' + str(c_dir_diff)])
		rate.sleep()

	rospy.loginfo(['End Yaw: ' + str(math.degrees(current_yaw))])
	# set the rotation speed to 0
	twist.angular.z = 0.0
	# stop the rotation
	time.sleep(1)
	pub.publish(twist)

def get_unmapped_coord():
	# global current_pos
	global checked_positions
	
	rospy.loginfo(['In get_unmapped_coord'])
	# try:
		# pos_to_check = [current_pos]
	# except UnboundLocalError:
		# rospy.loginfo('not defined yet bro')
		# return
	pos_to_check = [current_pos]
	for cur_pos in pos_to_check:
		# rospy.loginfo(cur_pos)
		i,j = cur_pos[0], cur_pos[1]
		for next_pos in [(i-1,j),(i,j+1),(i+1,j),(i,j-1)]:
			if occ_grid[next_pos] == 0:
				return next_pos
			elif occ_grid[next_pos] == 1:
				if next_pos not in checked_positions:
					pos_to_check.append(next_pos)
		checked_positions.append(cur_pos)

		if current_pos in pos_to_check:
			pos_to_check.remove(current_pos)
	return False
	rospy.loginfo('mapping complete')

def pick_direction():
	rospy.loginfo(['In pick_direction'])
	# publish to cmd_vel to move TurtleBot
	pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

	# stop moving
	twist = Twist()
	twist.linear.x = 0.0
	twist.angular.z = 0.0
	time.sleep(1)
	pub.publish(twist)

	next_coord = get_unmapped_coord()
	rospy.loginfo(next_coord)
	rospy.loginfo(['Picked direction: ' + str(next_coord)])

	rospy.loginfo('current_pos: '+ str(current_pos))
	rospy.loginfo('next_pos: '+ str(next_coord))
	angle = math.atan((next_coord[1]-current_pos[1])/(next_coord[0]-current_pos[0]))
	# rotate to that direction
	# rotatebot(float(lr2i))
	rotatebot(angle)

	# start moving
	rospy.loginfo(['Start moving'])
	twist.linear.x = linear_speed
	twist.angular.z = 0.0
	# not sure if this is really necessary, but things seem to work more
	# reliably with this
	time.sleep(1)
	pub.publish(twist)


def mover():
	global laser_range

	rospy.init_node('mover', anonymous=True)

	tf_buf = tf2_ros.Buffer()
	tf_listener = tf2_ros.TransformListener(tf_buf)
	rospy.sleep(1.0)

	# subscribe to odometry data
	rospy.Subscriber('odom', Odometry, get_odom_dir)
	# subscribe to LaserScan data
	rospy.Subscriber('scan', LaserScan, get_laserscan)
	# subscribe to map occupancy data
	rospy.Subscriber('map', OccupancyGrid, get_occupancy, tf_buf)

	pi_pub = rospy.Publisher('pi_chat', String, queue_size=10)

	rate = rospy.Rate(5) # 5 Hz

	# find direction with the largest distance from the Lidar
	# rotate to that direction
	# start moving
	pick_direction()

	while not rospy.is_shutdown():
		# check distances in front of TurtleBot
		lr2 = laser_range[0,front_angles]
		# distances beyond the resolution of the Lidar are returned
		# as zero, so we need to exclude those values
		lr20 = (lr2!=0).nonzero()
		# find values less than stop_distance
		lr2i = (lr2[lr20]<float(stop_distance)).nonzero()
		# rospy.loginfo(lr2i[0])

		# if the list is not empty
		if(len(lr2i[0])>0):
			rospy.loginfo(['Stop!'])
			# find direction with the largest distance from the Lidar
			# rotate to that direction
			# start moving
			pi_pub.publish('its time')
			pick_direction()

		rate.sleep()


if __name__ == '__main__':
	try:
		mover()
	except rospy.ROSInterruptException:	
		exit()
	except KeyboardInterrupt:
		exit()
