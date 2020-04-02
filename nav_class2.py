#!/usr/bin/env python

import cv2
import numpy as np
import math
import cmath
import time

import rospy
from geometry_msgs.msg import Twist

def get_rot_coord(image, angle, pos):
	image_center = tuple(np.array(image.shape[1::-1])/2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	pos_mat = np.array([[pos[0]],[pos[1]], [1]])
	return rot_mat.dot(pos_mat)

class Navigation():
	def __init__(self, linear_spd, angular_spd):
		self.linear_spd = linear_spd
		self.angular_spd = angular_spd

		self.yaw = 0.0
		self.lidar_data = np.array([])
		self.bot_position = ()
		self.occ_grid = np.array([])

		self.map_origin = ()
		self.map_height = 0
		self.map_width = 0

		self.mapping_complete = False

		self.occ_map_raw = np.array([])
		self.occ_map = np.array([])
		self.checked_positions = []
		self.edges = []

		self.pseudo_route = []	# list of edges/unmapped regions followed
		self.main_route = []	# list of actual positions the bot will follow

		self.dist_to_trgt = 0
		self.prev_dist = 99999999

	# methods to update sensor data
	def update_yaw(self, data):
		self.yaw = data
	
	def update_laserscan_data(self, data):
		self.lidar_data = data
	
	def update_occ_grid(self, grid, origin):
		self.occ_grid = grid
		self.map_origin = origin
		self.map_height = len(grid)
		self.map_width = len(grid[0])
	
	def update_bot_pos(self, data):
		self.bot_position = data
	
	# returns the square of the distance between two points
	def get_dist_sqr(self, pos1, pos2):
		return (pos2[1] - pos1[1])**2 + (pos2[0] - pos1[0])**2

	# returns the actual distance in terms of pixels
	def get_dist(self, pos1, pos2):
		return math.sqrt(self.get_dist_sqr(pos1, pos2))

	# convert coordinate from x,y to i,j or vice versa
	def swap_xy(self, pos):
		return (pos[1], pos[0])
	
	# finds the nearest unmapped region
	# returns false if no unmapped regions found
	def get_nearest_unmapped_region(self):
		rospy.loginfo('[NAV][OCC] Finding nearest unmapped region')
		pos_to_check = [self.bot_position]
		self.checked_positions = []

		for cur_pos in pos_to_check:
			# i,j = cur_pos[0],cur_pos[1]
			i,j = cur_pos
			rospy.loginfo('[MAP] %s', str(cur_pos))
			for next_pos in [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]:
				if next_pos not in self.checked_positions:
					rospy.loginfo('[MAP] Checking pos %s', str(next_pos))
					if self.occ_grid[next_pos] == 0:
						rospy.loginfo('[NAV][OCC] Found nearest unmapped region at %s', str(next_pos))
						self.pseudo_route.insert(0, next_pos)
						return next_pos
					elif self.occ_grid[next_pos] == 1:
						pos_to_check.append(next_pos)
			self.checked_positions.append(cur_pos)

			if cur_pos in pos_to_check:
				pos_to_check.remove(cur_pos)
		
		rospy.loginfo('[NAV][OCC] No unmapped region found!')
		self.mapping_complete = True
		return False
	
	# update occ_map and occ_map_raw
	def update_map(self):
		ret, occ_map_raw = cv2.threshold(self.occ_grid, 2, 255, 0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
		self.occ_map_raw = cv2.dilate(occ_map_raw, element)
		self.occ_map = cv2.cvtColor(self.occ_map_raw, cv2.COLOR_GRAY2RGB)
	
	# find the edges of the visible walls
	def update_edges(self):
		self.update_map()
		self.edge_map = np.zeros((self.map_height, self.map_width, 3), np.uint8)
		occ_map_bw, contours_ret, hierarchy = cv2.findContours(self.occ_map_raw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours_ret:
			approx = cv2.approxPolyDP(contour, 0.009*cv2.arcLength(contour, True), True)
			n = approx.ravel()
			for a in range(0,len(n),2):
				i = n[a]
				j = n[a+1]
				self.edges.append((i,j))
				cv2.circle(self.edge_map, (i,j), 3, (255,0,0),-1)
		
		if len(self.edges) == 0:
			rospy.loginfo('[NAV][EDGE] No edges detected, there is likely an issue')
			self.display_edges()
	
	# find the nearest edge to a position
	def get_closest_edge(self, pos):
		self.update_edges()
		rospy.loginfo('[NAV][EDGE] Locating the nearest edge to %s', str(pos))
		distances = [self.get_dist_sqr(edge_pos, pos) for edge_pos in self.edges]
		rospy.loginfo('DISTANCES: ')
		rospy.loginfo(distances)
		if len(distances) == 0:
			rospy.loginfo('error: No edges detected?')
			return False
		else:
			return self.edges[distances.index(min(distances))]
	
	# check if there are any obstacles blocking the straight line path between two points
	def path_blocked(self, cur_pos, next_pos):
		self.update_map()
		path_img = np.zeros((self.map_height, self.map_width, 1), np.uint8)
		overlay_img = np.zeros((self.map_height, self.map_width, 1), np.uint8)
		cv2.line(path_img, self.swap_xy(cur_pos), self.swap_xy(next_pos), 255, thickness=1, lineType=8)
		cv2.circle(path_img, self.swap_xy(cur_pos), 3, 255, -1)

		cv2.circle(path_img, (next_pos[1], next_pos[0]), 3, 128, -1)

		ret, path_img_bit = cv2.threshold(path_img, 2, 255, 0)

		# overlay_map = np.zeros()
		overlay_map = cv2.bitwise_and(path_img_bit, self.occ_map_raw)
		# rospy.loginfo('[NAV][PATH] %s', str(overlay_map))

		overlay_img = cv2.bitwise_or(self.occ_map_raw, path_img)

		# cv2.imshow('path_check img', path_img)
		# cv2.imshow('occ_map', self.occ_map_raw)
		# cv2.imshow('overlay_map', overlay_map)
		# cv2.imshow('overlay_img', overlay_img)
		# cv2.waitKey(0)

		return np.any(overlay_map)
	
	def display_map(self, refresh=True):
		self.update_edges()

		map_overlay = np.zeros((self.map_height, self.map_width, 3), np.uint8)
		cv2.circle(map_overlay, self.swap_xy(self.bot_position), 3, (0,0,255), -1)

		unmapped_region = self.get_nearest_unmapped_region()
		cv2.circle(map_overlay, self.swap_xy(unmapped_region), 3, (128,128,255), -1)

		if refresh:
			closest_edge = self.get_closest_edge(self.bot_position)
			cv2.circle(map_overlay, closest_edge, 3, (0,255,0), -1)

		# map_overlay = cv2.bitwise_or(map_overlay, self.edges)
		occ_map_disp = cv2.bitwise_or(self.occ_map, map_overlay)

		cv2.imshow('occ_map', self.occ_map)
		# cv2.imshow('edges', self.edge_map)
		# cv2.imshow('map_overlay'. self.map_overlay)
		cv2.imshow('final', occ_map_disp)
		cv2.waitKey(0)
	
	def get_angle(self, cur_pos_o, next_pos_o):
		cur_pos = self.swap_xy(cur_pos_o)
		next_pos = self.swap_xy(next_pos_o)

		i_dist = next_pos[0] - cur_pos[0]
		j_dist = next_pos[1] - cur_pos[1]

		rospy.loginfo('[NAV][ANGLE] cur_pos: %s\tnext_pos: %s', str(cur_pos), str(next_pos))
		rospy.loginfo('[NAV][ANGLE] tan %d/%d', i_dist, j_dist)
		return math.atan2(i_dist, j_dist)

	# get angle need to turn, accounting for yaw
	def get_angle2(self, target_pos_o):
		cur_pos = self.swap_xy(self.bot_position)
		target_pos = self.swap_xy(target_pos_o)
		yaw_pos = (int(cur_pos[0] + 10*math.sin(self.yaw)), int(cur_pos[1] + 10*math.cos(self.yaw)))

		# overlay_img = np.zeros((self.map_height, self.map_width, 3), np.uint8)
		# cv2.circle(overlay_img, cur_pos, 3, (0,0,255), -1)
		# cv2.circle(overlay_img, target_pos, 3, (0,255,0), -1)
		# cv2.circle(overlay_img, yaw_pos, 3, (255,0,0), -1)

		# cv2.line(overlay_img, yaw_pos, target_pos, (128,128,0), thickness=1, lineType=8)
		# cv2.line(overlay_img, cur_pos, target_pos, (128,0,128), thickness=1, lineType=8)
		# cv2.line(overlay_img, cur_pos, yaw_pos, (0,128,128), thickness=1, lineType=8)
		# overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)

		# using cos rule, angle to find is angle
		# A is bots position
		# B is yaw point (arbitrary point straight ahead of the bot)
		# C is unmapped region
		# a is BC
		# b is AC
		# c is AB
		angle = (self.get_dist_sqr(cur_pos, target_pos) + self.get_dist_sqr(cur_pos, yaw_pos) - self.get_dist_sqr(yaw_pos, target_pos))/(2*self.get_dist(cur_pos, target_pos)*self.get_dist(cur_pos, yaw_pos))
		# cv2.imshow('angle img', overlay_img)
		# cv2.waitKey(0)
		rospy.loginfo('[NAV][ANGLE] cos angle: %f', angle)
		return angle


	def target_reached(self, target):
		cur_dist = self.get_dist(self.bot_position, target)
		rospy.loginfo('[NAV][TRGT] Current distance to target: %d', cur_dist)
		if cur_dist > self.dist_to_trgt:
			rospy.loginfo('[NAV][TRGT] Moving the wrong way, recheck direction')
			return True
		elif cur_dist > self.prev_dist:
			rospy.loginfo('Went too far! recheck direction')
			return True
		elif cur_dist < 2:
			return True
		else:
			self.prev_dist = cur_dist
			return False

	def display_angles(self):
		# pass
		unmapped_region = self.get_nearest_unmapped_region()
		overlay_img = np.zeros((self.map_height, self.map_width, 3), np.uint8)

		cv2.line(overlay_img, self.swap_xy(self.bot_position), self.swap_xy(unmapped_region), (0,0,255), thickness=1, lineType=8)

		self.rotate_bot(1)

		length = 8
		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 1: %f', self.yaw)
		req_angle = self.get_angle2(unmapped_region)
		rospy.loginfo('required angle 1: %f', req_angle)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (0,64,0), thickness=1, lineType=8)

		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)

		self.rotate_bot(req_angle)

		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 2: %f', self.yaw)
		req_angle = self.get_angle2(unmapped_region)
		rospy.loginfo('required angle 2: %f', req_angle)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (128,0,0), thickness=1, lineType=8)

		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)

		self.rotate_bot(math.pi/3)

		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 3: %f', self.yaw)
		req_angle = self.get_angle2(unmapped_region)
		rospy.loginfo('required angle 3: %f', req_angle)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (0,192,0), thickness=1, lineType=8)

		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)

		self.rotate_bot(req_angle)

		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 4: %f', self.yaw)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (255,0,0), thickness=1, lineType=8)
		rospy.loginfo('required angle 4: %f', self.get_angle(self.bot_position, unmapped_region))
		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)
	
		cv2.imshow('overlay_img', overlay_img)
		cv2.waitKey(0)


	def test_func2(self, data):
		self.update_map()
		self.display_angles()
		# self.rotate_bot(math.degrees(60))
		# self.display_angles()
		# self.test_func("")


	def test_func(self, data):
		self.update_map()
		unmapped_region = self.get_nearest_unmapped_region()
		target = ()
		# cv2.imshow('map raw', self.occ_map_raw)
		# cv2.waitKey(0)
		if self.path_blocked(self.bot_position, unmapped_region):
			rospy.loginfo('[NAV][TEST] Path blocked')
			target = self.get_closest_edge(unmapped_region)
		else:
			rospy.loginfo('[NAV][TEST] Path free')
			target = unmapped_region
		rospy.loginfo('[NAV][TEST] target position: %s', str(target))
		# self.display_map()
		# angle = self.get_angle(self.swap_xy(self.bot_position),self.swap_xy(target))
		angle = self.get_angle2(target)
		rospy.loginfo('[NAV][TEST] rotation angle: %d', angle)

		self.dist_to_trgt = self.get_dist(self.bot_position, target)

		self.rotate_bot(angle)
		self.move_bot(self.linear_spd, 0.0)


		rate = rospy.Rate(1)
		loop_n = 0
		while True:
			if not self.target_reached(target):
				rospy.loginfo('loop iter: %d', loop_n)
				loop_n += 1
				rate.sleep()
			else:
				self.move_bot(0.0,0.0)
				unmapped_region = self.get_nearest_unmapped_region()
				if self.path_blocked(self.bot_position, unmapped_region):
					rospy.loginfo('[NAV][TEST][LOOP] PATH blocked')
					target = self.get_closest_edge(unmapped_region)
				else:
					rospy.loginfo('[NAV][TEST][LOOP] PATH free')
					target = unmapped_region
				angle = self.get_angle2(target)
				self.dist_to_trgt = self.get_dist(self.bot_position, target)
				rospy.loginfo('[NAV][TEST] rotation angle: %d', angle)
				rospy.loginfo('[NAV][TRGT] distance to target: %d', self.dist_to_trgt)

				self.rotate_bot(angle)
				self.move_bot(self.linear_spd, 0.0)

		self.move_bot(0.0,0.0)

	def move_bot(self, linear_spd, angular_spd):
		pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		twist = Twist()
		twist.linear.x = linear_spd
		twist.angular.z = angular_spd
		time.sleep(1)
		pub.publish(twist)
	
	def rotate_bot(self, rot_angle):
		# create Twist object
		twist = Twist()
		# set the update rate to 1 Hz
		rate = rospy.Rate(1)
	
		# get current yaw angle
		current_yaw = np.copy(self.yaw)
		# log the info
		rospy.loginfo(['Current: ' + str(math.degrees(current_yaw))])
		# we are going to use complex numbers to avoid problems when the angles go from
		# 360 to 0, or from -180 to 180
		c_yaw = complex(math.cos(current_yaw),math.sin(current_yaw))
		# calculate desired yaw
		# target_yaw = math.radians(rot_angle)
		target_yaw = rot_angle
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
		twist.angular.z = c_change_dir * self.angular_spd
		rospy.loginfo(twist.angular.z)
		# start rotation
		# pub.publish(twist)
		self.move_bot(0.0, c_change_dir *self.angular_spd)
	
		# we will use the c_dir_diff variable to see if we can stop rotating
		c_dir_diff = c_change_dir
		# rospy.loginfo(['c_change_dir: ' + str(c_change_dir) + ' c_dir_diff: ' + str(c_dir_diff)])
		# if the rotation direction was 1.0, then we will want to stop when the c_dir_diff
		# becomes -1.0, and vice versa
		while(c_change_dir * c_dir_diff > 0):
			# get current yaw angle
			current_yaw = np.copy(self.yaw)
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
		# pub.publish(twist)
		self.move_bot(0.0,0.0)