#!/usr/bin/env python

import cv2
import numpy as np
import math
import cmath
import time

import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

angular_tolerance = 0.05

class Navigation():
	def __init__(self, linear_spd, angular_spd):
		self.linear_spd = linear_spd
		self.angular_spd = angular_spd

		self.yaw = 0.0
		self.lidar_data = np.array([])
		self.bot_position = ()
		self.occ_grid = np.array([])

		self.bot_angular_range = range(-15,15)
		self.stop_dist = 0.15

		self.map_origin = ()
		self.map_height = 0
		self.map_width = 0
		self.map_res = 0

		self.mapping_complete = False

		self.occ_map_raw = np.array([])
		self.occ_map = np.array([])
		self.checked_positions = []
		self.edges = []

		self.pseudo_route = []	# list of edges/unmapped regions followed
		self.main_route = []	# list of actual positions the bot will follow

		self.unmapped_region = ()
		self.prev_target = ()
		self.cur_target = ()
		self.target_changed = False
		self.angle_to_target = 0.0
		self.inf_visible = False
		self.in_motion = False
		self.obstacle_detected = False

		self.dist_to_trgt = 0
		self.prev_dist = 99999999

	# methods to update sensor data
	def update_yaw(self, data):
		self.yaw = data
	
	def update_laserscan_data(self, data):
		self.lidar_data = data
		# self.inf_visible = (np.inf in data)\
		self.obstacle_check()

	def update_occ_grid(self, grid, origin, resolution):
		self.occ_grid = grid
		self.map_origin = origin
		self.map_res = resolution
		self.map_height = len(grid)
		self.map_width = len(grid[0])
		# if self.inf_visible:
		# 	rospy.loginfo(self.inf_visible)
		# 	inf_loc = np.where(self.lidar_data == np.inf)
		# 	rospy.loginfo(inf_loc)
		# 	for i in inf_loc:
		# 		rospy.loginfo(self.lidar_data[0][i])
		# 	self.pick_direction()

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
			i,j = cur_pos
			for next_pos in [(i-1,j), (i,j+1), (i+1,j), (i,j-1)]:
				if next_pos not in self.checked_positions:
					if self.occ_grid[next_pos] == 0:
						rospy.loginfo('[NAV][OCC] Found nearest unmapped region at %s', str(next_pos))
						self.pseudo_route.insert(0, next_pos)
						self.rviz_marker(next_pos, 0)
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
		if len(distances) == 0:
			rospy.loginfo('error: No edges detected?')
			return False
		else:
			closest_edge = self.swap_xy(self.edges[distances.index(min(distances))])
			self.rviz_marker(closest_edge, 2)
			return closest_edge
	
	# check if there are any obstacles blocking the straight line path between two points
	def path_blocked(self, cur_pos, next_pos):
		self.update_map()
		path_img = np.zeros((self.map_height, self.map_width, 1), np.uint8)
		overlay_img = np.zeros((self.map_height, self.map_width, 1), np.uint8)
		cv2.line(path_img, self.swap_xy(cur_pos), self.swap_xy(next_pos), 255, thickness=1, lineType=8)
		cv2.circle(path_img, self.swap_xy(cur_pos), 3, 255, -1)

		cv2.circle(path_img, (next_pos[1], next_pos[0]), 3, 128, -1)

		ret, path_img_bit = cv2.threshold(path_img, 2, 255, 0)

		overlay_map = cv2.bitwise_and(path_img_bit, self.occ_map_raw)

		overlay_img = cv2.bitwise_or(self.occ_map_raw, path_img)

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

	# get angle need to turn, accounting for yaw
	def get_angle(self, target_pos_o):
		cur_pos = self.swap_xy(self.bot_position)
		target_pos = self.swap_xy(target_pos_o)
		yaw_pos = (int(cur_pos[0] + 10*math.sin(self.yaw)), int(cur_pos[1] + 10*math.cos(self.yaw)))

		# using cos rule, angle to find is angle
		# A is bots position
		# B is yaw point (arbitrary point straight ahead of the bot)
		# C is unmapped region
		# a is BC
		# b is AC
		# c is AB
		cos_rule_rhs = (self.get_dist_sqr(cur_pos, target_pos) + self.get_dist_sqr(cur_pos, yaw_pos) - self.get_dist_sqr(yaw_pos, target_pos))/(2*self.get_dist(cur_pos, target_pos)*self.get_dist(cur_pos, yaw_pos))
		if cos_rule_rhs > 1:
			cos_rule_rhs = 1
		elif cos_rule_rhs < -1:
			cos_rule_rhs = -1	

		angle = math.acos(cos_rule_rhs)
		rospy.loginfo('[NAV][ANGLE] cos angle: %f', angle)
		return angle

	def target_reached(self, target):
		cur_dist = self.get_dist(self.bot_position, self.cur_target)
		rospy.loginfo('[NAV][TRGT] Current distance to target: %d', cur_dist)
		if cur_dist > self.dist_to_trgt:
			rospy.loginfo('[NAV][TRGT] Moving the wrong way, recheck direction')
			self.prev_dist = 999999999
			return True
		elif cur_dist > self.prev_dist:
			rospy.loginfo('[NAV][TRGT] Went too far! recheck direction')
			self.prev_dist = 999999999
			return True
		elif cur_dist < 2:
			self.prev_dist = 999999999
			return True
		else:
			self.prev_dist = cur_dist
			if cur_dist > 4:
				self.move_bot(2*self.linear_spd, 0.0)
			elif cur_dist > 6:
				self.move_bot(3*self.linear_spd, 0.0)
			return False

	def obstacle_check(self):
		lr2 = self.lidar_data[0,self.bot_angular_range]

		list_to_check = [x < self.stop_dist and not (x == 0 or x == np.inf) for x in lr2]
		if np.any(list_to_check):
			# rospy.loginfo(lr2[0])
			rospy.logwarn('[NAV][TRGT] Obstacle in front!, recheck direction')
			self.obstacle_detected = True
		else:
			self.obstacle_detected = False

	def display_angles(self):
		unmapped_region = self.get_nearest_unmapped_region()
		overlay_img = np.zeros((self.map_height, self.map_width, 3), np.uint8)

		cv2.line(overlay_img, self.swap_xy(self.bot_position), self.swap_xy(unmapped_region), (0,0,255), thickness=1, lineType=8)

		self.rotate_bot(1)

		length = 8
		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 1: %f', self.yaw)
		req_angle = self.get_angle(unmapped_region)
		rospy.loginfo('required angle 1: %f', req_angle)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (0,64,0), thickness=1, lineType=8)

		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)

		while req_angle > 0.08:
			rospy.loginfo('[NAV][ANGLE] req_angle: %f', req_angle)
			self.rotate_bot(req_angle)
			req_angle = self.get_angle(unmapped_region)

		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 2: %f', self.yaw)
		req_angle = self.get_angle(unmapped_region)
		rospy.loginfo('required angle 2: %f', req_angle)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (128,0,0), thickness=1, lineType=8)

		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)

		self.rotate_bot(math.pi/3)

		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 3: %f', self.yaw)
		req_angle = self.get_angle(unmapped_region)
		rospy.loginfo('required angle 3: %f', req_angle)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (0,192,0), thickness=1, lineType=8)

		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)

		while req_angle > 0.1:
			rospy.loginfo('[NAV][ANGLE] req_angle: %f', req_angle)
			self.rotate_bot(req_angle)
			req_angle = self.get_angle(unmapped_region)

		yaw_pi = int(self.bot_position[1] + length * math.sin(self.yaw))
		yaw_pj = int(self.bot_position[0] + length * math.cos(self.yaw))
		yaw_point = (yaw_pi, yaw_pj)
		rospy.loginfo('[NAV][ANGLE] current_yaw 4: %f', self.yaw)
		cv2.line(overlay_img, self.swap_xy(self.bot_position), yaw_point, (255,0,0), thickness=1, lineType=8)
		rospy.loginfo('required angle 4: %f', self.get_angle(self.bot_position, unmapped_region))
		overlay_img = cv2.bitwise_or(self.occ_map, overlay_img)
	
		cv2.imshow('overlay_img', overlay_img)
		cv2.waitKey(0)

	def rotate_to_point(self, pos):
		rot_angle = self.get_angle(pos)

		while rot_angle > angular_tolerance:
			self.rotate_bot(rot_angle)
			rot_angle = self.get_angle(pos)

	def get_furthest_visible(self, pos):
		closest_edge = self.get_closest_edge(self.swap_xy(pos))
		i,j = closest_edge
		length = 3
		corners = [(i-length,j-length),(i-length,j+length),(i+length,j+length),(i+length,j-length)]
		distances = [0,0,0,0]
		dist_i = 0

		for corner in corners:
			if not self.path_blocked(self.bot_position, corner):
				distances[dist_i] = self.get_dist_sqr(self.bot_position, corner)
			dist_i += 1

			furthest_visible = corners[distances.index(max(distances))]
			self.rviz_marker(furthest_visible, 1)
		return furthest_visible

	def pick_direction(self):
		unmapped_region = self.get_nearest_unmapped_region()
		self.rviz_marker(unmapped_region, 0)
		self.unmapped_region = unmapped_region

		# if self.inf_visible:
		# 	new_angle = self.get_angle(unmapped_region)
		# 	if abs(new_angle - self.angle_to_target) > angular_tolerance:
		# 		self.target_changed = True
		# 		self.cur_target = unmapped_region
		# 	else:
		# 		self.cur_target = unmapped_region
		# 		self.target_changed = False
		# 	return

		target = ()
		if self.path_blocked(self.bot_position, unmapped_region):
			target = self.get_furthest_visible(unmapped_region)
		else:
			target = unmapped_region

		self.cur_target = target
		if self.get_dist(self.cur_target, target) < 2:
			self.target_changed = False
		else:
			self.target_changed = True

		self.angle_to_target = self.get_angle(target)

		# if self.prev_target == ():
		# 	self.prev_target = target
		# 	return True, target, self.get_angle(target)

		# if self.get_dist(target, self.prev_target) > 2:
		# 	self.prev_target = target
		# 	return True, target, self.get_angle(target)
		# else:
		# 	return False, target, self.get_angle(target)

	def map_region(self):
		rate = rospy.Rate(5)
		while True:
			if self.cur_target == ():
				rospy.loginfo('no target yet')
				self.pick_direction()

			rospy.loginfo('[NAV][CHCK] obstacle presence: %s', self.obstacle_detected)
			if self.obstacle_detected:
				rospy.loginfo('obstacle detected')
				self.move_bot(-self.linear_spd, 0.0)
				time.sleep(1)
				self.move_bot(0.0,0.0)
				self.pick_direction()
				# self.rotate_bot(self.angle_to_target)
				self.rotate_to_point(self.cur_target)
				self.move_bot(self.linear_spd,0.0)
				# time.sleep(1)
				continue

			# if self.inf_visible and self.in_motion:
			# 	self.move_bot(3 * self.linear_spd, 0.0)
			# 	rate.sleep()
			# change, target, angle = self.pick_direction()
			if self.target_changed:
				rospy.loginfo('stoppping because of target change')
				self.target_changed = False
				self.move_bot(0.0,0.0)
				self.rotate_to_point(self.cur_target)
				# self.rotate_bot(self.angle_to_target)

			self.move_bot(self.linear_spd,0.0)

			if self.target_reached(self.cur_target):
				rospy.loginfo('Target reached, choosing new direction')
				self.target_changed = True
				self.pick_direction()
			else:
				rate.sleep()

	def test_func(self, data):
		self.pick_direction()
		self.rotate_to_point(self.cur_target)

	def move_bot(self, linear_spd, angular_spd):
		pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		if linear_spd != 0.0 or angular_spd != 0.0:
			self.in_motion = True
		else:
			self.in_motion = False
		twist = Twist()
		twist.linear.x = linear_spd
		twist.angular.z = angular_spd
		time.sleep(1)
		pub.publish(twist)
	
	def rotate_bot(self, rot_angle):
		# create Twist object
		# twist = Twist()
		# set the update rate to 1 Hz
		rate = rospy.Rate(3)
	
		# get current yaw angle
		current_yaw = np.copy(self.yaw)
		# log the info
		rospy.loginfo(['Current: ' + str(math.degrees(current_yaw))])
		# we are going to use complex numbers to avoid problems when the angles go from
		# 360 to 0, or from -180 to 180
		c_yaw = complex(math.cos(current_yaw),math.sin(current_yaw))
		# calculate desired yaw
		# target_yaw = math.radians(rot_angle)
		target_yaw = current_yaw + rot_angle
		self.rviz_arrow(self.bot_position, target_yaw, 0)
		# target_yaw = rot_angle
		# convert to complex notation
		c_target_yaw = complex(math.cos(target_yaw),math.sin(target_yaw))
		rospy.loginfo(['Desired: ' + str(math.degrees(cmath.phase(c_target_yaw)))])
		# divide the two complex numbers to get the change in direction
		c_change = c_target_yaw / c_yaw
		# get the sign of the imaginary component to figure out which way we have to turn
		# c_change_dir = np.sign(c_change.imag)
		c_change_dir = 1.0 if rot_angle > math.pi else -1.0
		# set linear speed to zero so the TurtleBot rotates on the spot
		# twist.linear.x = 0.0
		# set the direction to rotate
		# twist.angular.z = c_change_dir * self.angular_spd
		# rospy.loginfo(twist.angular.z)
		# start rotation
		# pub.publish(twist)
		self.move_bot(0.0, c_change_dir *self.angular_spd)
	
		# we will use the c_dir_diff variable to see if we can stop rotating
		c_dir_diff = c_change_dir
		# rospy.loginfo(['c_change_dir: ' + str(c_change_dir) + ' c_dir_diff: ' + str(c_dir_diff)])
		# if the rotation direction was 1.0, then we will want to stop when the c_dir_diff
		# becomes -1.0, and vice versa

		prev_rem_ang_dist = target_yaw - current_yaw


		rospy.loginfo('[NAV][ROT] Turning to face %f', target_yaw)
		while(c_change_dir * c_dir_diff > 0):
			# get current yaw angle
			current_yaw = np.copy(self.yaw)
			# get the current yaw in complex form
			c_yaw = complex(math.cos(current_yaw),math.sin(current_yaw))
			# rospy.loginfo('While Yaw: %f Target Yaw: %f', math.degrees(current_yaw), math.degrees(target_yaw))
			# get difference in angle between current and target
			c_change = c_target_yaw / c_yaw
			# get the sign to see if we can stop
			c_dir_diff = np.sign(c_change.imag)
			# rospy.loginfo(['c_change_dir: ' + str(c_change_dir) + ' c_dir_diff: ' + str(c_dir_diff)])
			rem_ang_dist = target_yaw - current_yaw

			# if rem_ang_dist > prev_rem_ang_dist:
				# self.move_bot(0.0, c_change_dir * self.angular_spd * (-1))
			# else:
				# prev_rem_ang_dist = rem_ang_dist


			if rem_ang_dist > 3:
				self.move_bot(0.0,3 * self.angular_spd * c_change_dir)
			elif rem_ang_dist > 2:
				self.move_bot(0.0,2.5 * self.angular_spd* c_change_dir)
			elif rem_ang_dist > 1:
				self.move_bot(0.0,1.5 * self.angular_spd* c_change_dir)
			else:
				self.move_bot(0.0,self.angular_spd* c_change_dir)

			rate.sleep()

		# while True:
		# 	rem_ang_dist = abs(self.yaw - self.angle_to_target)
		# 	rospy.loginfo(rem_ang_dist)
		# 	if rem_ang_dist > angular_tolerance:

		# 		if rem_ang_dist > 2:
		# 			self.move_bot(0.0,c_change_dir * self.angular_spd * 2)
		# 		elif rem_ang_dist < 0.6:
		# 			self.move_bot(0.0,c_change_dir * self.angular_spd * 0.25)
		# 		else:
		# 			self.move_bot(0.0,c_change_dir * self.angular_spd)

		# 		rate.sleep()
		# 	else:
		# 		break
	
		rospy.loginfo(['End Yaw: ' + str(math.degrees(current_yaw))])
		# set the rotation speed to 0
		# twist.angular.z = 0.0
		# stop the rotation
		# time.sleep(1)
		# pub.publish(twist)
		self.move_bot(0.0,0.0)

	def rviz_marker(self, pos, marker_type):
		pos_x = pos[0] * self.map_res + self.map_origin.x
		pos_y = pos[1] * self.map_res + self.map_origin.y

		map_pub = rospy.Publisher('nav_markers', Marker, queue_size=10)
		rospy.sleep(1)
		nav_marker = Marker()
		nav_marker.header.frame_id = "map"
		nav_marker.ns = "point_marker"
		nav_marker.id = marker_type
		nav_marker.pose.position.x = pos_x
		nav_marker.pose.position.y = pos_y
		nav_marker.type = nav_marker.CUBE
		nav_marker.action = nav_marker.MODIFY
		nav_marker.scale.x = 0.1
		nav_marker.scale.y = 0.1
		nav_marker.scale.z = 0.1
		nav_marker.pose.orientation.w = 1.0
		if marker_type == 0:
			nav_marker.color.r = 1.0
			nav_marker.color.g = 0.0
			nav_marker.color.b = 0.0
		elif marker_type == 1:
			nav_marker.color.r = 0.0
			nav_marker.color.b = 0.0
			nav_marker.color.g = 1.0
		elif marker_type == 2:
			nav_marker.color.r = 0.0
			nav_marker.color.b = 1.0
			nav_marker.color.g = 0.0

		nav_marker.color.a = 1.0

		map_pub.publish(nav_marker)
		rospy.loginfo('[NAV][RVIZ] Published marker at %s', str(pos))

	def rviz_arrow(self, pos, angle, marker_type):
		length = 1
		pos2_x = int(pos[1] + length * math.sin(angle)) * self.map_res + self.map_origin.x
		pos2_y = int(pos[0] + length * math.cos(angle)) * self.map_res + self.map_origin.y

		pos1_x = pos[0] * self.map_res + self.map_origin.x
		pos1_y = pos[1] * self.map_res + self.map_origin.y

		map_pub = rospy.Publisher('nav_markers', Marker, queue_size=10)
		rospy.sleep(1)
		nav_marker = Marker()
		nav_marker.header.frame_id = "map"
		nav_marker.ns = "angle_marker"
		nav_marker.id = marker_type
		nav_marker.type = nav_marker.ARROW
		nav_marker.points.append(Point(pos1_x, pos1_y, 0))
		nav_marker.points.append(Point(pos1_y, pos2_y, 0))
		nav_marker.action = nav_marker.MODIFY
		nav_marker.scale.x = 0.01
		nav_marker.scale.y = 0.015
 		nav_marker.color.r = 0.0
		nav_marker.color.g = 1.0
		nav_marker.color.b = 1.0
		nav_marker.color.a = 1.0

		map_pub.publish(nav_marker)
		rospy.loginfo('[NAV][RVIZ] Published angle indicator to %f', angle)