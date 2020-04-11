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

angular_tolerance = 0.3

class Navigation():
	def __init__(self, linear_spd, angular_spd):
		self.linear_spd = linear_spd
		self.angular_spd = angular_spd

		self.yaw = 0.0
		self.lidar_data = np.array([])
		self.lidar_data_front = np.array([])
		self.bot_position = ()
		self.occ_grid = np.array([])
		self.prev_occ_recv = 0.0

		self.bot_angular_range = range(-20,20)
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
		self.picking_direction = False

		self.dist_to_trgt = 0
		self.prev_dist = 99999999

	# methods to update sensor data
	def update_yaw(self, data):
		self.yaw = data
	
	def update_laserscan_data(self, data):
		self.lidar_data = data
		self.lidar_data_front = data[0,self.bot_angular_range]
		self.inf_visible = (np.inf in self.lidar_data_front)
		self.obstacle_check()

	def update_occ_grid(self, grid, origin, resolution):
		self.occ_grid = grid
		self.map_origin = origin
		self.map_res = resolution
		self.map_height = len(grid)
		self.map_width = len(grid[0])

		# cur_time = rospy.get_rostime().secs
		# if cur_time - self.prev_occ_recv > 5 and not self.picking_direction:
		# 	rospy.loginfo('[NAV][OCC] Checking target location')
		# 	self.update_map()
		# 	self.pick_direction()
		# 	self.prev_occ_recv = cur_time

		# if self.inf_visible:
		# 	rospy.loginfo(self.inf_visible)
		# 	inf_loc = np.where(self.lidar_data == np.inf)
		# 	rospy.loginfo(inf_loc)
		# 	for i in inf_loc:
		# 		rospy.loginfo(self.lidar_data[0][i])
		# 	self.pick_direction()

	def update_bot_pos(self, data):
		self.bot_position = data

	def obstacle_check(self):
		list_to_check = [x < self.stop_dist and not (x == np.inf) for x in self.lidar_data_front]
		if np.any(list_to_check):
			rospy.logwarn('[NAV][TRGT] Obstacle in front!, recheck direction')
			self.obstacle_detected = True
		else:
			self.obstacle_detected = False

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
		rospy.logdebug('[NAV][OCC] Finding nearest unmapped region')
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
		# return False
		exit()
	
	# update occ_map and occ_map_raw
	def update_map(self):
		rospy.loginfo('[NAV][MAP] Updating map')
		ret, occ_map_raw = cv2.threshold(self.occ_grid, 2, 255, 0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
		self.occ_map_raw = cv2.dilate(occ_map_raw, element)
		# self.occ_map_raw = cv2.morphologyEx(occ_map_raw, cv2.MORPH_CLOSE, element)
		self.occ_map = cv2.cvtColor(self.occ_map_raw, cv2.COLOR_GRAY2RGB)
	
	# find the edges of the visible walls
	def update_edges(self):
		# self.update_map()
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
			rospy.logwarn('[NAV][EDGE] No edges detected, there is likely an issue')
			self.display_edges()
	
	# find the nearest edge to a position
	def get_closest_edge(self, pos):
		self.update_edges()
		rospy.loginfo('[NAV][EDGE] Locating the nearest edge to %s', str(pos))
		distances = [self.get_dist_sqr(edge_pos, pos) for edge_pos in self.edges]
		if len(distances) == 0:
			rospy.logwarn('error: No edges detected?')
			return False
		else:
			closest_edge = self.swap_xy(self.edges[distances.index(min(distances))])
			rospy.loginfo('[NAV][TRGT] Closest edge located at %s', str(closest_edge))
			self.rviz_marker(closest_edge, 2)
			return closest_edge

	def get_furthest_visible(self, pos):
		closest_edge = self.get_closest_edge(self.swap_xy(pos))
		i,j = closest_edge
		length = 4
		corners = [(i-length,j-length),(i-length,j+length),(i+length,j+length),(i+length,j-length)]
		distances = [0,0,0,0]
		dist_i = 0

		for corner in corners:
			if not self.path_blocked(self.bot_position, corner):
				distances[dist_i] = self.get_dist_sqr(self.bot_position, corner)
				# rospy.logwarn(dist_i)
			dist_i += 1

		furthest_visible = corners[distances.index(max(distances))]
		# rospy.loginfo(distances)
		rospy.loginfo('[NAV][TRGT] Furthest visible located at %s', str(furthest_visible))
		self.rviz_marker(furthest_visible, 1)
		return furthest_visible

	def pick_direction(self):
		self.picking_direction = True
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

		self.update_map()
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

		self.picking_direction = False
		# if self.prev_target == ():
		# 	self.prev_target = target
		# 	return True, target, self.get_angle(target)

		# if self.get_dist(target, self.prev_target) > 2:
		# 	self.prev_target = target
		# 	return True, target, self.get_angle(target)
		# else:
		# 	return False, target, self.get_angle(target)
	
	# check if there are any obstacles blocking the straight line path between two points
	def path_blocked(self, cur_pos, next_pos):
		# self.update_map()
		path_img = np.zeros((self.map_height, self.map_width, 1), np.uint8)
		cv2.line(path_img, self.swap_xy(cur_pos), self.swap_xy(next_pos), 255, thickness=1, lineType=8)

		ret, path_img_bit = cv2.threshold(path_img, 2, 255, 0)

		overlay_map = cv2.bitwise_and(path_img_bit, self.occ_map_raw)
	
		# overlay_img = np.zeros((self.map_height, self.map_width, 1), np.uint8)
		# overlay_img = cv2.bitwise_or(self.occ_map_raw, path_img)

		# cv2.imshow('overlay_img', overlay_img)
		# cv2.imshow('overlay_map', overlay_map)
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

	def get_angle(self, pos):
		cur_pos = self.swap_xy(self.bot_position)
		target_pos = self.swap_xy(pos)

		i_dist = target_pos[1] - cur_pos[1]
		j_dist = target_pos[0] - cur_pos[0]

		angle = math.atan2(j_dist, i_dist)
		self.angle_to_target = angle
		return angle

	def target_reached(self, target):
		if self.inf_visible:
			self.move_bot(3*self.linear_spd, 0.0)
			return False
		cur_dist = self.get_dist(self.bot_position, self.cur_target)
		rospy.loginfo('[NAV][TRGT] Current distance to target: %d', cur_dist)
		# if cur_dist > self.dist_to_trgt:
			# rospy.loginfo('[NAV][TRGT] Moving the wrong way, recheck direction')
			# self.prev_dist = 999999999
			# return True
		if cur_dist > self.prev_dist:
			rospy.loginfo('[NAV][TRGT] Went too far! recheck direction')
			self.prev_dist = 999999999
			return True
		elif cur_dist < 2:
			self.prev_dist = 999999999
			return True
		else:
			self.prev_dist = cur_dist
			if cur_dist > 4:
				self.move_bot(1.5*self.linear_spd, 0.0)
			elif cur_dist > 6:
				self.move_bot(2.5*self.linear_spd, 0.0)
			return False

	def map_region(self, data):
		self.pick_direction()
		self.rotate_bot(self.angle_to_target)
		self.move_bot(self.linear_spd, 0.0)

		rate = rospy.Rate(1)
		while True:
			if self.obstacle_detected:
				rospy.logwarn('[NAV][OBS] Obstacle detected!')
				self.move_bot(-3*self.linear_spd, 0.0)
				time.sleep(1)
				self.move_bot(0.0,0.0)
				time.sleep(0.1)
				self.pick_direction()
				# self.rotate_bot(self.get_angle(self.cur_target))
				self.rotate_bot(self.angle_to_target)
				self.move_bot(self.linear_spd, 0.0)

			elif self.target_reached(self.cur_target):
				rospy.loginfo('[NAV][TRGT] Reached target! changing direction')
				self.move_bot(0.0,0.0)
				time.sleep(2)
				self.pick_direction()
				# self.rotate_bot(self.get_angle(self.cur_target))
				self.rotate_bot(self.angle_to_target)
				self.move_bot(self.linear_spd, 0.0)

			elif self.angle_to_target > angular_tolerance:
				rospy.logwarn('[NAV][TRGT] angular tolerance exceeeded, adjusting yaw')
				# self.rotate_to_target()
				self.pick_direction()
				self.rotate_bot(self.get_angle(self.cur_target))
				self.move_bot(self.linear_spd, 0.0)
			rate.sleep()

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
	
	def rotate_bot(self, angle):
		current_yaw = np.copy(self.yaw)
		target_yaw = angle

		c_current_yaw = complex(math.cos(current_yaw), math.sin(current_yaw))
		c_target_yaw = complex(math.cos(target_yaw), math.sin(target_yaw))

		rospy.loginfo('[NAV][ROT] current yaw: %f | target yaw: %f', current_yaw, target_yaw)
		c_change_dir = np.sign((c_target_yaw/c_current_yaw).imag)
		rospy.loginfo('[NAV][ROT] Starting rotation')

		c_dir_diff = c_change_dir
		rate = rospy.Rate(5)
		while c_change_dir * c_dir_diff > 0: 
			current_yaw = np.copy(self.yaw)
			c_current_yaw = complex(math.cos(current_yaw), math.sin(current_yaw))
			c_dir_diff = np.sign((c_target_yaw/c_current_yaw).imag)
			rem_ang_dist = abs(current_yaw - target_yaw)
			if rem_ang_dist > math.pi:
				rem_ang_dist = (2*math.pi) - rem_ang_dist

			if rem_ang_dist > 3:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 3)
			elif rem_ang_dist > 2:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 2.5)
			elif rem_ang_dist > 1:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 1.5)
			elif rem_ang_dist > 0.5:
				self.move_bot(0.0, self.angular_spd * c_change_dir)
			else:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 0.5)

			rate.sleep()

		if abs(self.yaw - self.angle_to_target) > 0.2:
			rospy.logwarn('[NAV][ROTN] Facing the wrong way, target possibly changed')
			self.rotate_bot(self.angle_to_target)

		rospy.loginfo('[NAV][ROT] Finished rotating, now facing %f', self.yaw)
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
		rospy.logdebug('[NAV][RVIZ] Published marker at %s, type: %d', str(pos), marker_type)

	def rviz_arrow(self, pos, angle, marker_type):
		length = 1
		pos2_x = int(pos[0] + length * math.sin(angle)) * self.map_res + self.map_origin.x
		pos2_y = int(pos[1] + length * math.cos(angle)) * self.map_res + self.map_origin.y

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