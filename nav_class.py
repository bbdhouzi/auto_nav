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

angular_tolerance = 0.5

class Navigation():
	def __init__(self, linear_spd, angular_spd):
		self.linear_spd = linear_spd
		self.angular_spd = angular_spd

		self.yaw = 0.0
		self.lidar_data = np.array([])
		self.lidar_data_front_wide = np.array([])
		self.lidar_data_front_narrow = np.array([])
		self.bot_position = ()
		self.occ_grid = np.array([])
		self.prev_occ_recv = 0.0
		self.home_loc = (0,0)

		self.bot_angular_range_obs = range(-30,30)
		self.bot_angular_range_inf = range(-8,8)
		self.stop_dist = 0.2

		self.map_origin = ()
		self.map_height = 0
		self.map_width = 0
		self.map_res = 0

		self.mapping_complete = False

		self.occ_map_raw = np.array([])
		self.occ_map = np.array([])
		self.checked_positions = []
		self.edges = []
		self.edges_ordered = []

		self.pseudo_route = []	# list of edges/unmapped regions followed
		self.main_route = []	# list of actual positions the bot will follow

		self.unmapped_region = ()
		self.prev_target = ()
		self.cur_target = ()
		self.target_changed = False
		self.angle_to_target = 0.0
		self.inf_visible = False
		self.is_moving = False
		self.is_rotating = False
		self.obstacle_detected = False
		self.picking_direction = False

		self.dist_to_trgt = 0
		self.prev_dist = 99999999

	# methods to update sensor data
	def update_yaw(self, data):
		self.yaw = data
	
	def update_laserscan_data(self, data):
		self.lidar_data = data
		self.lidar_data_front_wide = data[0,self.bot_angular_range_obs]
		self.lidar_data_front_narrow = data[0, self.bot_angular_range_inf]
		self.inf_visible = (np.inf in self.lidar_data_front_narrow)
		self.obstacle_check()

	def update_occ_grid(self, grid, origin, resolution):
		self.occ_grid = grid
		self.map_origin = origin
		self.map_res = resolution
		self.map_height = len(grid)
		self.map_width = len(grid[0])

	def update_bot_pos(self, data):
		self.bot_position = data

	def obstacle_check(self):
		list_to_check = [x < self.stop_dist and not (x == np.inf) for x in self.lidar_data_front_wide]
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

		time.sleep(0.5)
		if True:
		# for x in range(2):
			# rospy.logwarn(x)
			# pos_to_check = [self.bot_position]
			# self.checked_positions = []
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
		return (0,0)
		# exit()
	
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

			edge_copy = self.edges[:]
			closest_edge_list = []
			for i in range(len(distances)):
				min_index = distances.index(min(distances))
				distances.pop(min_index)
				closest_edge_list.append(edge_copy[min_index])
				edge_copy.pop(min_index)

			self.edges_ordered = closest_edge_list

			return closest_edge

	def get_furthest_visible(self, pos):
		def internal_func():
			i,j = closest_edge
			length = 5
			corners = [(i-length,j-length), (i-length,j), (i-length,j+length), (i,j+length), (i+length,j+length), (i+length,j), (i+length,j-length), (i,j-length)]
			dist_i = 0

			for corner in corners:
				if not self.path_blocked(self.bot_position, corner) and abs(self.get_angle(closest_edge) - self.get_angle(corner)) <= (2*math.pi/3):
					distances[dist_i] = self.get_dist_sqr(self.bot_position, corner)
					# rospy.logwarn(dist_i)
				dist_i += 1
			return corners

		closest_edge = self.get_closest_edge(self.swap_xy(pos))
		distances = [0,0,0,0,0,0,0,0]

		if self.get_dist(self.bot_position, closest_edge) < 3:
			self.edges_ordered.pop(0)
			closest_edge = self.edges_ordered[0]

		corners = internal_func()

		while not any(distances):
			if len(distances) == 0:
				logwarn('[NAV][TRGT] Critical failure, No edges with visible corners')
				return
			rospy.loginfo('[NAV][TRGT] No visible corners found!')
			self.edges_ordered.pop(0)
			closest_edge = self.swap_xy(self.edges_ordered[0])
			rospy.loginfo('[NAV][TRGT] Checking with next closest edge at %s', str(closest_edge))
			self.rviz_marker(closest_edge, 3)
			distances = [0,0,0,0,0,0,0,0]
			corners = internal_func()
			# self.cur_target = corners[0]
			# self.display_map()
			# return False

		furthest_visible = corners[distances.index(max(distances))]
		# rospy.loginfo(distances)
		rospy.loginfo('[NAV][TRGT] Furthest visible located at %s', str(furthest_visible))
		self.rviz_marker(furthest_visible, 1)
		return furthest_visible

	def pick_direction(self, final_target=None):
		def internal_func():
			self.picking_direction = True
			if not self.mapping_complete:
				unmapped_region = self.get_nearest_unmapped_region()
				self.rviz_marker(unmapped_region, 0)
				self.unmapped_region = unmapped_region

			if final_target is not None:
				unmapped_region = final_target

			self.update_map()
			target = ()
			if self.path_blocked(self.bot_position, unmapped_region):
				target = self.get_furthest_visible(unmapped_region)
				if not target:
					time.sleep(3)
					target = self.get_furthest_visible(unmapped_region)
			else:
				target = unmapped_region

			self.cur_target = target
			self.angle_to_target = self.get_angle(target)

			self.prev_dist = 999999999
			self.picking_direction = False

		internal_func()
		# if self.angle_to_target > math.pi/2 or self.angle_to_target < (-1 * math.pi/2):\
		if abs(self.yaw - self.angle_to_target) > math.pi/2:
				rospy.logwarn("[NAV][ANGLE] Situation calls for turning backwards, rechecking direction")
				time.sleep(2)
				internal_func()
	
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
		cv2.circle(map_overlay, self.swap_xy(self.unmapped_region), 3, (128,128,255), -1)

		if refresh:
			closest_edge = self.get_closest_edge(self.bot_position)
			cv2.circle(map_overlay, self.swap_xy(closest_edge), 3, (0,255,0), -1)

		map_overlay = cv2.bitwise_or(map_overlay, self.edge_map)
		occ_map_disp = cv2.bitwise_or(self.occ_map, map_overlay)

		# cv2.imshow('occ_map', self.occ_map)
		# cv2.imshow('edges', self.edge_map)
		# cv2.imshow('map_overlay', map_overlay)
		# cv2.imshow('final', occ_map_disp)
		# cv2.waitKey(0)

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

		if cur_dist > self.prev_dist:
			rospy.loginfo('[NAV][TRGT] Went too far! recheck direction')
			self.prev_dist = 999999999
			return True
		elif cur_dist <= 4:
			self.prev_dist = 999999999
			rospy.loginfo('[NAV][TRGT] Target Reached!')
			return True
		else:
			if cur_dist == self.prev_dist and cur_dist < 6:
				self.move_bot(0.5 * self.linear_spd, 0.0)
				# return False
			self.prev_dist = cur_dist
			if cur_dist > 14:
				self.move_bot(2 * self.linear_spd, 0.0)
			elif cur_dist > 10:
				self.move_bot(1.5 * self.linear_spd, 0.0)
			elif cur_dist > 6:
				self.move_bot(self.linear_spd, 0.0)
			else:
				self.move_bot(0.5 * self.linear_spd, 0.0)

			return False

	# Checks current status of target and bots position
	# Return codes:
	#	0: Target reached
	#	1: Overshot target, recheck direction
	#	2: Target not reached, infinite distance ahead
	#	3: Target not reached, > 14 units ahead
	#	4: Target not reached, > 10 units ahead
	#	5: Target not reached, > 6 units ahead
	#	6: Target not reached, < 6 units ahead
	def target_condition(self):
		if self.inf_visible:
			self.move_bot(3*self.linear_spd, 0.0)
			return 2

		cur_dist = self.get_dist(self.bot_position, self.cur_target)
		rospy.loginfo('[NAV][TRGT] Current distance to target: %f', cur_dist)

		if cur_dist > self.prev_dist:
			rospy.loginfo('[NAV][TRGT] Went too far!, recheck direction')
			self.prev_dist = 999999999
			return 1
		elif cur_dist <= 5:
			rospy.loginfo('[NAV][TRGT] Target reached!')
			self.prev_dist = 999999999
			return 0
		elif cur_dist > 14:
			self.prev_dist = cur_dist
			return 3
		elif cur_dist > 10:
			self.prev_dist = cur_dist
			return 4
		elif cur_dist > 6:
			self.prev_dist = cur_dist
			return 5
		else:
			self.prev_dist = cur_dist
			return 6

	def move_to_target(self):
		self.pick_direction()
		self.rotate_bot(self.angle_to_target)
		self.move_bot(self.linear_spd, 0.0)

	def map_region(self):
		self.move_to_target()

		rate = rospy.Rate(3)
		while not self.mapping_complete:
			if self.obstacle_detected:
				rospy.logwarn('[NAV][OBS] Obstacle detected')
				if self.is_rotating:
					self.move_bot(-1 * self.linear_spd, 0.0)
					self.is_rotating = False
				else:
					self.move_bot(-2 * self.linear_spd, 0.0)

				time.sleep(1)
				self.move_bot(0.0,0.0)
				time.sleep(1)
				self.move_to_target()
			elif self.inf_visible and abs(self.angle_to_target - self.yaw) > angular_tolerance:
				rospy.logwarn('[NAV][TRGT] Angular to exceeeded, adjusting now')
				self.move_bot(self.linear_spd, 0.0)
				self.move_to_target()
			else:
				target_condition = self.target_condition()
				if target_condition == 0 or target_condition == 1:
					if self.obstacle_detected:
						self.move_bot(-1*self.linear_spd, 0.0)
						time.sleep(1)
						self.move_bot(0.0,0.0)
						time.sleep(2)
					else:
						self.move_bot(0.0,0.0)
						time.sleep(3)
					self.move_to_target()
				elif target_condition == 2:
					self.move_bot(3*self.linear_spd, 0.0)
				elif target_condition == 3:
					self.move_bot(2*self.linear_spd, 0.0)
				elif target_condition == 4:
					self.move_bot(1.5*self.linear_spd, 0.0)
				elif target_condition == 5:
					self.move_bot(self.linear_spd, 0.0)
				elif target_condition == 6:
					self.move_bot(0.5*self.linear_spd, 0.0)

			rate.sleep()
		rospy.loginfo('its done mate')
		self.update_map()
		return self.occ_grid

	def return_to_home(self):
		rate = rospy.Rate(3)
		self.update_map()
		while True:
			if self.path_blocked(self.bot_position, self.home_loc):
				target = self.get_closest_edge(self.bot_position)
			else:
				target = self.home_loc

			self.pick_direction(target)
			self.display_map()
			self.rotate_bot(self.angle_to_target)
			self.move_bot(self.linear_spd, 0.0)

			if self.obstacle_detected:
				rospy.logwarn('[NAV][OBS] Obstacle detected!')
				self.move_bot(-1*self.linear_spd, 0.0)
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
				time.sleep(3)
				self.pick_direction()
				# self.rotate_bot(self.get_angle(self.cur_target))
				self.rotate_bot(self.angle_to_target)
				self.move_bot(self.linear_spd, 0.0)

			# elif self.angle_to_target > angular_tolerance:
			# 	rospy.logwarn('[NAV][TRGT] angular tolerance exceeeded, adjusting yaw')
			# 	# self.rotate_to_target()
			# 	self.pick_direction()
			# 	self.rotate_bot(self.get_angle(self.cur_target))
			# 	self.move_bot(self.linear_spd, 0.0)

			rate.sleep()

	def move_bot(self, linear_spd, angular_spd):
		pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		if linear_spd != 0.0 or angular_spd != 0.0:
			self.is_moving = True
		else:
			self.is_moving = False
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
		self.is_rotating = True
		
		rem_ang_dist = abs(current_yaw - target_yaw)
		# while c_change_dir * c_dir_diff > 0:
		while rem_ang_dist > 0.05:
			if self.obstacle_detected and any([x < 0.1 for x in self.lidar_data_front_wide]):
				self.move_bot(-self.linear_spd, 0.0)
				time.sleep(1)
				self.move_bot(0.0,0.0)
				time.sleep(1)
				self.move_to_target()
				# return
			current_yaw = np.copy(self.yaw)
			c_current_yaw = complex(math.cos(current_yaw), math.sin(current_yaw))
			c_dir_diff = np.sign((c_target_yaw/c_current_yaw).imag)
			rem_ang_dist = abs(current_yaw - target_yaw)
			
			if rem_ang_dist > math.pi:
				rem_ang_dist = (2*math.pi) - rem_ang_dist

			if rem_ang_dist > 3:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 4)
			elif rem_ang_dist > 2:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 4)
			elif rem_ang_dist > 1.5:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 3.5)
			elif rem_ang_dist > 1:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 2)
			elif rem_ang_dist > 0.4:
				self.move_bot(0.0, self.angular_spd * c_change_dir)
			else:
				self.move_bot(0.0, self.angular_spd * c_change_dir * 0.5)
			rate.sleep()

		# if abs(self.yaw - self.angle_to_target) > 0.2:
		# 	rospy.logwarn('[NAV][ROTN] Facing the wrong way, target possibly changed')
		# 	self.rotate_bot(self.angle_to_target)

		rospy.loginfo('[NAV][ROT] Finished rotating, now facing %f', self.yaw)
		self.is_rotating = False
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
		elif marker_type == 3:
			nav_marker.color.r = 1.0
			nav_marker.color.b = 1.0
			nav_marker.color.g = 0.0


		nav_marker.color.a = 1.0

		map_pub.publish(nav_marker)
		rospy.logdebug('[NAV][RVIZ] Published marker at %s, type: %d', str(pos), marker_type)