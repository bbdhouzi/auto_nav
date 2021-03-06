#!/usr/bin/env python

import cv2
import numpy as np
import math
import cmath
import time

import rospy
from geometry_msgs.msg import Twist
<<<<<<< HEAD
=======
from visualization_msgs.msg import Marker

import matplotlib.pyplot as plt
from PIL import Image
>>>>>>> nav_class

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

<<<<<<< HEAD
class Navigation():
	def __init__(self, linear_spd, angular_spd):
		self._checked_positions = []
		self._target_position = ()
		self._route = []
		self._corners = []
		self._occ_map_raw = np.array([])
		self._occ_map = np.array([])
=======
def get_rot_coord(image, angle, pos):
	image_center = tuple(np.array(image.shape[1::-1])/2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	pos_mat = np.array([[pos[0]],[pos[1]], [1]])
	return rot_mat.dot(pos_mat)

class Navigation():
	def __init__(self, linear_spd, angular_spd):
		self._checked_positions = []
		self._edges = []
		self._occ_map_raw = np.array([])
		self._occ_map = np.array([])
		self.map_origin = ()
		self.map_width = 0
		self.map_height = 0

		self._edge_map = np.array([])

		self.marker_no = 0

		self._main_route = []		#this is the actual route followed by the bot| list of corners
		self._pseudo_route = []		#this is the list of edges the bot has to go around| list of edges
>>>>>>> nav_class

		self.yaw = 0.0
		self.laserscan_data = np.array([])
		self._bot_position = ()
		self.occ_grid = np.array([])

		self.linear_spd = linear_spd
		self.angular_spd = angular_spd

		self._mapping_complete = False

<<<<<<< HEAD
	def checked_pos_append(self, pos):
		self._checked_positions.append(pos)

	def set_target_pos(self, pos):
		self._target_position = pos

=======
	## these methods are to update the sensor data from the main file
>>>>>>> nav_class
	def update_yaw(self, data):
		self.yaw = data

	def update_laserscan_data(self, data):
		self.laserscan_data = data

<<<<<<< HEAD
	def update_occ_grid(self, data):
		self.occ_grid = data

	def update_bot_pos(self, pos):
		self._bot_position = pos

	def mapping_complete(self):
		return self._mapping_complete

	def get_nearest_unmapped_region(self):
		rospy.loginfo('[NAV][OCC] Finding the closest unmapped region')
		pos_to_check = [self._bot_position]
		for cur_pos in pos_to_check:
			i,j = cur_pos[0],cur_pos[1]
=======
	def update_occ_grid(self, grid, origin):
		self.occ_grid = grid
		self.map_origin = origin
		self.map_width = len(grid[0])
		self.map_height = len(grid)

	# bot pos is stored as (x,y) coordinate
	def update_bot_pos(self, pos):
		self._bot_position = pos
	## end of sensor data update methods

	def get_next_target(self):
		return self._main_route[0]

	# returns the square of the distance between 2 points
	# mostly used to compare distances so exact distance is not required
	def get_distance(self, pos1, pos2):
		return (pos2[0]-pos1[0])**2 + (pos2[1] - pos1[1])**2

	# Find the nearest unmapped region (-1 in occ_grid)
	# returns False if there are no unmapped regions, i.e mapping has been completed
	def get_nearest_unmapped_region(self):
		rospy.loginfo('[NAV][OCC] Finding the closest unmapped region')
		pos_to_check = [self._bot_position]
		self._checked_positions = []
		# rospy.loginfo(self.occ_grid)
		for cur_pos in pos_to_check:
			# i,j = cur_pos[0],cur_pos[1]
			i,j = cur_pos[1],cur_pos[0]
			rospy.loginfo('[MAP] %s', str(cur_pos))
>>>>>>> nav_class
			for next_pos in [(i-1,j),(i,j+1),(i+1,j),(i,j-1)]:
				if next_pos not in self._checked_positions:
					if self.occ_grid[next_pos] == 0:
						rospy.loginfo('[NAV][OCC] Found closest unmapped region at %s', str(next_pos))
<<<<<<< HEAD
						return next_pos
=======
						self._pseudo_route.insert(0, next_pos)
						return (next_pos[1],next_pos[0])
>>>>>>> nav_class
					elif self.occ_grid[next_pos] == 1:
						pos_to_check.append(next_pos)
			self._checked_positions.append(cur_pos)

			if cur_pos in pos_to_check:
				pos_to_check.remove(cur_pos)

<<<<<<< HEAD
		rospy.loginfo('[NAV][OCC] No unmapped region found!, mapping complete')
		self._mapping_complete = True
		# return False
		self.display_map()
		exit()
	
	def update_map(self):
		ret, occ_map_raw = cv2.threshold(self.occ_grid, 2, 255, 0)
=======
		rospy.loginfo('length of array: %d', len(self._checked_positions))
		rospy.loginfo('[NAV][OCC] No unmapped region found!, mapping complete')
		self._mapping_complete = True
		self.display_map()
		return False	
		# exit()
	
	# update occ_map and occ_map_raw
	def update_map(self):
		ret, occ_map_raw = cv2.threshold(self.occ_grid, 1, 255, 0)
>>>>>>> nav_class
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
		self._occ_map_raw = cv2.dilate(occ_map_raw, element)
		self._occ_map = cv2.cvtColor(self._occ_map_raw, cv2.COLOR_GRAY2RGB)

<<<<<<< HEAD
	def get_corners(self):
		self.update_map()
		occ_map_bw, contours_ret, hierarchy = cv2.findContours(self._occ_map_raw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		for contour in contours_ret:
			approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
			n = approx.ravel()
			i = 0
			for j in n:
				if i%2 == 0:
					x = n[i]
					y = n[i+1]

					self._corners.append((x,y))
				i += 1

	def get_closest_corner(self, cur_pos=None, corner_list=None):
		if cur_pos is None:
			cur_pos = self._bot_position
		
		if corner_list is None:
			corner_list = self._corners
		else:
			self.get_corners()

		# self.get_corners()
		distances = [((x-self._bot_position[0])**2 + (y-self._bot_position[1])**2) for x,y in corner_list]
		return corner_list[distances.index(min(distances))]

	def display_map(self): 
		self.get_corners()
		occ_map = cv2.circle(self._occ_map, (int(self._bot_position[1]),int(self._bot_position[0])), 1, (0,0,255), -1)
		map_overlay = np.zeros((len(self._occ_map),len(self._occ_map[0]), 3), np.uint8)
		for i in range(1,len(self._corners)):
			cv2.line(map_overlay, (self._corners[i-1]),(self._corners[i]), (0,255,0), thickness=1, lineType=8)
			cv2.circle(map_overlay, (self._corners[i-1][0],self._corners[i-1][1]), 1, (255,0,0),-1)
		
		# map_overlay[self.get_closest_corner()] = (0,0,255)
		closest_corner = self.get_closest_corner()
		occ_map = cv2.circle(occ_map, (int(closest_corner[1]),int(closest_corner[0])), 1, (0,255,255), -1)
		# map_overlay = rotate_image(map_overlay, np.degrees(self.yaw)+180)

		occ_map_disp = np.zeros((len(self._occ_map), len(self._occ_map[0]), 3), np.uint8)
		occ_map_disp = cv2.bitwise_or(occ_map, map_overlay, occ_map_disp)

		cv2.imshow('MAP', occ_map_disp)
		# cv2.imshow('MAP3', occ_map)
		# cv2.imshow('MAP4', map_overlay)
		# cv2.imshow('MAP2', self._occ_map)
		cv2.waitKey(3)
=======
	#find the edges of the visible walls
	def update_edges(self):
		self.update_map()
		self._edge_map = np.zeros((self.map_height, self.map_width, 3), np.uint8)

		occ_map_bw, contours_ret, hierarchy = cv2.findContours(self._occ_map_raw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
		for cnt_idx in range(len(contours_ret)):
			# if hierarchy[0][cnt_idx][3] != -1:
			if True:
				approx = cv2.approxPolyDP(contours_ret[cnt_idx], 0.009*cv2.arcLength(contours_ret[cnt_idx], True), True)
				n = approx.ravel()
				for i in range(0,len(n),2):
					x = n[i]
					y = n[i+1]

					self._edges.append((x,y))
					cv2.circle(self._edge_map, (x,y), 3, (255,0,0), -1)

		if len(self._edges) == 0:
			rospy.loginfo('[NAV][EDGE] No edges detected, there is likely an error')
			self.show_map()

	# Find the nearest edge to a particular position,
	# inputs: 
	# 	pos - position to check for
	# 	edge_list - list of edges to check against, assumed as latest edge list if not provided
	def get_closest_edge(self, pos, edge_list=None):
		if edge_list is None:
			self.update_edges()
			edge_list = self._edges

		rospy.loginfo('[NAV][EDGE] Locating the nearest edge to %s', str(pos))
		distances = [self.get_distance(edge_pos, pos) for edge_pos in edge_list]
		rospy.loginfo(distances)
		return edge_list[distances.index(min(distances))]

	# uses opencv line overlap to check if the path to next_pos is blocked from cur_pos.
	# cur_pos is assumed to be the bots current position if not provided
	def path_blocked(self, next_pos, cur_pos=None):
		self.update_map()
		if cur_pos is None:
			cur_pos = self._bot_position
		
		path_img = np.zeros((len(self._occ_map),len(self._occ_map[0]), 1), np.uint8)
		cv2.line(path_img, (int(cur_pos[0]), int(cur_pos[1])), (next_pos), 255, thickness=1, lineType=8)

		cv2.imshow('PATH', path_img)
		cv2.waitKey(0)

		return np.any(np.logical_and(path_img, self._occ_map_raw))

	def show_map(self):
		img1 = Image.fromarray(self._occ_map)
		img2 = Image.fromarray(self._edge_map)

		plt.imshow(img1)
		# plt.imshow(img2)

		plt.draw_all()
		plt.pause(0.00000001)
		plt.show()

	def display_map(self):
		self.update_edges() # also calls self.update_map()

		occ_map = cv2.circle(self._occ_map, self._bot_position, 3, (0,0,255), -1)
		map_overlay = np.zeros((len(self._occ_map),len(self._occ_map[0]), 3), np.uint8)
		# cv2.circle(map_overlay, self._bot_position, 3, (0,0,255, -1))

		unmapped_region = self.get_nearest_unmapped_region()
		cv2.circle(map_overlay, unmapped_region, 3, (128,128,256), -1)

		closest_edge = self.get_closest_edge(unmapped_region)
		cv2.circle(map_overlay, closest_edge, 3, (0,255,0), -1)
		# map_overlay = rotate_image(map_overlay, np.degrees(self.yaw)+180)

		# rot_pos = get_rot_coord(self._edge_map,np.degrees(self.yaw)+180, closest_edge)
		# print(rot_pos)
		# cv2.circle(self._edge_map, (rot_pos[0], rot_pos[1]), 3, (0,255,255), -1)

		occ_map_disp = np.zeros((len(self._occ_map), len(self._occ_map[0]), 3), np.uint8)

		# map_overlay = cv2.bitwise_or(map_overlay, self._edge_map, map_overlay)
		occ_map_disp = cv2.bitwise_or(occ_map, map_overlay, occ_map_disp)

		# img = Image.fromarray(occ_map_disp)
		# plt.imshow(img)
		# plt.draw_all()
		# plt.pause(0.00000001)
		# plt.show()

		cv2.imshow('occ_map_disp', occ_map_disp)
		cv2.imshow('occ_map', occ_map)
		# cv2.imshow('map_overlay', map_overlay)
		# cv2.imshow('edges', self._edge_map)
		# cv2.imshow('MAP2', self._occ_map)
		cv2.waitKey(0)
>>>>>>> nav_class

	def get_direction(self, next_pos, cur_pos=None):
		if cur_pos is None:
			cur_pos = self._bot_position
<<<<<<< HEAD
		return math.atan2((next_pos[1]-cur_pos[1]),(next_pos[0]-cur_pos[0]))+math.radians(15)

	def path_blocked(self, next_pos, cur_pos=None):
		self.update_map()
		if cur_pos is None:
			cur_pos = self._bot_position
		
		# rospy.loginfo(cur_pos)
		# rospy.loginfo(next_pos)
		path_img = np.zeros((len(self._occ_map),len(self._occ_map[0]), 1), np.uint8)
		cv2.line(path_img, (int(cur_pos[0]), int(cur_pos[1])), (next_pos), 255, thickness=1, lineType=8)

		# rospy.loginfo(np.any(np.logical_and(path_img, self._occ_map)))
		# cv2.imshow('MAP1', self._occ_map)
		# cv2.imshow('MAP2', path_img)
		# cv2.waitKey(3)
		return np.any(np.logical_and(path_img, self._occ_map))

	def target_reached(self, pos):
		if not self.mapping_complete():
			if self.path_blocked(pos):
				return False
			else:
				return True
		# i,j = self._bot_position[0],self._bot_position[1]
		# if pos in [(i,j), (i-1,)]
		# if self._bot_position[0] in range(pos[0]-1, pos[0]+1,1) and self._bot_position in range(pos[1]-1,pos[1]+1,1):
		# 	return True
		# else:
		# 	return False
	
	def get_distance(self, pos1, pos2):
		return (pos2[0]-pos1[0])**2 + (pos2[1] - pos1[1])**2

	def set_target(self, next_pos, cur_pos=None):
		if cur_pos is None:
			cur_pos = self._bot_position
		
		if self.path_blocked(next_pos, cur_pos):
			target_pos = self.get_closest_corner()
			i,j = target_pos[0],target_pos[1]
			corner_points = [(i-1,j-1), (i-1,j+1),(i+1,j+1),(i+1,j-1)]
			distances = [self.get_distance(point, cur_pos) for point in corner_points]
			while len(distances) > 0:
=======

		# rot_pos = get_rot_coord(self._occ_map, np.degrees(self.yaw), next_pos)
		rot_pos = next_pos
		# return math.atan2((rot_pos[1]-cur_pos[1]),(rot_pos[0]-cur_pos[0]))
		rospy.loginfo('rot_pos: %s', str(rot_pos))
		rospy.loginfo('cur_pos: %s', str(cur_pos))
		j_dist = rot_pos[1] - cur_pos[1]
		i_dist = rot_pos[0] - cur_pos[0]
		rospy.loginfo('tan %d/%d', i_dist, j_dist)
		return math.atan2((rot_pos[1]-cur_pos[1]),(rot_pos[0]-cur_pos[0]))

	def target_reached(self, pos):
		if not self._mapping_complete:
			return not self.path_blocked(pos)

	# find the actual target the bot should move towards
	# if next_pos is not specified, it is assumed to be the first target in self.pseudo_route
	# if cur_pos is not specified, it is assumed to be the bots current position
	def set_target(self, next_pos=None, cur_pos=None):
		if cur_pos is None:
			cur_pos = self._bot_position
		if next_pos is None:
			next_pos = self._pseudo_route[0]
		
		rospy.loginfo('[NAV][TRGT] Choosing the best path to reach %s from %s', next_pos, cur_pos)
		radius = 1
		if self.path_blocked(next_pos, cur_pos):
			target_pos = self.get_closest_edge()
			i,j = target_pos[0],target_pos[1]
			corner_points = [(i-radius,j-radius), (i-radius,j+radius),(i+radius,j+radius),(i+radius,j-radius)]
			distances = [self.get_distance(point, cur_pos) for point in corner_points]
			while len(distances) > 0:
				rospy.loginfo(distances)
>>>>>>> nav_class
				cur_point = corner_points[distances.index(max(distances))]
				if self.path_blocked(cur_pos, cur_point):
					corner_points.remove(cur_point)
					distances.remove(max(distances))
				else:
<<<<<<< HEAD
					self._route.insert(0,cur_point)
					self.display_map()
					return
			rospy.loginfo('[NAV][TRGT] weird error no visible target points')

	def move_to_loc(self, pos):
		rospy.loginfo('[NAV] moving to %s', pos)
		self._route = [pos]
		rate = rospy.Rate(1)
		while not self.target_reached(pos):
			self.set_target(pos)
			angle = self.get_direction(self._route[0])
			self.rotate_bot(angle)
			self.move_bot(self.linear_spd, 0.0)
			if not self.target_reached(self._route[0]):
				rate.sleep()
			else:
				return True
			
	def test_func(self, data):
		# pass
		# self.path_blocked(data)
		# self.rotate_bot(30)
		# self.move_bot(self.linear_spd, 0.0)
		# time.sleep(5)
		# self.move_bot(0.0, 0.0)
		next_pos = self.get_nearest_unmapped_region()
		set_target(next_pos)
		move_bot()

=======
					rospy.loginfo('[NAV][TRGT] Found optimal target at %s', cur_point)
					self._main_route.insert(0,cur_point)
					self.set_marker(cur_point)
					# self.display_map()
					return cur_point
			rospy.loginfo('[NAV][TRGT] weird error no visible target points')
		else:
			self._main_route.insert(0, next_pos)

	def move_to_loc(self, pos):
		rospy.loginfo('[NAV] moving to %s', pos)
		self._main_route = [pos]
		rate = rospy.Rate(1)
		while not self.target_reached(self._main_route[0]):
			self.set_target(self._main_route[0])
			# angle = self.get_direction(self._route[0])
			angle = self.get_direction(self._main_route[0])
			self.rotate_bot(angle)
			self.move_bot(self.linear_spd, 0.0)
			if not self.target_reached(self._main_route[0]):
				rate.sleep()
			else:
				self._main_route.pop(1)
				# return True
			
	def test_func(self, data):
		# pass
		# self.map_region()
		unmapped_region = self.get_nearest_unmapped_region()

		target = ()
		if self.path_blocked(unmapped_region, self._bot_position):
			rospy.loginfo('PATH BLOCKED')
			target = self.get_closest_edge(unmapped_region)
		else:
			rospy.loginfo('FREE MAP')
			target = unmapped_region

		rospy.loginfo('TARGET: %s', str(target))

		# next_pos = self.get_closest_edge(unmapped_region)
		self.display_map()
		rospy.loginfo('GETTING ANGLE')
		angle = self.get_direction((target[1],target[0]))
		self.rotate_bot(math.degrees(self.yaw) + math.degrees(angle) + 180)
		self.move_bot(self.linear_spd, 0.0)
		# self.rotate_bot(180)
		# self.move_bot(0.0, 1.0)
>>>>>>> nav_class

	def move_bot(self, linear_spd, angular_spd):
		pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		twist = Twist()
		twist.linear.x = linear_spd
		twist.angular.z = angular_spd
		time.sleep(1)
		pub.publish(twist)

<<<<<<< HEAD

	def map_region(self):
		rospy.loginfo('[NAV][MAP] Starting to map the unknown region')
		# while not self.mapping_complete():
		while True:
			next_pos = self.get_nearest_unmapped_region()
			# rospy.loginfo(next_pos)
			if not next_pos:
				return
			else:
				self.move_to_loc(next_pos)
	
	def rotate_bot(self, angle):
		rate = rospy.Rate(1)
		current_yaw = np.copy(self.yaw)
		c_yaw = complex(math.cos(current_yaw), math.sin(current_yaw))
		target_yaw = current_yaw + angle
		c_target_yaw = complex(math.cos(target_yaw), math.sin(target_yaw))
		rospy.loginfo("[NAV][YAW] current: %s, desired: %s", str(math.degrees(cmath.phase(c_yaw))), str(math.degrees(cmath.phase(c_target_yaw))))

		c_change = c_target_yaw / c_yaw
		c_change_dir = np.sign(c_change.imag)
		self.move_bot(0.0, (self.angular_spd * c_change_dir))
		
		c_dir_diff = c_change_dir
		while (c_change_dir * c_dir_diff > 0):
			current_yaw = np.copy(self.yaw)
			c_yaw = complex(math.cos(current_yaw), math.sin(math.sin(current_yaw)))
			rospy.loginfo("[NAV][YAW] current: %s, desired: %s", str(math.degrees(cmath.phase(current_yaw))), str(math.degrees(cmath.phase(target_yaw))))

			c_change = c_target_yaw / c_yaw
			c_dir_diff = np.sign(c_change.imag)
			rate.sleep()
		
		self.move_bot(0.0, 0.0)
		rospy.loginfo('[NAV] Facing the right direction')
=======
	# def rotate_bot(self, angle):
	# 	rate = rospy.Rate(1)
	# 	current_yaw = np.copy(self.yaw)
	# 	c_yaw = complex(math.cos(current_yaw), math.sin(current_yaw))
	# 	target_yaw = current_yaw + angle
	# 	c_target_yaw = complex(math.cos(target_yaw), math.sin(target_yaw))
	# 	rospy.loginfo("[NAV][YAW] current: %s, desired: %s", str(math.degrees(cmath.phase(c_yaw))), str(math.degrees(cmath.phase(c_target_yaw))))

	# 	c_change = c_target_yaw / c_yaw
	# 	c_change_dir = np.sign(c_change.imag)
	# 	self.move_bot(0.0, (self.angular_spd * c_change_dir))
		
	# 	c_dir_diff = c_change_dir
	# 	while (c_change_dir * c_dir_diff > 0):
	# 		current_yaw = np.copy(self.yaw)
	# 		c_yaw = complex(math.cos(current_yaw), math.sin(math.sin(current_yaw)))
	# 		rospy.loginfo("[NAV][YAW] current: %s, desired: %s", str(math.degrees(cmath.phase(c_yaw))), str(math.degrees(cmath.phase(c_target_yaw))))

	# 		c_change = c_target_yaw / c_yaw
	# 		c_dir_diff = np.sign(c_change.imag)
	# 		rate.sleep()
		
	# 	self.move_bot(0.0, 0.0)
	# 	rospy.loginfo('[NAV] Facing the right direction')

	def move_to_target(self, target_pos):
		rate = rospy.Rate(1)
		angle = self.get_direction(target_pos)
		self.rotate_bot(angle)
		self.move_bot(self.linear_spd, 0.0)
		while not self.target_reached(target_pos):
			distance = self.get_distance(self._bot_position, target_pos)
			rospy.loginfo('[NAV][MOV] Remaining distance: %d', distance)
			rate.sleep()
		return

	def map_region(self):
		rospy.loginfo('[NAV][MAP] Starting to map the unknown region')
		while not self.mapping_complete():
			self.get_nearest_unmapped_region()
			self.set_target()
			# self.display_map()
			self.set_marker(self._main_route[0])
			# self.move_to_target(self._main_route[0])
		# while not self.mapping_complete():
		# while True:
		# 	next_pos = self.get_nearest_unmapped_region()
		# 	rospy.loginfo(next_pos)
		# 	if not next_pos:
		# 		return
		# 	else:
		# 		self.move_to_loc(next_pos)
>>>>>>> nav_class
	
	def set_route(self, target_pos, cur_pos=None):
		if cur_pos is None:
			cur_pos = self._bot_position
		# self._route = [target_pos]
		cur_route = [cur_pos, target_pos]
		route_pos = 0
		while True:
			if self.path_blocked(cur_route[route_pos], cur_route[-1]):
				self.get_corners()
				corner_list = self._corners.copy()
<<<<<<< HEAD
				closest_corner = self.get_closest_corner(corner_list)
				while not self.path_blocked(cur_pos, closest_corner):
					corner_list.remove(closest_corner)
					closest_corner = self.get_closest_corner(corner_list)
				route_pos += 1
				cur_route.insert(route_pos, closest_corner)
		
		


		
=======
				closest_corner = self.get_closest_edge(corner_list)
				while not self.path_blocked(cur_pos, closest_corner):
					corner_list.remove(closest_corner)
					closest_corner = self.get_closest_edge(corner_list)
				route_pos += 1
				cur_route.insert(route_pos, closest_corner)

	def set_marker(self, pos):
		map_pub = rospy.Publisher('nav_markers', Marker, queue_size=10)
		x_pos = (pos[0] - self._bot_position[1]) * 0.05
		y_pos = (pos[1] - self._bot_position[0]) * 0.05

		rospy.sleep(2)
		nav_marker = Marker()
		nav_marker.header.frame_id = "base_link"
		nav_marker.ns = "marker_test"
		nav_marker.id = self.marker_no
		nav_marker.pose.position.x = x_pos
		nav_marker.pose.position.y = y_pos
		nav_marker.type = nav_marker.CUBE
		nav_marker.action = nav_marker.MODIFY
		nav_marker.scale.x = 1.0
		nav_marker.scale.y = 1.0
		nav_marker.scale.z = 1.0
		nav_marker.pose.orientation.w = 1.0
		nav_marker.color.r = 1.0
		nav_marker.color.g = 0.0
		nav_marker.color.b = 0.0
		nav_marker.color.a = 1.0

		map_pub.publish(nav_marker)
		rospy.loginfo('Published marker at %s', str(pos))
		self.marker_no += 1
		
	def rotate_bot(self, rot_angle):
	
		# create Twist object
		twist = Twist()
		# set up Publisher to cmd_vel topic
		pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
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

>>>>>>> nav_class

