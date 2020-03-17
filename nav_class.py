#!/usr/bin/env python

import cv2
import numpy as np
import math
import cmath
import time

import rospy
from geometry_msgs.msg import Twist

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

class Navigation():
	def __init__(self, linear_spd, angular_spd):
		self._checked_positions = []
		self._target_position = ()
		self._route = []
		self._corners = []
		self._occ_map_raw = np.array([])
		self._occ_map = np.array([])

		self.yaw = 0.0
		self.laserscan_data = np.array([])
		self._bot_position = ()
		self.occ_grid = np.array([])

		self.linear_spd = linear_spd
		self.angular_spd = angular_spd

		self._mapping_complete = False

	def checked_pos_append(self, pos):
		self._checked_positions.append(pos)

	def set_target_pos(self, pos):
		self._target_position = pos

	def update_yaw(self, data):
		self.yaw = data

	def update_laserscan_data(self, data):
		self.laserscan_data = data

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
			for next_pos in [(i-1,j),(i,j+1),(i+1,j),(i,j-1)]:
				if next_pos not in self._checked_positions:
					if self.occ_grid[next_pos] == 0:
						rospy.loginfo('[NAV][OCC] Found closest unmapped region at %s', str(next_pos))
						return next_pos
					elif self.occ_grid[next_pos] == 1:
						pos_to_check.append(next_pos)
			self._checked_positions.append(cur_pos)

			if cur_pos in pos_to_check:
				pos_to_check.remove(cur_pos)

		rospy.loginfo('[NAV][OCC] No unmapped region found!, mapping complete')
		self._mapping_complete = True
		# return False
		self.display_map()
		exit()
	
	def update_map(self):
		ret, occ_map_raw = cv2.threshold(self.occ_grid, 2, 255, 0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
		self._occ_map_raw = cv2.dilate(occ_map_raw, element)
		self._occ_map = cv2.cvtColor(self._occ_map_raw, cv2.COLOR_GRAY2RGB)

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

	def get_direction(self, next_pos, cur_pos=None):
		if cur_pos is None:
			cur_pos = self._bot_position
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
				cur_point = corner_points[distances.index(max(distances))]
				if self.path_blocked(cur_pos, cur_point):
					corner_points.remove(cur_point)
					distances.remove(max(distances))
				else:
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


	def move_bot(self, linear_spd, angular_spd):
		pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		twist = Twist()
		twist.linear.x = linear_spd
		twist.angular.z = angular_spd
		time.sleep(1)
		pub.publish(twist)


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
				closest_corner = self.get_closest_corner(corner_list)
				while not self.path_blocked(cur_pos, closest_corner):
					corner_list.remove(closest_corner)
					closest_corner = self.get_closest_corner(corner_list)
				route_pos += 1
				cur_route.insert(route_pos, closest_corner)
		
		


		

