#!/usr/bin/env python

import cv2
import numpy as np
import math

import rospy
from geometry_msgs.msg import Twist

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

class Navigation():
	def __init__(self):
		self._checked_positions = []
		self._target_position = ()
		self._route = []
		self._corners = []
		self._occ_map = np.array([])

		self.yaw = 0.0
		self.laserscan_data = np.array([])
		self._bot_position = ()
		self.occ_grid = np.array([])

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
		pos_to_check = [self._bot_position]
		for cur_pos in pos_to_check:
			i,j = cur_pos[0],cur_pos[1]
			for next_pos in [(i-1,j),(i,j+1),(i+1,j),(i,j-1)]:
				if next_pos not in self._checked_positions:
					if self.occ_grid[next_pos] == 0:
						return next_pos
					elif self.occ_grid[next_pos] == 1:
						pos_to_check.append(next_pos)
			self._checked_positions.append(cur_pos)

			if cur_pos in pos_to_check:
				pos_to_check.remove(cur_pos)

		return False

	def get_corners(self):
		ret, occ_map_raw = cv2.threshold(self.occ_grid, 2, 255, 0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
		map_dilate = cv2.dilate(occ_map_raw, element)
		occ_map_bw, contours_ret, hierarchy = cv2.findContours(map_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		contours = contours_ret[0]
		self._occ_map = cv2.cvtColor(occ_map_bw, cv2.COLOR_GRAY2RGB)

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

	def get_closest_corner(self):
		self.get_corners()
		distances = [((x-self._bot_position[0])**2 + (y-self._bot_position[1])**2) for x,y in self._corners]
		return self._corners[distances.index(min(distances))]

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

	def get_direction(self):
		next_pos = self.get_closest_corner()
		# rospy.loginfo(next_pos)
		# rospy.loginfo(self._bot_position)
		return math.atan2((next_pos[1]-self._bot_position[1]),(next_pos[0]-self._bot_position[0]))+math.radians(30)

	def path_blocked(self, next_pos):
		path_img = np.zeros((len(self._occ_map),len(self._occ_map[0]), 1), np.uint8)
		cv2.line(path_img, (int(self._bot_position[0]), int(self._bot_position[1])), (next_pos), 255, thickness=1, lineType=8)

		# rospy.loginfo(np.any(np.logical_and(path_img, self._occ_map)))
		# cv2.imshow('MAP1', self._occ_map)
		# cv2.imshow('MAP2', path_img)
		# cv2.waitKey(3)
		return np.any(np.logical_and(path_img, self._occ_map))

	def target_reached(self, pos):
		if self._bot_position[0] in range(pos[0]-1, pos[0]+1,1) and self._bot_position in range(pos[1]-1,pos[1]+1,1):
			return True
		else:
			return False
	
	def move_circular(self, pos):
		pass

	def move_to_loc(self, pos):
		route = [pos]
		while not self.target_reached(pos):
			if self.path_blocked(route[0]):
				 route.insert(0, self.get_closest_corner())
			elif self.target_reached(route[0]):
				self.move_circular(route[0])
			time.sleep(1)

	def test_func(self, data):
		# pass
		self.path_blocked(data)

	def move_bot(self, linear_spd, angular_spd):
		pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		twist = Twist()
		twist.linear.x = linear_spd
		twist.angular.z = angular_spd
		time.sleep(1)
		pub.publish(twist)


	def map_region(self):
		while not self.mapping_complete:
			self.move_to_loc(self.get_closest_corner())