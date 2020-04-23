# auto_nav
The mapper script runs as a ros node to move a turtlebot through an unknown region and map the region while also locating an RGB target. Once the mapping is completed, the bot will return to the location of the target, aim and fire at it.

# Dependencies
## ROS master
The master has been tested to run on Ubuntu 16.04 and ROS Kinetic Kame. The master also requiress openCV and rviz in order to work

## ROS slave
The raspberry pi has been tested with the modified raspbian image provided [here](http://emanual.robotis.com/docs/en/platform/turtlebot3/raspberry_pi_3_setup/#install-linux-based-on-raspbian)
The raspberry pi also needs to have the raspicam_node package installed from [here](https://github.com/UbiquityRobotics/raspicam_node)

# Repo strucure
The latest stable version is available in the `master` branch and development work is done in the `nav_class` branch

# Software flow
![software flowchart](https://github.com/bbdhouzi/auto_nav/blob/master/reference_files/Software%20Block%20diagram.png)

# Procedure

The steps required to run the navigation program are as follows:
* Raspberry pi
  1. roslaunch turtlebot3_bringup turtlebot3_robot.launch
  2. roslaunch raspicam_node camerac1_1280x720.launch
  3. rosrun auto_nav shooter

* ROS master
  1. roslaunch turtlebot3_slam turtlebot3_slam.launch slam_mthods:=gmapping
  2. rosrun auto_nav target_id
  3. rosrun auto_nav mapper

It is recommended to create aliases for these within your `~/.bashrc` file to make it easier to start the nodes


# Important scripts
## nav_class.py

This file contains the Navigation class that performs all the navigation related work for the turtlebot.

**NOTE** in this class, edges refer to the ends of the walls/obstacles. corner refers to the offset point from the edge that the bot will aim to reach. A point is visible to another point if there are no obstacles in the closest path between them.

Here is a brief description of the main methods in the class

### update functions
The `Navigation.update_yaw(data)`, `Navigation.update_laserscan_data(data)`, `Navigation.update_occ_grid(grid, origin, resolution)`, `Navigation.update_bot_pos(data)` functions are used to update the instance of the class with the latest data available. The `occ_grid` provided to the object has all its values incremented by 1 to simplify the logic later on.

`Navigation.update_target_pos(data)` is used to record the location of the target once it has been identified.

`Navigation.update_map()` converts the occupancy grid into an image that can be used by openCV in the other functions and `Navigation.update_edges` gets a list of edges as defined above and stores it in an attribute.

### distance functions
`Navigation.get_dist_sqr(pos1, pos2)` returns the square of the distance between two points in the occupancy grid

`Navigation.get_dist(pos1, pos2)` return the square root of the value given by `get_dist_sqr(pos1,pos2)`

### obstacle_check()
This method checks for the presence of any obstacle in front of the bot to prevent it any collisions.

### get_nearest_unmapped_region()
This method scans the occupancy grid for the nearest unmapped region to the bot. In the context of the modified occupancy grid, unknown regions have a value of 0, empty regions have values of 1 and the probability of an obstacle ranges from 1 to 101.

The grid is scanned by traversing through the array until an unknown region is identified. Area beyond any obstacles are not considered as they will always be unknown and may lie outside the actual maze.

Mapping is considered complete if there are no unknown regions (0) adjacent to empty regions (1).

This method returns the location of the nearest unmapped region or (0,0) if mapping is complete.

###  get_closest_edge(pos)
This method return the closest edge to `pos`. This is found by finding the edge with the least distance between it and `pos` in the occupancy grid.

### get_furthest_visibile(pos)
This method returns the furthest corner to the closest edge that can be reached by the bot. The closest edge is found using the `get_closest_edge` method. This method finds 8 points around the edge that make up the corners and midpoints of a square of length 5 and finds the furthest visible, as defined above, one from `pos` by calculating the distances between `pos` and the visible corners in the occupancy grid. if no corners are visible, it will instead find the closest edge to the bot and the furthest visible corner from that edge.

### pick_direcion()
This method decides the next target location for the bot. This is done by first finding the nearest unmapped region to the bot and checking if there are any obstacles between the bot and this point. If there is no obstacle, it sets the unmapped region as the target. If there is an obstacle, it finds the closest edge to the unmapped region and the furthest visible corner from that edge.

Once the target has been identified, it finds the desired yaw of the bot in order to reach that point and the distance in the occupancy grid to the target.

If the required yaw differs from the current yaw by more than 90 degrees, the process above is run again. This is to endure that the bot does not travel in the wrong direction if it is working on outdated information.

###  path_blocked(cur_pos, next_pos)
This method uses opencv image manipulation to determine if there is an obstacle in the shortest path between the bot and its target position. it works by drawing a line between the current position and the target position on an image and checking if any points on the line overlap with any points on the walls detected in occupancy grid using openCVs bitwise_or function

### get_angle(pos)
This method returns the desired yaw for the bot to move towards its target position.

### target_condition()
This method checks how far the target is from the bot and whether it has reached the target. 

#### return values
0: Target reached
1: Overshot target, recheck direction
2: Target not reached, infinite distance ahead (0.0 in laserscan data)
3: Target not reached, > 14 units ahead
4: Target not reached, > 10 units ahead
5: Target not reached, > 6 units ahead
6: Target not reached, < 6 units ahead

### move_to_target()
This method calls the above definded functions to find the target location and then turn the bot to face the desired yaw and move forward.

### Navigaton.map_region()
This is the entry point for the class, this method starts begins the mapping procedure and moves the bot around the maze until the entire area has been mapped. the `target_condition` function is run repeatedly in a loop and the speed of the bot is adjusted depending on its return value. once the mapping is completed, it returns the occupancy grid of the mapped maze. 

### move_bot(linear_spd, angular_spd)
This method bot moves the bot with the speeds provided as the arguments

### rotate_bot(agle)
This method adjusts the yaw of the bot until it reaches the desired yaw given the `angle` argument

### rviz_marker(pos, marker_type)
This method adds a point on the rviz display to see where the bot is headed and the points of interest it identitifed.

**unmapped regions** have a *marker type of 0* and appear **red** on rviz
**furthest visible corners** have a *marker type of 1* and appear **green** on rviz
**closest edges** have a *marker type of 2* and appear **blue** on rviz
**other points of interest** found when there are no furthest visible corners have a *marker type of 3* and appear **pink** on rviz

## mapper

This is the main node that has to be run on the ROS master device, it uses data from `/map` and `/scan` topics along with openCV to autonomously navigate and map an unknown region.

It uses the "Navigation" class defined in `nav_class.py` to navigate the area.

The vertical amount published on the `/target/vertical` is the offset as a percentage from the center of the of the cameras view. It is used to find the desired angle of the firing mechanism. The vertical field of view of the camera is 42.4 degrees. Since the offset is from the center, the 21.2 is used to calculate the angle offset. `cam_angle` refers to the angle of the camera from the horizontal.

## target_id
This node gets the image the camera connected to the bot and scans the images for the RGB target. If a target is identified, it publishes `True` to the `/target/identified` topic. the vertical and horizontal distance from the center of the frame to the center of the target is found as values ranging from 0 to 100 and published to the `/target/horizontal` and `/target/vertical` topics. 

## shooter
This is the only node that has to be run on the bot itself. This node handles the firing of the ping pong ball. It receives the vertical angle required for shooting the ball in the `/servo/aiming` topic and adjusts the angle of the cannon accordingly. when it receives a `True` message in `/servo/firing`, the bot will trigger its firing mechanism. This involves change of angle of one servo (connected the BCM pin 13 on the raspberry pi) to 40 degrees for 0.6 seconds and then returning back to 0. The value for the angle and time were determined experimentally to be the optimum values. Once the trigger mechanism has been triggered, it will also engage the relaoding mechanism which will adjust the angle of the servo to allow one ball to drop into the chamber.
