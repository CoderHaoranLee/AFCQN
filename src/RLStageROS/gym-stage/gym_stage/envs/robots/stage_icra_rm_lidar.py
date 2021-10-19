import gym
import rospy
import roslaunch
import time
import numpy as np
from gym import utils, spaces
from gym_stage.envs import stage_env
from geometry_msgs.msg import Twist, PoseStamped
from std_srvs.srv import Empty
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from nav_msgs.msg import OccupancyGrid, Path
from gym.utils import seeding
import cv2
import matplotlib.pyplot as plt
from messages.msg import MapIndex
from math import *
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker


class StageICRALidarNNEnv(stage_env.StageEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        stage_env.StageEnv.__init__(self, "StageICRALidar_v0.launch") # TODO
        self.goal_pub_ = rospy.Publisher("/map_index", MapIndex, queue_size=10)
        self.unpause = rospy.ServiceProxy('/stage/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/stage/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/stage/reset_simulation', Empty)
        self.reset_pub = rospy.Publisher('/slam_reset', Bool, queue_size=5)

        self.true_map = cv2.imread("/root/ros_workspaces/ros_codes/catkin_ws/src/RLStageROS/world/icra.pgm") # TODO
        self.true_map = np.asarray(self.true_map)
        self.true_H, self.true_W, _ = self.true_map.shape
        self.true_half_H, self.true_half_W = 82, 20

        self.last_occu_count = None
        self.last_map_image = None

        self.map_resolution = None
        self.map_origin = None
        self.reward_range = (-np.inf, np.inf)

        self.goal_x_set_ = np.arange(0, 8, 1)
        self.goal_y_set_ = np.arange(4, -1, -1)
        self.goal_x_set_, self.goal_y_set_ = np.meshgrid(self.goal_x_set_, self.goal_y_set_)
        self.goal_x_set_ = self.goal_x_set_.reshape(-1)
        self.goal_y_set_ = self.goal_y_set_.reshape(-1)

        self.state_shape = [40, 64]

        self.last_action = None
        self.action_history = None

        self.robo_pose = None
        self.last_pose_mask = None
        self.consider_last_pose = True
        self.test = False

        #  listen to frontier
        self.maker_sub_ = rospy.Subscriber('/explore/frontiers', MarkerArray, self.maker_callback)
        self.frontiers = None
        self.use_frontiers = False

        self.local_costmap_sub_ = rospy.Subscriber('/local_costmap/local_costmap/costmap', OccupancyGrid, self.convert_local_map_to_image)
        self.local_map_image_pub_ = rospy.Publisher('local_map/image_raw', Image, queue_size=2)
        self.bridge_ = CvBridge()

        self.pose_listener_ = tf.TransformListener()

        self._seed()

    def maker_callback(self, makers):
        if self.robo_pose is None:
            return
        frontiers = []
        for m in makers.markers:
            if m.type == Marker.POINTS:
                points = m.points
                for p in points:
                    grid_x = self.true_half_W + (p.x * 20 )
                    grid_y = self.true_half_H - (p.y * 20 )
                    if grid_x >= self.true_W-1:
                        grid_x = self.true_W-1
                    if grid_x < 0:
                        grid_x = 0
                    if grid_y < 0:
                        grid_y = 0
                    if grid_y >= self.true_H-1:
                        grid_y = self.true_H-1
                    frontiers.append([int(grid_x), int(grid_y)])
        self.frontiers = frontiers

    def get_frontier_map(self, frontiers):
        frontiers_map = np.zeros((self.true_H, self.true_W))
        if frontiers is not None:
            if len(frontiers) > 0:
                frontiers_idx = np.asarray(frontiers)
                frontiers_map[frontiers_idx[:, 1], frontiers_idx[:, 0]] = 1
        frontiers_map = cv2.resize(frontiers_map, (self.state_shape[1], self.state_shape[0]))
        frontiers_map = np.expand_dims(frontiers_map, axis=-1)
        return frontiers_map

    def get_state_size(self):
        return [self.true_H/2, self.true_W/2, 2]

    def get_action_size(self):
        return 41

    def convert_map_to_image(self, data):
        resolution = data.info.resolution
        H, W = data.info.height, data.info.width
        map_origin_x = data.info.origin.position.x
        map_origin_y = data.info.origin.position.y
        map_pixel_x = -int(map_origin_x / resolution)
        map_pixel_y = H + int(map_origin_y / resolution)

        map_data = np.asarray(data.data)
        idx_equ0 = map_data == 0
        idx_less0 = map_data < 0
        idx_greater0 = map_data > 0
        map_data[idx_less0] = 128.0
        map_data[idx_greater0] = 255.0

        map_data = map_data.reshape(H, W)
        map_data = np.flipud(map_data)

        copy_map_data = np.ones((self.true_H, self.true_W)) * 128.0

        copy_map_data[max(0, self.true_half_H - map_pixel_y):min(self.true_H, self.true_half_H + (H - map_pixel_y)),
        max(0, self.true_half_W - map_pixel_x):min(self.true_W, self.true_half_W + (W - map_pixel_x))] = map_data[max(0,map_pixel_y - self.true_half_H):min(H, map_pixel_y + (self.true_H - self.true_half_H)),
                                                                                                         max(0, map_pixel_x - self.true_half_W):min(W, map_pixel_x + (self.true_W - self.true_half_W))]
        return copy_map_data

    def convert_local_map_to_image(self, data):
        resolution = data.info.resolution
        H, W = data.info.height, data.info.width
        map_data = np.asarray(data.data)
        map_data = map_data.reshape(H, W)
        map_data = np.fliplr(map_data)
        map_data = np.asarray(map_data, np.uint8)
        
        ## align the direction of the car
        try:
            (trans,rot) = self.pose_listener_.lookupTransform('/map', '/base_link', rospy.Time(0))
            # print trans
            (roll, pitch, yaw) = euler_from_quaternion (rot)
            # print roll, pitch, yaw
            if yaw < 0:
                yaw += 2 * pi
            yaw = yaw * 180 / pi
            center = (W/2, H/2)
            M = cv2.getRotationMatrix2D(center, yaw, 1)     
            map_data = cv2.warpAffine(map_data, M, (W, H))

            msg = self.bridge_.cv2_to_imgmsg(map_data, '8UC1')
            self.local_map_image_pub_.publish(msg)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "cannot find the transformation between map and base_link"
        

        

        # print "local map shape:", map_data.shape
        # return map_data
    
    # def wait_for_local_map(self):
    #     # print "waiting for local cost map"
    #     try:
    #         current_local_map = rospy.wait_for_message('/local_costmap/local_costmap/costmap', OccupancyGrid, timeout=5)
    #         map_image = self.convert_local_map_to_image(current_local_map)
    #         msg = self.bridge_.cv2_to_imgmsg(map_image, 'bgr8')
    #         self.local_map_image_pub_.publish(msg)
    #         print "local map shape:", map_image.shape
    #     except:
    #         print "cannot received local map"

    def waiting_for_static_map(self):
        # print 'waiting for static map ... '
        try:
            last_map = rospy.wait_for_message('/map', OccupancyGrid, timeout=5)
        except:
            print 'cannot received map'
            last_map = None
        count_static = 0
        static_map = None
        while count_static < 5:
            try:
                current_map = rospy.wait_for_message('/map', OccupancyGrid, timeout=5)
                self.map_resolution = current_map.info.resolution
                self.map_origin = current_map.info.origin
            except:
                print 'cannot received map, too'
                current_map = None
            if last_map is not None and current_map is not None:
                last_map_image = self.convert_map_to_image(last_map)
                current_map_image = self.convert_map_to_image(current_map)
                delta_map_image = 1000
                if last_map_image.shape[0] == current_map_image.shape[0] and last_map_image.shape[1] == \
                        current_map_image.shape[1]:
                    delta_map_image = np.sum(last_map_image - current_map_image)
                if delta_map_image < 100:
                    count_static += 1
                    static_map = current_map_image
                else:
                    count_static = 0
        return static_map

    def map_to_observation(self, map):
        map = cv2.resize(map, (self.state_shape[1], self.state_shape[0]))
        # map_edge = np.uint8(map)
        # map_edge[map_edge==128] = 255
        # edge = cv2.Canny(map_edge, 1, 10)
        # plt.imshow(edge)
        # plt.show()
        map = np.expand_dims(map, axis=-1)
        return map / 255.0

    def compare_map(self, map):
        map = cv2.resize(map, (self.true_W, self.true_H))
        map_explored = np.where(map==0)[0].shape[0]
        true_explored = np.where(self.true_map[:, :, 0]==255)[0].shape[0]
        rate = float(map_explored) / float(true_explored)
        return rate

    def generate_pose_mask(self, x, y):
        zero_slide = np.zeros((self.true_H, self.true_W))
        y = self.true_H - y
        # print x, y
        x = int(x)
        y = int(y)
        zero_slide[y-5:y+5, x-5:x+5] = 255
        zero_slide = cv2.resize(zero_slide, (self.state_shape[1], self.state_shape[0]))
        # plt.imshow(zero_slide)
        # plt.show()
        zero_slide = np.expand_dims(zero_slide, axis=-1)
        return zero_slide/255.0

    def combine_map_pose(self, map_mask, pose_mask):
        # plt.subplot(1, 2, 1)
        # plt.imshow(map_mask[:, :, 0])
        # plt.subplot(1, 2, 2)
        # plt.imshow(pose_mask[:, :, 0])
        # plt.show()
        if self.consider_last_pose:
            observe =  np.concatenate([map_mask, pose_mask, np.copy(self.last_pose_mask)], axis=-1)
        else:
            observe =  np.concatenate([map_mask, pose_mask], axis=-1)
        return observe

    def distance_reward(self, current_action, last_action):
        if last_action is None:
            last_x, last_y = self.robo_pose[0], self.robo_pose[1]
        else:
            last_x, last_y = self.action_to_point(last_action)

        current_x, current_y = self.action_to_point(current_action)
        dist = sqrt((current_x-last_x)**2 + (current_y-last_y)**2)
        print "distance reward: ", dist*5
        return dist * 5.0

    def is_explored(self, x, y):
        y = self.true_H - y

        region = self.last_map_image[y-10:y+10, x-10:x+10]
        explored_rate = float(np.where(region==0)[0].shape[0]) / 400.0
        # print 'explored_rate: ', explored_rate
        # plt.subplot(1, 2, 1)
        # plt.imshow(self.last_map_image)
        # plt.plot(x, y, 'o')
        # plt.subplot(1, 2, 2)
        # plt.imshow(region)
        # plt.show()
        if explored_rate > 0.025:
            return True
        else:
            return False

    def action_to_point(self, action):
        return self.goal_x_set_[action] * 20 + 10, self.goal_y_set_[action] * 20 + 10

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        beta = 50.0
        # self.wait_for_local_map()
        # 1. if is terminal action
        if action == 0:
            print 'select terminal action!'
            # waiting for map and compare with true map
            # static_map = self.waiting_for_static_map()
            explored_rate = self.compare_map(self.last_map_image)
            if explored_rate > 0.9:
                reward = 1 # 1000
            else:
                reward = -1 # -1000
            done = True
            if self.last_action is not None:
                last_x, last_y = self.action_to_point(self.last_action)
            else:
                last_x, last_y = self.robo_pose[0], self.robo_pose[1]
            pose_mask = self.generate_pose_mask(last_x, last_y)
            map_mask = self.map_to_observation(self.last_map_image)
            # TODO save local map
            observe = self.combine_map_pose(map_mask, pose_mask)
            self.last_pose_mask = pose_mask
            info = {"grid": 0.0, "path": 0.0}
            return observe, reward, done, info
        current_action = action - 1
        x, y = self.action_to_point(current_action)
        self.action_history[action] += 1
        print "select point: ", x, y
        # 2. if this action is equal to last action
        if self.action_history[action] > 1:
            reward = -1  # -200 * (self.action_history[action] - 1)
            done = False
            pose_mask = self.generate_pose_mask(x, y)
            map_mask = self.map_to_observation(self.last_map_image)
            # TODO save local map
            observe = self.combine_map_pose(map_mask, pose_mask)
            self.last_pose_mask = pose_mask
            info = {"grid":0.0, "path":0.0}
            return observe, reward, done, info
        # 3. if (x, y) is explored
        is_explored = self.is_explored(x, y)
        if not is_explored:
            reward = -1 # -100
            done = True
            if self.last_action is not None:
                last_x, last_y = self.action_to_point(self.last_action)
            else:
                last_x, last_y = self.robo_pose[0], self.robo_pose[1]
            pose_mask = self.generate_pose_mask(last_x, last_y)
            map_mask = self.map_to_observation(self.last_map_image)
            # TODO 
            observe = self.combine_map_pose(map_mask, pose_mask)
            self.last_pose_mask = pose_mask
            if self.test:
                info = {"grid": 0.0, "path": -1.0}
                done = False
            info = {"grid": 0.0, "path": 0.0}
            return observe, reward, done, info
        # publish goal and waiting for feed back
        map_index = MapIndex()
        map_index.x = int(y)
        map_index.y = int(x)
        self.goal_pub_.publish(map_index)
        # this is for c++ communication with python bug
        goal_published = False
        while not goal_published:
            try:
                map_idx_fb = rospy.wait_for_message('/map_index_callback', Bool, timeout=3)
                goal_published = True
                # print 'received callback'
            except:
                print 'cannot receive goal publish feedback, publish again'
                self.goal_pub_.publish(map_index)

        # waiting for planner feedback
        is_arrived = False
        feed_back = None
        path_length = 0
        try:
            path_length_msg = rospy.wait_for_message('/global_path_length', Float32, timeout=5)
            path_length = path_length_msg.data
            feed_back = rospy.wait_for_message('/global_feedback', Bool, timeout=120)
        except:
            print 'cannot receive plannner feedback'
            reward = -1 # -200
            done = True
            if self.last_action is not None:
                last_x, last_y = self.action_to_point(self.last_action)
            else:
                last_x, last_y = self.robo_pose[0], self.robo_pose[1]
            pose_mask = self.generate_pose_mask(last_x, last_y)
            map_mask = self.map_to_observation(self.last_map_image)
            observe = self.combine_map_pose(map_mask, pose_mask)
            self.last_pose_mask = pose_mask
            info = {"grid": 0.0, "path": 0.0}
            if self.test:
                info = {"grid": 0.0, "path": -1.0}
                done = False
            return observe, reward, done, info

        # whether the goal is arrived
        if feed_back.data:
            is_arrived = True
        else:
            is_arrived = False
            reward = -1 # -200
            done = True
            if self.last_action is not None:
                last_x, last_y = self.action_to_point(self.last_action)
            else:
                last_x, last_y = self.robo_pose[0], self.robo_pose[1]
            pose_mask = self.generate_pose_mask(last_x, last_y)
            map_mask = self.map_to_observation(self.last_map_image)
            # TODO 
            observe = self.combine_map_pose(map_mask, pose_mask)
            self.last_pose_mask = pose_mask
            info = {"grid": 0.0, "path": 0.0}
            if self.test:
                info = {"grid": 0.0, "path": -1.0}
                done = False
            return observe, reward, done, info
        # arrived the goal, calculate increased occupied grids
        static_map_image = self.waiting_for_static_map()
        current_occ_count = np.where(static_map_image == 0)[0].shape[0]
        last_occ_count = np.where(self.last_map_image == 0)[0].shape[0]
        expand_grid = current_occ_count - last_occ_count
        reward = (float(expand_grid) - path_length * beta) / 3000.0
        print "increase occupied grid: ", expand_grid, " path cost: ", path_length*beta
        info = {"grid":expand_grid, "path":path_length}
        done = False

        pose_mask = self.generate_pose_mask(x, y)
        map_mask = self.map_to_observation(static_map_image)
        observe = self.combine_map_pose(map_mask, pose_mask)

        self.last_map_image = static_map_image
        self.last_action = current_action
        self.last_pose_mask = pose_mask
        return observe, reward, done, info
        # calculate reward
        # entropy reduce // cannot arrived // path length

    def _reset(self):
        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/stage/reset_simulation')
        try:
            #reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/stage/reset_simulation service call failed")
        rest = Bool()
        rest.data = True
        self.reset_pub.publish(rest)
        try:
            reset_pose = rospy.wait_for_message('/reset_pose', PoseStamped, timeout=5)
            reset_x = reset_pose.pose.position.x
            reset_y = reset_pose.pose.position.y
            self.true_half_W = int(round(reset_x) * 20)
            self.true_half_H = self.true_H - int(round(reset_y) * 20)
            self.robo_pose = [int(round(reset_x) * 20), int(round(reset_y) * 20), 0]
            print "received reset pose: ", reset_x, reset_y
        except:
            print "cannot receive reset pose !!"
        map_data = None
        static_map = None
        while map_data is None:
            # map_data = self.waiting_for_static_map()
            try:
                map_data = rospy.wait_for_message('/map', OccupancyGrid, timeout=5)
                self.map_resolution = map_data.info.resolution
                self.map_origin = map_data.info.origin
                # print "receive map data"
            except:
                print "cannot receive map data!"

            if map_data is not None:
                static_map = self.convert_map_to_image(map_data)
                # print np.where(static_map == 0)[0].shape[0]
                if np.where(static_map == 0)[0].shape[0] > 2000:
                    map_data = None
        self.last_map_image = static_map
        self.last_action = None

        self.action_history = [0] * self.get_action_size()
        pose_mask = self.generate_pose_mask(self.robo_pose[0], self.robo_pose[1])
        map_mask = self.map_to_observation(static_map)
        self.last_pose_mask = np.zeros_like(pose_mask)
        # waiting for frontier
        if self.use_frontiers:
            frontier_makers = rospy.wait_for_message("/explore/frontiers", MarkerArray, timeout=5)
            self.maker_callback(frontier_makers)
        return self.combine_map_pose(map_mask, pose_mask)
