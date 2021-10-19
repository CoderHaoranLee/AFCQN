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

# from move_base_msgs import MoveBaseActionResult
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler


class StageCampCarLidarNNEnv(stage_env.StageEnv):
    def __init__(self):
        # Launch the simulation with the given launchfile name
        stage_env.StageEnv.__init__(self, "StageCampCarLidar_v0.launch") # TODO
        self.unpause = rospy.ServiceProxy('/stage/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/stage/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/stage/reset_simulation', Empty)
        self.reset_pub = rospy.Publisher('/slam_reset', Bool, queue_size=5)

        self.true_map = cv2.imread("/root/ros_workspaces/catkin_ws/src/RLStageROS/world/maps/simple_camp.png") # TODO
        self.true_map = np.asarray(self.true_map)
        self.true_H, self.true_W, _ = self.true_map.shape
        self.true_half_H, self.true_half_W = self.true_H/2, self.true_W/2

        self.last_occu_count = None
        self.last_map_image = None

        self.map_resolution = None
        self.map_origin = None
        self.reward_range = (-np.inf, np.inf)


        self.state_shape = [100, 100]
        self.action_size = 11

        self.last_action = None

        self.robo_pose = None
        self.last_pose_mask = None
        self.consider_last_pose = True
        self.test = False

        self.local_costmap_sub_ = rospy.Subscriber('/move_base/local_costmap/costmap', OccupancyGrid, self.convert_local_map_to_image)
        self.local_map_image_pub_ = rospy.Publisher('local_map/image_raw', Image, queue_size=2)
        self.bridge_ = CvBridge()
        self.last_local_map_image = None
        self.current_local_map_image = None

        self.crop_front = 20.
        self.crop_lr = 20.

        self.pose_listener_ = tf.TransformListener()

        # move bace action 
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to move base server")

        self._seed()


    def get_state_size(self):
        return [self.state_shape[0], self.state_shape[1], 3]

    def get_action_size(self):
        return self.action_size

    def convert_map_to_image(self, data):
        resolution = data.info.resolution
        H, W = data.info.height, data.info.width
        map_origin_x = data.info.origin.position.x
        map_origin_y = data.info.origin.position.y
        map_pixel_x = -int(map_origin_x / resolution)
        map_pixel_y = H + int(map_origin_y / resolution)
        print map_pixel_x, map_pixel_y

        map_data = np.asarray(data.data)
        idx_equ0 = map_data == 0
        idx_less0 = map_data < 0
        idx_greater0 = map_data > 0
        map_data[idx_less0] = 128.0
        map_data[idx_greater0] = 255.0

        map_data = map_data.reshape(H, W)
        map_data = np.flipud(map_data)
        map_data = np.asarray(map_data, dtype=np.float)

        # copy_map_data = np.ones((self.true_H, self.true_W)) * 128.0

        # copy_map_data[max(0, self.true_half_H - map_pixel_y):min(self.true_H, self.true_half_H + (H - map_pixel_y)),
        # max(0, self.true_half_W - map_pixel_x):min(self.true_W, self.true_half_W + (W - map_pixel_x))] = map_data[max(0,map_pixel_y - self.true_half_H):min(H, map_pixel_y + (self.true_H - self.true_half_H)),
        #                                                                                                  max(0, map_pixel_x - self.true_half_W):min(W, map_pixel_x + (self.true_W - self.true_half_W))]
        return map_data

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
            yaw = yaw + 270
            center = (W/2, H/2)
            M = cv2.getRotationMatrix2D(center, yaw, 1)     
            map_data = cv2.warpAffine(map_data, M, (W, H))
            local_map_image = cv2.resize(map_data[:self.state_shape[0], :] / 255., (self.state_shape[1], self.state_shape[0]))
            self.current_local_map_image = np.expand_dims(local_map_image, axis=-1)
            msg = self.bridge_.cv2_to_imgmsg(map_data, '8UC1')
            self.local_map_image_pub_.publish(msg)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "cannot find the transformation between map and base_link"
        

    def waiting_for_static_map(self):
        # print 'waiting for static map ... '
        try:
            last_map = rospy.wait_for_message('/map', OccupancyGrid, timeout=5)
        except:
            print 'cannot received map'
            last_map = None
        count_static = 0
        static_map = None
        local_observe = None
        map_h = None
        map_w = None

        current_pixel_x = None
        current_pixel_y = None
        while count_static < 5:
            rospy.loginfo("waiting for static map")
            try:
                current_map = rospy.wait_for_message('/map', OccupancyGrid, timeout=5)
                self.map_resolution = current_map.info.resolution
                self.map_origin = current_map.info.origin
                map_h = current_map.info.height
                map_w = current_map.info.width
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
            last_map = current_map
        # lookup base_link pose
        x, y, yaw = 0, 0, 0
        try:
            (trans,rot) = self.pose_listener_.lookupTransform('/map', '/base_link', rospy.Time(0))
            # print trans
            (roll, pitch, yaw) = euler_from_quaternion (rot)
            x, y = trans[0], trans[1]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "waiting map: cannot find the transformation between map and base_link"
            return static_map, None
        yaw = -yaw
        x1 = 0. * cos(yaw) - (self.crop_lr/2.) * sin(yaw)
        y1 = 0. * sin(yaw) + (self.crop_lr/2.) * cos(yaw)
        x2 = self.crop_front * cos(yaw) - (self.crop_lr/2.) * sin(yaw)
        y2 = self.crop_front * sin(yaw) + (self.crop_lr/2.) * cos(yaw)
        x3 = self.crop_front * cos(yaw) + (self.crop_lr/2.) * sin(yaw)
        y3 = self.crop_front * sin(yaw) - (self.crop_lr/2.) * cos(yaw)
        x4 = 0. * cos(yaw) + (self.crop_lr/2.) * sin(yaw)
        y4 = 0. * sin(yaw) - (self.crop_lr/2.) * cos(yaw)

        x1 = (x + x1 - self.map_origin.position.x) / self.map_resolution
        y1 = (y + y1 - self.map_origin.position.y) / self.map_resolution
        x2 = (x + x2 - self.map_origin.position.x) / self.map_resolution
        y2 = (y + y2 - self.map_origin.position.y) / self.map_resolution
        x3 = (x + x3 - self.map_origin.position.x) / self.map_resolution
        y3 = (y + y3 - self.map_origin.position.y) / self.map_resolution
        x4 = (x + x4 - self.map_origin.position.x) / self.map_resolution
        y4 = (y + y4 - self.map_origin.position.y) / self.map_resolution

        corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int)
        compensate_map, compensation = self.compensate_map(static_map, corners)

        corners = corners+compensation
        rect = cv2.minAreaRect(corners)
        box = cv2.boxPoints(rect)

        # box = np.int0(corners)
        # # print("bounding box: {}, yaw:{}".format(box,yaw))
        # cv2.drawContours(compensate_map, [box], 0, (0,0,200), 5)

        crop_width = int(self.crop_front / self.map_resolution)
        crop_height = int(self.crop_lr / self.map_resolution)
        src_pts = corners.astype("float32")
        dst_pts = np.array([[0, 0],
                        [crop_width-1, 0],
                        [crop_width-1, crop_height-1],
                        [0, crop_height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        local_observe = cv2.warpPerspective(compensate_map, M, (crop_width, crop_height))
        local_observe = np.flipud(local_observe)
        # plt.subplot(1, 2, 1)
        # plt.imshow(compensate_map)
        # plt.subplot(1, 2, 2)
        # plt.imshow(local_observe)
        # plt.show()
        # corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int)
        # compensate_map = self.compensate_map(static_map, corners)
        # current_pixel_x = (x - self.map_origin.position.x) / self.map_resolution
        # current_pixel_y = (y - self.map_origin.position.y) / self.map_resolution
        # new_yaw = yaw
        # if new_yaw < -pi/2:
        #     new_yaw += pi
        # if new_yaw > pi/2:
        #     new_yaw -= pi
        # if new_yaw>=0 and new_yaw<=pi/2:
        #     new_h = map_h * cos(new_yaw) + map_w * sin(new_yaw)
        #     new_w = map_h * sin(new_yaw) + map_w * cos(new_yaw)
        # elif new_yaw>=-pi/2 and new_yaw<=0:
        #     new_h = map_h * sin(new_yaw) + map_w * cos(new_yaw)
        #     new_w = map_h * cos(new_yaw) + map_w * sin(new_yaw)
        # else:
        #     rospy.loginfo("bad new_yaw:{}".format(new_yaw))
        #     new_h = map_h
        #     new_w = map_w
        # if yaw < 0:
        #         yaw += 2 * pi
        # center = (int(current_pixel_x), int(current_pixel_y))
        # yaw = yaw * 180 / pi
        # M = cv2.getRotationMatrix2D(center, yaw, 1)     
        # map_data = cv2.warpAffine(static_map, M, (int(new_w), int(new_h)), borderValue=128)
        
        # crop_x = self.crop_front / self.map_resolution
        # crop_y = self.crop_lr / self.map_resolution
        # local_observe = self.crop_local_observe_from_map(map_data, current_pixel_x, current_pixel_y, crop_x, crop_y)
        return static_map, self.map_to_observation(local_observe)

    def map_to_observation(self, map_data):
        # print map_data.shape, self.state_shape
        map_image = cv2.resize(map_data, (self.state_shape[1], self.state_shape[0]))
        map_image = np.expand_dims(map_image, axis=-1)
        return map_image / 255.0
    
    def crop_local_observe_from_map(self, map_data, current_pixel_x, current_pixel_y, crop_x, crop_y):
        '''
        x is front , y is left
        '''
        local_observe = np.zeros((int(crop_y), int(crop_x))) + 128.
        H, W = map_data.shape
        current_pixel_x = int(current_pixel_x)
        current_pixel_y = int(current_pixel_y)
        crop_x = int(crop_x)
        crop_y = int(crop_y)
        left = min(W, current_pixel_x+crop_x)
        up = max(0, current_pixel_y - crop_y/2)
        down = min(H, current_pixel_y + crop_y/2)
        new_up = crop_y / 2. - (current_pixel_y - up)
        new_down = crop_y / 2. + (down - current_pixel_y)
        print("current_pixel_x", current_pixel_x, "current_pixel_y",current_pixel_y)
        print("crop_x",crop_x, "crop_y",crop_x)
        if current_pixel_x >=W or current_pixel_y >=H:
            return None
        if new_up<0 or new_down<0:
            return None
        if (int(new_down)-int(new_up)) != (int(down)-int(up)) or (int(left)-int(current_pixel_x)) != (int(left)-int(current_pixel_x)):
            print("bad local map")
            return None
        print("H,", H, "W", W, "left", left, "up", up, "down", down, "new_up", new_up, "new_down", new_down)
        local_observe[int(new_up):int(new_down), :(int(left)-int(current_pixel_x))] = map_data[int(up):int(down), int(current_pixel_x):int(left)]
        plt.imshow(local_observe)
        plt.show()
        local_observe = cv2.resize(local_observe, (self.state_shape[1], self.state_shape[0]))
        local_observe = np.expand_dims(local_observe, axis=-1) /255.
        return local_observe
    
    def compensate_map(self, map_data, corners):
        '''
        corners: np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        '''
        H, W = map_data.shape
        min_corner = np.min(corners, axis=0)
        left_corner, up_corner = min_corner[0], min_corner[1]
        max_corner = np.max(corners, axis=0)
        right_corner, down_corner = max_corner[0], max_corner[1]
        if left_corner<0:
            left_comp = - left_corner
        else:
            left_comp = 0
        if up_corner<0:
            up_comp = - up_corner
        else:
            up_comp = 0
        if right_corner>W:
            right_comp = right_corner - W
        else:
            right_comp = 0
        if down_corner>H:
            down_comp = down_corner - H
        else:
            down_comp = 0
        new_H = H + up_comp + down_comp
        new_W = W + left_comp + right_comp
        new_map = np.zeros((new_H, new_W)) + 128. 
        new_map[up_comp:up_comp+H, left_comp:left_comp+W] = map_data
        compensation = np.array([[left_comp, up_comp],[left_comp, up_comp],[left_comp, up_comp],[left_comp, up_comp]])
        return new_map, compensation

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

    def combine_map_pose(self, local_map, map_mask, pose_mask):
        if local_map is None:
            local_map = np.zeros_like(map_mask) + 255
            rospy.loginfo("local map is None, bad data")
        observe = np.concatenate([local_map, map_mask, pose_mask], axis=-1)
        # if self.consider_last_pose:
        #     observe =  np.concatenate([map_mask, pose_mask, np.copy(self.last_pose_mask)], axis=-1)
        # else:
        #     observe =  np.concatenate([map_mask, pose_mask], axis=-1)
        return observe

    def action_to_point(self, action):
        '''
        return x, y, yaw
        '''
        # lookup current position
        x, y, yaw = 0, 0, 0
        r, c = None, None
        try:
            (trans,rot) = self.pose_listener_.lookupTransform('/map', '/base_link', rospy.Time(0))
            # print trans
            (roll, pitch, yaw) = euler_from_quaternion (rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "Action to point: cannot find the transformation between map and base_link"
            return None, None, None, False
        if action == 1:
            # left backward
            delta_x, delta_y, delta_theta = 0., 2., pi
            r, c = 21, 0
        elif action == 2:
            # turn left
            delta_x, delta_y, delta_theta = 3., 2., pi/2
            r, c = 10, 0
        elif action == 3:
            # turn left upward
            delta_x, delta_y, delta_theta = 4., 2., pi/2
            r, c = 0, 0
        elif action == 4:
            # left forward
            delta_x, delta_y, delta_theta = 5., 2., pi/4.
            r, c = 0, 11
        elif action == 5:
            # forward
            delta_x, delta_y, delta_theta = 5, 1., 0.
            r, c = 0, 23
        elif action == 6:
            # right forward
            delta_x, delta_y, delta_theta = 5., 0., 0.
            r, c = 0, 33
        elif action == 7:
            # turn right upward
            delta_x, delta_y, delta_theta = 5, -1, 0.
            r, c = 0, 54
        elif action == 8:
            # turn right
            delta_x, delta_y, delta_theta = 5., -2., -pi/4.
            r, c = 10, 54
        elif action == 9:
            delta_x, delta_y, delta_theta = 4., -2., -pi/2.
            r, c = 10, 54
        elif action == 10:
            delta_x, delta_y, delta_theta = 3., -2, -pi/2.
            r, c = 10, 54
        else:
            print "Invalid Action !", action
            return None, None, None, False
        x = trans[0] + delta_x * cos(yaw) - delta_y * sin(yaw)
        y = trans[1] + delta_x * sin(yaw) + delta_y * cos(yaw)
        yaw = yaw + delta_theta
        if self.current_local_map_image is not None:
            valid = self.current_local_map_image[r, c] < 120
        else:
            valid = False
        return x, y, yaw, True
    
    def action_to_path_length(self, action):
        path_length = 100
        if action == 1:
            path_length = 5
        elif action == 2:
            path_length = 4
        elif action == 3:
            path_length = 3
        elif action == 4:
            path_length = 2
        elif action == 5:
            path_length = 1
        elif action == 6:
            path_length = 2
        elif action == 7:
            path_length = 3
        elif action == 8:
            path_length = 4
        elif action == 9:
            path_length = 5
        else:
            path_length = 100
        return path_length

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
            explored_rate = self.compare_map(self.last_map_image)
            if explored_rate > 0.9:
                reward = 1 # 1000
            else:
                reward = -1 # -1000
            done = True
            if self.last_action is not None:
                last_x, last_y, _, _ = self.action_to_point(self.last_action)
            else:
                last_x, last_y = self.robo_pose[0], self.robo_pose[1]
            pose_mask = self.generate_pose_mask(last_x, last_y)
            map_mask = self.map_to_observation(self.last_map_image)
            # TODO save local map
            local_map_mask = self.current_local_map_image
            observe = self.combine_map_pose(local_map_mask, map_mask, pose_mask)
            self.last_pose_mask = pose_mask
            info = {"grid": 0.0, "path": 0.0}
            self.last_occu_count = 0
            return observe, reward, done, info
        
        current_action = action
        x, y, yaw, valid = self.action_to_point(current_action)
        print "select point: ", x, y, yaw, valid
        if not valid:
            reward = -1
            done = True
            pose_mask = self.generate_pose_mask(self.robo_pose[0], self.robo_pose[0])
            map_mask = self.map_to_observation(self.last_map_image)
            observe = self.combine_map_pose(self.current_local_map_image, map_mask, pose_mask)
            info = {"grid": 0.0, "path": 0.0}
            self.last_occu_count = 0
            return observe, reward, done, info
        else:
            # publish goal
            goal = MoveBaseGoal()
            goal.target_pose.header.frame_id = 'map'
            goal.target_pose.header.stamp = rospy.Time.now()
            q_angle = quaternion_from_euler(0, 0, yaw, axes='sxyz')
            goal.target_pose.pose = Pose(Point(x, y, 0.0), Quaternion(*q_angle))
            self.move_base.send_goal(goal)
            finished_within_time = self.move_base.wait_for_result(rospy.Duration(10)) # 10s
            path_length = self.action_to_path_length(action)
            if not finished_within_time:
                self.move_base.cancel_goal()
                rospy.loginfo("Timed out achieving goal")
                reward = -1 # -200
                done = True
                if self.last_action is not None:
                    last_x, last_y, _, _ = self.action_to_point(self.last_action)
                else:
                    last_x, last_y = self.robo_pose[0], self.robo_pose[1]
                pose_mask = self.generate_pose_mask(last_x, last_y)
                map_mask = self.map_to_observation(self.last_map_image)
                local_map_mask = self.current_local_map_image
                observe = self.combine_map_pose(local_map_mask, map_mask, pose_mask)
                self.last_pose_mask = pose_mask
                info = {"grid": 0.0, "path": 0.0}
                self.last_occu_count = 0
                if self.test:
                    info = {"grid": 0.0, "path": -1.0}
                    done = False
                return observe, reward, done, info
            else:
                # We made it!
                state = self.move_base.get_state()
                if state == GoalStatus.SUCCEEDED:
                    rospy.loginfo("Goal succeeded!")

                # arrived the goal, calculate increased occupied grids
                static_map_image, local_map_image = self.waiting_for_static_map()
                current_occ_count = np.where(static_map_image == 0)[0].shape[0]
                # last_occ_count = np.where(self.last_map_image == 0)[0].shape[0]
                expand_grid = current_occ_count - self.last_occu_count
                if expand_grid < 0:
                    expand_grid = 0
                reward = (float(expand_grid) - path_length * beta) / 3000.0
                print "increase occupied grid: ", expand_grid, " path cost: ", path_length*beta
                info = {"grid":expand_grid, "path":path_length}
                done = False

                self.current_local_map_image = local_map_image
                pose_mask = self.generate_pose_mask(x, y)
                map_mask = self.map_to_observation(static_map_image)
                observe = self.combine_map_pose(self.current_local_map_image, map_mask, pose_mask)

                self.last_map_image = static_map_image
                self.last_occu_count = current_occ_count
                self.last_action = current_action
                self.last_pose_mask = pose_mask
                self.robo_pose = [x, y, yaw]
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
        time.sleep(2) # waiting for map update
        map_data = None
        static_map = None
        map_count = 0
        local_observe = None
        map_h = None
        map_w = None
        current_pixel_x = None
        current_pixel_y = None
        # waiting 1
        x, y, yaw = 0, 0, 0
        while map_count < 5:
            # map_data = self.waiting_for_static_map()
            rospy.loginfo("Work hard for map data ......")
            
            try:
                map_data = rospy.wait_for_message('/map', OccupancyGrid, timeout=5)
                self.map_resolution = map_data.info.resolution
                self.map_origin = map_data.info.origin
                map_h = map_data.info.height
                map_w = map_data.info.width
                try:
                    (trans,rot) = self.pose_listener_.lookupTransform('/map', '/base_link', rospy.Time(0))
                    # print trans
                    x, y = trans[0], trans[1]
                    (roll, pitch, yaw) = euler_from_quaternion (rot)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print "waiting map: cannot find the transformation between map and base_link"
                
                # print "receive map data"
                map_count += 1
            except:
                print "cannot receive map data!"

            if map_data is not None:
                static_map = self.convert_map_to_image(map_data)
                # print np.where(static_map == 0)[0].shape[0]
                # if np.where(static_map == 0)[0].shape[0] > 2000:
                #     map_data = None
                #     rospy.loginfo("free space is too large...")
        x1 = 0. * cos(yaw) - (self.crop_lr/2.) * sin(yaw)
        y1 = 0. * sin(yaw) + (self.crop_lr/2.) * cos(yaw)
        x2 = self.crop_front * cos(yaw) - (self.crop_lr/2.) * sin(yaw)
        y2 = self.crop_front * sin(yaw) + (self.crop_lr/2.) * cos(yaw)
        x3 = self.crop_front * cos(yaw) + (self.crop_lr/2.) * sin(yaw)
        y3 = self.crop_front * sin(yaw) - (self.crop_lr/2.) * cos(yaw)
        x4 = 0. * cos(yaw) + (self.crop_lr/2.) * sin(yaw)
        y4 = 0. * sin(yaw) - (self.crop_lr/2.) * cos(yaw)

        x1 = (x + x1 - self.map_origin.position.x) / self.map_resolution
        y1 = (y + y1 - self.map_origin.position.y) / self.map_resolution
        x2 = (x + x2 - self.map_origin.position.x) / self.map_resolution
        y2 = (y + y2 - self.map_origin.position.y) / self.map_resolution
        x3 = (x + x3 - self.map_origin.position.x) / self.map_resolution
        y3 = (y + y3 - self.map_origin.position.y) / self.map_resolution
        x4 = (x + x4 - self.map_origin.position.x) / self.map_resolution
        y4 = (y + y4 - self.map_origin.position.y) / self.map_resolution

        corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.int)
        compensate_map, compensation = self.compensate_map(static_map, corners)

        corners = corners+compensation
        rect = cv2.minAreaRect(corners)
        box = cv2.boxPoints(rect)
        crop_width = int(self.crop_front / self.map_resolution)
        crop_height = int(self.crop_lr / self.map_resolution)
        src_pts = corners.astype("float32")
        dst_pts = np.array([[0, 0],
                        [crop_width-1, 0],
                        [crop_width-1, crop_height-1],
                        [0, crop_height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        local_observe = cv2.warpPerspective(compensate_map, M, (crop_width, crop_height))
        local_observe = np.flipud(local_observe)
        # plt.imshow(local_observe)
        # plt.show()
        
        # local_observe = self.crop_local_observe_from_map(static_map, current_pixel_x, map_h/2, crop_x, crop_y)
        
        self.last_map_image = static_map
        self.last_occu_count = 0
        self.last_action = None

        # waiting for local costmap
        # try:
        #     current_local_map = rospy.wait_for_message('/move_base/local_costmap/costmap', OccupancyGrid, timeout=5)
        #     self.convert_local_map_to_image(current_local_map)
        # except:
        #     print '_step: cannot received local costmap'
        #     self.current_local_map_image = None
        self.current_local_map_image = self.map_to_observation(local_observe)

        pose_mask = self.generate_pose_mask(self.robo_pose[0], self.robo_pose[1])
        map_mask = self.map_to_observation(static_map)
        self.last_pose_mask = np.zeros_like(pose_mask)
        
        return self.combine_map_pose(self.current_local_map_image, map_mask, pose_mask)
