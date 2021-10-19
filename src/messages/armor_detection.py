#! /usr/bin/env python
import rospy
import roslib
import sys
from sensor_msgs.msg import Image
from messages.msg import EnemyPos
from messages.msg import ArmorsPos
from messages.msg import RoboState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import time
import cnn_model
import tensorflow as tflow
import cv2
import numpy as np
import math
import copy
import tf
from tf_tools import *

cnn_label = 0
test_label = 1
side_label = 1
debug_label = 0
debug_light = 0
debug_label_armor = 0
armor_height = 60.0
armor_width = 123.0
army_color = "B" # "R"
detection_None = 0

class Light:
    def __int__(self):
        self.point = None
        self.middle_line = None
        self.left_line = None
        self.right_line = None
        self.light_color = None

    def light_judge(self,point_box = []):
        if len(point_box) == 0:
            point_box = self.point
        else:
            self.point = point_box
        y_order = sorted(point_box, key=lambda x:x[-1])
        x_order = sorted(point_box, key=lambda x:x[0])
        middle_point_1 = np.int0(0.5*(np.array(y_order[0]) + np.array(y_order[1])))
        middle_point_2 = np.int0(0.5*(np.array(x_order[0]) + np.array(x_order[1])))
        middle_point_3 = np.int0(0.5*(np.array(y_order[2]) + np.array(y_order[3])))
        middle_point_4 = np.int0(0.5 *(np.array(x_order[2]) + np.array(x_order[3])))
        self.x_length = middle_point_4[0] - middle_point_2[0]
        self.y_length = middle_point_3[1] - middle_point_1[1]
        edge_length = []
        edge_vector = []
        for i in range(3):
            # if i ==3:
            edge_vector_i = np.array(point_box[i])-np.array(point_box[i+1])
            length = np.linalg.norm(edge_vector_i, ord=2)
            edge_vector.append(edge_vector_i)

            edge_length.append(length)

        min_edge = min(edge_length)
        max_edge = max(edge_length)
        if max_edge > 200  and min_edge > 5:
            return False
        if min_edge == 0:
            return False
        if max_edge/min_edge < 1.2:
            # if  debug_light == 1:
            #     print "light:",max_edge,min_edge
            return False
        edge_vector = []
        edge_vector.append(np.array(point_box[3])-np.array(point_box[0]))
        edge_length.append(np.linalg.norm(np.array(point_box[3])-np.array(point_box[0]), ord=2))
        if self.x_length == 0:
            return False

        self.middle_line = [np.array(middle_point_1), np.array(middle_point_3)]
        self.middle_length = np.linalg.norm(np.array(middle_point_1)-np.array(middle_point_3), ord=2)
        # self.middle_line = [np.array(x_order[]), np.array(middle_point_3)]
        return 1

class Armor:
    def __init__(self):
        self.points = None
        self.edge_ratio = 3.5
        self.min_edge = 10
        self.max_angle = 137
        self.min_angle = 55
        self.min_edge_ratio = 0.33
        self.max_edge_ratio = 3
        self.middle_line_angle = 30
        self.middle_line_ratio = 3
        self.second_edge_ratio = 2.5

        self.line_area = None

    def armor_judge(self, light_pair):
        light_1 =light_pair[0]
        light_2 = light_pair[1]
        point_box = light_1.middle_line + light_2.middle_line[::-1]
        if len(point_box) == 0:
            point_box = self.points
        else:
            self.points = point_box
        y_order = sorted(point_box, key=lambda x: x[-1])
        x_order = sorted(point_box, key=lambda x: x[0])
        middle_point_1 = np.int0(0.5*(np.array(y_order[0]) + np.array(y_order[1])))
        middle_point_2 = np.int0(0.5*(np.array(x_order[0]) + np.array(x_order[1])))

        middle_point_3 = np.int0(0.5*(np.array(y_order[2]) + np.array(y_order[3])))
        middle_point_4 = np.int0(0.5 *(np.array(x_order[2]) + np.array(x_order[3])))
        self.edge_length = []
        edge_angle = []
        edge_vector = []
        for i in range(3):
            # if i ==3:
            edge_vector_i = point_box[i]-point_box[i+1]
            length = np.linalg.norm(edge_vector_i, ord=2)
            edge_vector.append(edge_vector_i)

            self.edge_length.append(length)
        edge_vector.append(point_box[3]-point_box[0])
        self.edge_length.append(np.linalg.norm(point_box[3]-point_box[0], ord=2))
        self.edge_length.sort()
        min_edge = self.edge_length[0]
        max_edge = self.edge_length[3]
        if min_edge != self.min_edge:
            edge_ratio = max_edge/min_edge
            if edge_ratio > self.edge_ratio:
                # if debug_label == 1:
                #     print "1.edge_ratio:", edge_ratio
                return 0
        else:
            # if debug_label == 1:
            #     print "2.min_edge = 0"
            return 0
        second_edge_ratio = self.edge_length[2]/self.edge_length[1]
        if second_edge_ratio >3:
            # if debug_label == 1:
            #     print "9.second_edge_ratio:", second_edge_ratio

            return 0

        edge_angle = []
        for i in range(3):
            angle = np.arccos(np.dot(edge_vector[i],
                              (edge_vector[i+1])/(self.edge_length[i]*self.edge_length[i+1])))
            angle =(angle*180)/math.pi
            if angle > self.max_angle or angle < self.min_angle:
                # if debug_label == 1:
                #     print "3.angle:", angle
                return 0

        for i in range(2):
            edge_ratio_2 = self.edge_length[i+2]/self.edge_length[i]
            if edge_ratio_2 > self.max_edge_ratio or edge_vector < self.min_edge_ratio :
                # if debug_label == 1:
                #     print "4.edge_ratio_2:", edge_ratio_2
                return 0

        x_raw_length = middle_point_4[0] - middle_point_2[0]

        x_length = middle_point_4[0] - middle_point_2[0]-0.5*light_1.x_length - 0.5*light_2.x_length
        y_length = middle_point_3[1] - middle_point_1[1]

        length = [x_length,y_length]
        self.min_lenth = min(length)
        self.max_length = max(length)
        left_middle_point = [point_box[0][0],int(0.5*(point_box[0][1] + point_box[1][1]))]
        right_middle_point = [point_box[2][0],int(0.5*(point_box[3][1] + point_box[2][1]))]
        if self.min_lenth !=0:

            ratio = float(self.max_length)/float(self.min_lenth)
        else:

            return 0
        if right_middle_point[0]-left_middle_point[0]!=0:

            line_angle = math.atan(abs(float(right_middle_point[1]-left_middle_point[1]))
                          / float(right_middle_point[0]-left_middle_point[0]))
        else:
            return 0
        line_angle = abs(line_angle*180)/math.pi
        # print ratio,line_angle
        if ratio < self.middle_line_ratio:
            if line_angle > self.middle_line_angle:
                # if debug_label == 1:
                #     print "6.middle_line_angle:", line_angle
                return 0
        else:
            #
            # t "5.middle_line_ratio:", ratio
            return 0
        if light_1.point[1][0]> light_2.point[1][0]:
            self.left_light = light_2.middle_line
            self.right_light = light_1.middle_line
        else:
            self.left_light = light_1.middle_line
            self.right_light = light_2.middle_line
        if self.left_light[0][1] >= self.right_light[0][1]:
            under_point_1 = self.left_light[0][0]
            under_point_2 = self.right_light[1][1]
            under_point_3 = self.left_light[0][0]+x_raw_length
            under_point_4 =min(self.right_light[1][1]+x_raw_length,640)
        else:
            under_point_1 = self.left_light[0][0]
            under_point_2 = self.left_light[1][1]
            under_point_3 = self.left_light[0][0] + x_raw_length
            under_point_4 = min(self.left_light[1][1] + x_raw_length, 640)
        if test_label ==1:
            if self.left_light[0][1] >= self.right_light[0][1]:
                under_point_1 = self.left_light[0][0]
                under_point_2 = self.right_light[0][1]
                under_point_3 = self.left_light[0][0] + x_raw_length
                under_point_4 = int(min(self.right_light[0][1] + 1.5*x_raw_length, 640))
            else:
                under_point_1 = self.left_light[0][0]
                under_point_2 = self.left_light[0][1]
                under_point_3 = self.left_light[0][0] + x_raw_length
                under_point_4 = int(min(self.left_light[0][1] + 1.5*x_raw_length, 640))

        self.line_area = [under_point_1, under_point_2, under_point_3, under_point_4]
        self.points =copy.deepcopy( light_1.middle_line + light_2.middle_line[::-1])
        return 1

class Distance:
    def __init__(self):
        # ???
        self.armor_points = np.array([[[-armor_width/2.0],[armor_height/2.0], [0.0]],
                                      [[armor_width/2.0], [armor_height/2.0], [0.0]],
                                      [[armor_width/2.0], [-armor_height/2.0], [0.0]],
                                      [[-armor_width/2.0], [-armor_height/2.0], [0.0]]])
        self.image_points = np.random.random((4,2,1))
        #
        self.camera_matrix = np.array([[520.0334010310696, 0.0, 304.8460838356355],
                                 [0.0, 522.1145816053537, 268.2227873795626],
                                 [0.0, 0.0, 1.0]])
        self.camera_distortion =np.array([[0.05985687918256749],[-0.0825295157658502],[0.004130642265799225],
        [0.001066639503312635],[0]])

        # self.pub_tf = tf.TransformBroadcaster()


    def get_distance(self, image_points):
        imagePoints = np.array(image_points, dtype=np.float64).reshape((4, 2))
        # print np.shape(imagePoints),imagePoints
        # print np.shape(self.armor_points),np.shape(np.array(image_points)),np.shape(self.camera_matrix)
        # print np.shape(self.camera_distortion)
        # sort points
        points = (imagePoints.tolist())
        points = sorted(points, key=lambda a_entry:a_entry[0])
        points = np.asarray(points)
        # right_down = None
        # right_up = None
        # left_down = None
        # left_up = None
        if points[0][1]>points[1][1]:
            right_down = points[1]
            right_up = points[0]
        else:
            right_down = points[0]
            right_up = points[1]
        if points[2][1]>points[3][1]:
            left_down = points[3]
            left_up = points[2]
        else:
            left_down = points[2]
            left_up = points[3]
        imagePoints = np.array([left_up, right_up, right_down, left_down]).reshape((4,2,1))
        _, rvec, tvec = cv2.solvePnP(self.armor_points, imagePoints, self.camera_matrix, self.camera_distortion)
        # print tvec
        fly_time = tvec[1]/float(1000*20)
        gravity_offset = 0.5*9.8*fly_time*1000
        xyz = [tvec[0],tvec[1]-gravity_offset+33,tvec[2]]
        pitch = math.atan(-xyz[1]/xyz[2])
        yaw = math.atan2(xyz[0], xyz[2])
        # pitch = pitch*180/np.pi
        # yaw = yaw*180/np.pi
        dis = (xyz[0]**2+xyz[2]**2)**0.5

        tvec = tvec / 1000.0

        center_x = 0.5*(left_up[0] + right_down[0])
        center_y = 0.5*(left_up[1] + right_down[1])
        delta_y = center_y - 240
        delta_x = 320 - center_x
        pitch = 5e-3 * delta_y
        yaw = 5e-3 * delta_x
        # print pitch, yaw, dis
        # self.pub_tf.sendTransform((tvec[0], tvec[1], tvec[2]), tf.transformations.quaternion_from_euler(tvec[0], tvec[1], tvec[2]),
        #                           rospy.Time.now(),
        #                           "armor",
        #                           "tool0")
        return rvec, tvec, pitch, yaw, dis[0]
        # return pitch, yaw, dis[0]


class gimbal_tracking(object):
    def __init__(self):
        self.current_pitch = 0.0
        self.current_yaw = 0.0
        self.gim_cmd_pub_ = rospy.Publisher('enemy_pos', EnemyPos, queue_size=1)
        self.sub_ = rospy.Subscriber('robo_state', RoboState, self.update_angle)

    def update_angle(self, data):
        self.current_pitch = data.gimbal_pitch
        self.current_yaw = data.gimbal_yaw

    def send_cmd(self, pitch, yaw):
        cmd_pitch = pitch
        cmd_yaw = yaw
        cmd = EnemyPos()
        cmd.enemy_pitch = cmd_pitch
        cmd.enemy_yaw = cmd_yaw
        cmd.enemy_dist = 0.0
        self.gim_cmd_pub_.publish(cmd)


class armor_detection(object):
    def __init__(self):
        # self.weights = cnn_model.weights
        # self.biases = cnn_model.biases
        # self.x = tflow.placeholder(tflow.float32, [None, 30, 30, 3])

        # self.keep_prob = tflow.placeholder(tflow.float32)
        # self.pred = cnn_model.alex_net(self.x, self.weights, self.biases, self.keep_prob)
        # self.init = tflow.initialize_all_variables()
        # self.saver = tflow.train.Saver()
        # self.sess = tflow.Session()
        # self.saver.restore(self.sess, "/home/drl/ros_codes/catkin_ws/src/messages/Model2/model.ckpt")
        # self.graph = tflow.get_default_graph()
        # self.prediction = self.graph.get_tensor_by_name('add_2:0')
        # self.y_result = tflow.nn.softmax(self.prediction)

        self.i = 0
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera_0", Image, self.callback)
        self.pub_gim = rospy.Publisher('enemy_pos', EnemyPos, queue_size=1)
        self.pub_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.pub_img = rospy.Publisher('detection', Image, queue_size=1)
        self.debug = 1
        self.army_color = 'B'

        self.distance = Distance()
        self.pub_tf = tf.TransformBroadcaster()

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_a = 0.0

        self.tf_listener = tf.TransformListener()
        self.armors_pub = rospy.Publisher('armors_pos', ArmorsPos, queue_size=1)

        self.listened = []

        self.gimab_cmd = gimbal_tracking()

        self.cmd_pitch = 0.0
        self.cmd_yaw = 0.0

    def get_light(self, img):

        if self.army_color == "B":
            lower_blue = np.array([0, 0, 200])
            upper_blue = np.array([190, 70, 255])
        else:
            lower_blue = np.array([20, 0, 200])
            upper_blue = np.array([190, 70, 255])

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # get mask
        start = time.time()*1000
        mask = cv2.inRange(hsv, lower_blue, upper_blue, )
        end = time.time()*1000
        print "inRange time:", float(end-start)/1000.0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        end2 = time.time()*1000
        print "morphologyEx time:", float(end2-end)/1000.0
        return mask

    def get_light_2(self,img):
        b, g, r = cv2.split(img)
        thresh = 0
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray_binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        # cv2.imshow("gray", gray_binary)
        if self.army_color == "R":
            thresh = 50
            result_img = cv2.subtract(r, g)
        else:
            result_img = cv2.subtract(b, g)
            thresh = 90

        ret, result_img = cv2.threshold(result_img, thresh, 255, cv2.THRESH_BINARY)
        index_range = np.argwhere(result_img == 255)
        # print np.shape(index_range)
        try:
            y_down = index_range[:, 0].max()
            y_up = index_range[:, 0].min()
            x_right = index_range[:, 1].max()
            x_left = index_range[:, 1].min()
        except Exception as e:
            print e
            return False

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result_img = cv2.dilate(result_img, kernel)
        cc = result_img & gray_binary

        return result_img, [x_left, y_up, x_right, y_down]


    def get_light_3(self, img):
        start = time.time()*1000
        result_light = self.get_light_2(img)
        end = time.time()*1000
        t = float(end-start) / 1000.0
        print "get_light_2:", t
        if result_light:
            result_2 = result_light[0]
            box_point = result_light[1]
            # print box_point
            start = time.time()*1000
            result_1 = self.get_light(img)
            end = time.time()*1000
            t = float(end-start) / 1000.0
            print "get_light: ", t
            print box_point
            result_3 = result_1[box_point[1]:box_point[3], box_point[0]:box_point[2]]

            return result_3, box_point[0], box_point[1]
        else:
            return False

    def spilt_light(self, im):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sure_bg = cv2.dilate(im, kernel, iterations=1)
        binary, contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(im, contours, -1, (152, 152, 125), 5)
        light_list = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            light_list.append(box.tolist())

        armors, sure_bg = self.get_armor(light_list, sure_bg)
        return sure_bg, armors

    def get_armor(self,lights_point, img):
        armor_lights = []

        for light_i_points in lights_point:
            light_i = Light()
            light_type = light_i.light_judge(light_i_points)
            if light_type == 1:
                armor_lights.append(light_i)

                if debug_light == 1:
                    cv2.drawContours(img, [np.array(light_i.point)], 0, (152, 152, 125), 3)
        if debug_label == 1:
            if len(armor_lights) == 0:
                print "no lights "
                return [], img

        armors = []
        for i in range(len(armor_lights) - 1):

            for j in range(i + 1, len(armor_lights)):

                armor_ij = Armor()
                armor_label = armor_ij.armor_judge([armor_lights[i], armor_lights[j]])
                points = armor_lights[i].middle_line + armor_lights[j].middle_line[::-1]
                if armor_label == 1:
                    # armor_ij.points = armor_lights[i]. + armor_lights[j]

                    armors.append(armor_ij)
                else:
                    if debug_label == 1:
                        print "armor,ij:", i, j, points
        return armors, img

    def detection(self, im):
        start = time.time()*1000
        result_3 = self.get_light_3(im)
        end = time.time()*1000
        t = float(end - start) / 1000.0
        print "get_light_3 time:", t
        # print result_3
        if result_3:
            im1 = result_3[0]
            w = result_3[1]
            h = result_3[2]

            print "im ", im1.shape
            if im1.shape[0] == 0:
                return None, []
            start = time.time()*1000
            im1, armors = self.spilt_light(im1)
            end = time.time()*1000
            print "split_light time:", float(end-start)/1000.0
            # # if len(armors)>=1:
            armor_j = 0
            armors_2 = []
            for armor in armors:
                for i in range(len(armor.points)):
                    armor.points[i] += np.array([w,h])
                if debug_label_armor == 1:

                    cv2.drawContours(im1, [np.array(armor.points)], 0, (152, 152, 125), 3)
                im_ij = im[armor.line_area[1]+h:armor.line_area[3]+h,
                        armor.line_area[0]+w:armor.line_area[2]+w]
                if 0 not in np.shape(im_ij):
                    t = cv2.resize(im_ij, (30, 30))
                    t = np.array([t])
                    # a = self.sess.run(self.y_result, feed_dict={self.x: t, self.keep_prob: 1.})
                    if 1:
                        armors_2.append(armor)
                else:
                    armors_2.append(armor)
                armor_j += 1
            return im, armors_2
        else:
            return None, []

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            print e

        if self.debug == 1:
            self.i += 1
            # print "start detection",self.i
        # cv_image = cv2.resize(cv_image, (320, 240))
        a = int(time.time()*1000)
        _, armors = self.detection(cv_image)
        distant_list = []

        # try:
        #     (trans, rot) = self.tf_listener.lookupTransform('map', 'tool0', rospy.Time(0))
        #     self.pose_x = trans[0]
        #     self.pose_y = trans[1]
        #     roll, pitch, yaw = euler_from_quaternion((rot[0], rot[1], rot[2], rot[3]))
        #     self.pose_a = yaw
        # except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        #     print "cannot update pose! use last pose..."
        armors_pose = []
        if len(armors) != 0:
            for armor in armors:
                # distant = Distance()
                if self.debug == 1:
                    # print armor.points
                    cv2.drawContours(cv_image, [np.array(armor.points)], 0, (152, 152, 125), 3)
                    b = int(time.time() * 1000)
                    cost_time = float(b - a) / 1000.0
                    print 'cost time',cost_time
                    # todo:choose one distant
                self.pub_img.publish(CvBridge().cv2_to_imgmsg(cv_image, 'bgr8'))
                rvec, tvec, pitch, yaw, dist = self.distance.get_distance(armor.points)
                # self.pub_tf.sendTransform((tvec[0], tvec[1], tvec[2]), tf.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2]),
                #                           rospy.Time.now(),
                #                           "armor",
                #                           "camera0")
                self.gimab_cmd.send_cmd(self.gimab_cmd.current_pitch+pitch, self.gimab_cmd.current_yaw+yaw)
                print pitch, yaw, self.gimab_cmd.current_pitch+pitch, self.gimab_cmd.current_yaw+yaw
                self.cmd_pitch = self.gimab_cmd.current_pitch+pitch
                self.cmd_yaw = self.gimab_cmd.current_yaw+yaw
                # pose_i = self.calculate_pose([self.pose_x, self.pose_y, self.pose_a], tvec)
                # armors_pose.append(pose_i)
                # print 'rotation matrix pose: ', pose_i
                #
                # try:
                #     (trans, rot) = self.tf_listener.lookupTransform('map', 'armor', rospy.Time(0))
                #     print 'tf lookup pose: ', trans
                #
                #     armors_pose.append([trans[0], trans[1]])
                # except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                #     pass

            # return armors_pose
        else:
            self.gimab_cmd.send_cmd(self.cmd_pitch, self.cmd_yaw)
            print '----', self.cmd_pitch, self.cmd_yaw
            pass
            # self.pub_img.publish(CvBridge().cv2_to_imgmsg(cv_image, 'bgr8'))
            # self.pub_tf.sendTransform((0, 0, 0), tf.transformations.quaternion_from_euler(0, 0, 0),
            #                           rospy.Time.now(),
            #                           "armor",
            #                           "camera0")
        armors_poses = ArmorsPos()
        if len(armors_pose) >= 2:
            for i in range(len(armors_pose)):
                d01 = math.sqrt((armors_pose[0][0] - armors_pose[1][0])**2 +
                                (armors_pose[0][1] - armors_pose[1][1])**2)
                if d01 > 0.2:
                    armors_poses.armor_0 = armors_pose[0]
                    armors_poses.armor_1 = armors_pose[1]
                else:
                    d0 = math.sqrt((armors_pose[0][0])**2 + (armors_pose[0][1])**2)
                    d1 = math.sqrt((armors_pose[1][0])**2 + (armors_pose[1][1])**2)
                    if len(armors_pose)>2:
                        if d0<d1:
                            armors_poses.armor_0 = armors_pose[0]
                        else:
                            armors_poses.armor_1 = armors_pose[1]
                        armors_poses.armor_1 = armors_pose[2]
                    else:
                        if d0<d1:
                            armors_poses.armor_0 = armors_pose[0]
                        else:
                            armors_poses.armor_1 = armors_pose[1]
                        armors_poses.armor_0 = armors_pose[0]
                        armors_poses.armor_1 = [0, 0]
        elif len(armors_pose) == 1:
            armors_poses.armor_0 = armors_pose[0]
            armors_poses.armor_1 = [0, 0]
        else:
            armors_poses.armor_0 = [0, 0]
            armors_poses.armor_1 = [0, 0]
            # return 0, 0, 0
        to_w = []
        to_w.extend(armors_poses.armor_0)
        to_w.extend(armors_poses.armor_1)
        self.listened.append(to_w)
        self.armors_pub.publish(armors_poses)


def main(args):
    rospy.init_node('armor_detection', anonymous=True)
    detect = armor_detection()
    print "check detection pose ... "
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "shutting down"
    if rospy.is_shutdown():
        np.save("pose_listened.npy", np.asarray(detect.listened))

if __name__ == '__main__':
    main(sys.argv)