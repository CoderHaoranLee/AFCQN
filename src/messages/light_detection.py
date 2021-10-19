# -*- coding:utf-8 -*-
import numpy as np
from scipy.ndimage import filters
import cv2
import math
import os
#import profile
import copy

test_label = 1
armor_height = 60.0
armor_width = 123.0

class Light:
    """

    """
    def __init__(self, point_box):
        self.light_debug = 1
        self.middle_line = None#两个点
        self.left_line = None
        self.right_line = None
        self.light_color = None
        self.point = point_box
        #light 的判断标准
        # self.get_light_feature(point_box)

    def get_light_feature(self, point_box):
        if len(point_box) == 0:
            point_box = self.point
        else:
            self.point = point_box
        y_order = sorted(point_box, key=lambda x: x[-1])
        x_order = sorted(point_box, key=lambda x: x[0])
        middle_point_1 = np.int0(0.5 * (np.array(y_order[0]) + np.array(y_order[1])))
        middle_point_2 = np.int0(0.5 * (np.array(x_order[0]) + np.array(x_order[1])))
        middle_point_3 = np.int0(0.5 * (np.array(y_order[2]) + np.array(y_order[3])))
        middle_point_4 = np.int0(0.5 * (np.array(x_order[2]) + np.array(x_order[3])))
        # self.left_line = [x_order[0],x_order[1]]
        # self.right_line = [x_
        self.x_length = middle_point_4[0] - middle_point_2[0]
        self.y_length = middle_point_3[1] - middle_point_1[1]
        self.edge_length = []
        self.edge_vector = []
        for i in range(3):
            edge_vector_i = np.array(point_box[i]) - np.array(point_box[i + 1])
            length = np.linalg.norm(edge_vector_i, ord=2)
            self.edge_vector.append(edge_vector_i)
            self.edge_length.append(length)
        self.edge_vector.append(np.array(point_box[3]) - np.array(point_box[0]))
        self.edge_length.append(np.linalg.norm(np.array(point_box[3]) - np.array(point_box[0]), ord=2))
        self.min_edge = min(self.edge_length)
        self.max_edge = max(self.edge_length)
        self.middle_line = [np.array(middle_point_1), np.array(middle_point_3)]
        self.middle_length = np.linalg.norm(np.array(middle_point_1) - np.array(middle_point_3), ord=2)
        self.centre_point = np.int0(0.5*(self.middle_line[0] + self.middle_line[1]))


    def light_judge(self):
        """

        :param point_box:
        :return:
        """

        # edge_length
        self.edge_length = []
        self.edge_vector = []
        for i in range(3):
            edge_vector_i = np.array(self.point[i]) - np.array(self.point[i + 1])
            length = np.linalg.norm(edge_vector_i, ord=2)
            self.edge_vector.append(edge_vector_i)
            self.edge_length.append(length)
        self.edge_vector.append(np.array(self.point[3]) - np.array(self.point[0]))
        self.edge_length.append(np.linalg.norm(np.array(self.point[3]) - np.array(self.point[0]), ord=2))
        self.min_edge = min(self.edge_length)
        self.max_edge = max(self.edge_length)
        if self.max_edge > 100 and self.min_edge < 2:
            if self.light_debug == 1:
                print "light:", self.max_edge, self.min_edge
            return False
        if self.min_edge == 0:
            if self.light_debug == 1:
                print "light:", self.max_edge, self.min_edge
            return False
        if self.max_edge / self.min_edge < 1.2:
            if self.light_debug == 1:
                print "light:", self.max_edge, self.min_edge
            return False

        y_order = sorted(self.point, key=lambda x: x[-1])
        x_order = sorted(self.point, key=lambda x: x[0])
        middle_point_1 = np.int0(0.5 * (np.array(y_order[0]) + np.array(y_order[1])))
        # middle_point_2 = np.int0(0.5 * (np.array(x_order[0]) + np.array(x_order[1])))
        middle_point_3 = np.int0(0.5 * (np.array(y_order[2]) + np.array(y_order[3])))
        # middle_point_4 = np.int0(0.5 * (np.array(x_order[2]) + np.array(x_order[3])))
        self.middle_line = [np.array(middle_point_1), np.array(middle_point_3)]
        self.middle_length = np.linalg.norm(np.array(middle_point_1) - np.array(middle_point_3), ord=2)
        self.centre_point = np.int0(0.5 * (self.middle_line[0] + self.middle_line[1]))

        if self.middle_length < 5:
            if self.light_debug == 1:
                print "middle_length:", self.middle_length
            return False
        # if self.x_length == 0:
        #     if self.light_debug == 1:
        #         print ":", self.max_edge, self.min_edge
        #     return False
        # if self.x_length <= self.y_length:
        #     ratio = float(self.y_length)/float(self.x_length)
        #     if ratio > 1.5:
        # self.middle_line = [np.array(x_order[]), np.array(middle_point_3)]

        return True

class Armor:
    def __init__(self,light_pair):
        self.points = None
        self.max_edge_ratio = 5 #最长边与最短边的比
        self.min_length = 10
        self.max_angle = 137
        self.min_angle = 55
        self.min_opedge_ratio = 0.33 #对边的比例下限
        self.max_opedge_ratio = 3
        self.min_adjedge_ratio = 0.25  #
        self.max_adjedge_ratio = 3
        self.middle_line_angle = 30 # 装甲板横中线与水平线的夹角
        self.max_middleline_ratio = 3  #中线的比例
        self.second_edge_ratio = 2.5 # 临边的比例上限
        self.line_area = None
        self.get_armor_feature(light_pair)


    def get_armor_feature(self, light_pair):
        if light_pair[0].centre_point[0]< light_pair[1].centre_point[0]:
            self.left_light = light_pair[0]
            self.right_light = light_pair[1]

        else:
            self.left_light = light_pair[1]
            self.right_light = light_pair[0]
        self.points = copy.deepcopy(self.left_light.middle_line + self.right_light.middle_line[::-1])
        point_box = self.points
        # y_order = sorted(point_box, key=lambda x: x[-1])
        # x_order = sorted(point_box, key=lambda x: x[0])
        # middle_point_1 = np.int0(0.5 * (np.array(y_order[0]) + np.array(y_order[1])))
        # middle_point_2 = np.int0(0.5 * (np.array(x_order[0]) + np.array(x_order[1])))
        # middle_point_3 = np.int0(0.5 * (np.array(y_order[2]) + np.array(y_order[3])))
        # middle_point_4 = np.int0(0.5 * (np.array(x_order[2]) + np.array(x_order[3])))
        self.edge_length = []
        self.edge_vector = []
        self.edge_length.append(self.left_light.middle_length)
        self.edge_vector.append(self.left_light.middle_line[1]-self.left_light.middle_line[0])
        self.edge_length.append(np.linalg.norm(self.right_light.middle_line[1]-self.left_light.middle_line[1],ord=2))
        self.edge_vector.append(self.right_light.middle_line[1]-self.left_light.middle_line[1])
        self.edge_length.append(self.right_light.middle_length)
        self.edge_vector.append(self.right_light.middle_line[1] - self.right_light.middle_line[0])
        self.edge_length.append(np.linalg.norm(self.right_light.middle_line[0] - self.left_light.middle_line[0], ord=2))
        self.edge_vector.append(self.right_light.middle_line[1] - self.left_light.middle_line[1])
        self.min_edge = min(self.edge_length)
        self.max_edge = max(self.edge_length)

    def get_edge_length(self):
        self.edge_length = []
        self.edge_length.append(self.left_light.middle_length)
        self.edge_length.append(np.linalg.norm(self.right_light.middle_line[1] - self.left_light.middle_line[1], ord=2))
        self.edge_length.append(self.right_light.middle_length)
        self.edge_length.append(np.linalg.norm(self.right_light.middle_line[0] - self.left_light.middle_line[0], ord=2))


    def get_edge_vector(self):
        self.edge_vector = []
        self.edge_vector.append(self.left_light.middle_line[1] - self.left_light.middle_line[0])
        self.edge_vector.append(self.right_light.middle_line[1] - self.left_light.middle_line[1])
        self.edge_vector.append(self.right_light.middle_line[1] - self.right_light.middle_line[0])
        self.edge_vector.append(self.right_light.middle_line[1] - self.left_light.middle_line[1])

    def armor_judge(self):
        self.get_edge_length()
        self.min_edge = min(self.edge_length)
        self.max_edge = max(self.edge_length)

        # 最长边与最短边
        edge_ratio = float(self.max_edge)/self.min_edge
        if edge_ratio > self.max_edge_ratio:
            if self.debug_label == 1:
                print "1、edge_ratio:", edge_ratio
            return 0

        # 对边的比例筛选
        if self.min_edge > 0:
            for i in range(2):
                opedge_ratio = self.edge_length[i] / self.edge_length[i+2]
                if opedge_ratio < self.min_opedge_ratio or opedge_ratio > self.max_opedge_ratio:
                    if self.debug_label == 1:
                        print "1、opedge_ratio:", opedge_ratio
                    return 0

        # 按临边比例筛选
        for i in range(4):
            j = (i + 1) / 3
            adjacent_edge_ratio = self.edge_length[i] / self.edge_length[j]
            if adjacent_edge_ratio < self.min_adjedge_ratio or adjacent_edge_ratio > self.max_adjedge_ratio:
                if self.debug_label == 1:
                    print "second_edge_ratio:", adjacent_edge_ratio

                return 0

        #夹角的限定范围
        self.get_edge_vector()
        for i in range(3):
            angle = np.arccos(np.dot(self.edge_vector[i],
                              (self.edge_vector[i+1])/(self.edge_length[i]*self.edge_length[i+1])))
            angle = (angle*180)/math.pi
            if angle > self.max_angle or angle < self.min_angle:
                if self.debug_label == 1:
                    print "3、angle:", angle
                return 0



        #按照横向中线与水平面夹角范围进行筛选
        left_middle_point = self.left_light.centre_point
        right_middle_point = self.right_light.centre_point

        if right_middle_point[0] - left_middle_point[0] != 0:

            line_angle = math.atan(abs(float(right_middle_point[1] - left_middle_point[1]))
                                   / float(right_middle_point[0] - left_middle_point[0]))
        else:
            return 0
        line_angle = abs(line_angle * 180) / math.pi
        if line_angle > self.middle_line_angle:
            if self.debug_label == 1:
                print "6、middle_line_angle:", line_angle
            return 0
        # 横向中线与纵向中线的比例

        # x_length = middle_point_4[0] - middle_point_2[0]-0.5*light_1.x_length - 0.5*light_2.x_length
        # y_length = middle_point_3[1] - middle_point_1[1]
        # if min_lenth != 0:
        #     ratio = float(max_length)/float(min_lenth)
        # else:
        #     return 0
        #
        # length = [x_length,y_length]
        # min_lenth = min(length)
        # max_length = max(length)
        #
        # if ratio < self.max_middleline_ratio:
        #
        #         return 0
        # else:
        #     if self.debug_label == 1:
        #         print "5、middle_line_ratio:", ratio
        #     return 0
        # TODO:确定要裁切的下方区域四个坐标
        x_raw_length = right_middle_point[0] - left_middle_point[0]
        light_1 = self.left_light
        light_2 = self.right_light
        if light_1.point[1][0] > light_2.point[1][0]:
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
        if test_label == 1:
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
        return 1

class Distance:
    def __init__(self):
        self.armor_points = np.array([[[armor_width],[ 0.0], [0.0]],
                                      [[armor_width], [armor_height], [0.0]],
                                      [[armor_width], [armor_height], [0.0]],
                                      [[0.0], [armor_height], [0.0]]])
        self.image_points = np.random.random((4,2,1))
        #
        self.camera_matrix = np.array([[520.0334010310696, 0.0, 304.8460838356355],
                                 [0.0, 522.1145816053537, 268.2227873795626],
                                 [0.0, 0.0, 1.0]])
        self.camera_distortion =np.array([[0.05985687918256749],[-0.0825295157658502],[0.004130642265799225],
        [0.001066639503312635],[0]])
        self.xyz = None
        self.distance = None
        self.pitch = None
        self.yaw = None


    def get_distance(self, image_points):
        imagePoints = np.array(image_points, dtype=np.float64).reshape((4, 2, 1))
        # print np.shape(imagePoints),imagePoints
        # print np.shape(self.armor_points),np.shape(np.array(image_points)),np.shape(self.camera_matrix)
        # print np.shape(self.camera_distortion)
        _,_,tvec = cv2.solvePnP(self.armor_points, imagePoints,self.camera_matrix,self.camera_distortion)
        #print tvec
        fly_time = tvec[1]/float(1000*20)
        gravity_offset = 0.5*9.8*fly_time*1000
        xyz = [tvec[0],tvec[1]-gravity_offset+33,tvec[2]]
        self.xyz = xyz
        pitch = math.atan(-xyz[1]/xyz[2])
        yaw = math.atan2(xyz[0], xyz[2])
        pitch = pitch*180/np.pi
        yaw = yaw*180/np.pi
        dis = (xyz[0]**2+xyz[2]**2)**0.5
        self.distance = dis
        self.pitch = pitch
        self.yaw = pitch
        print "俯仰角:",pitch
        print "偏航角",yaw
        print "距离:",dis[0]
        # return pitch, yaw, dis[0]



def filter_robo(armor_list, armor_dis):
    possible_robo = {}
    nearest_point = []
    for i in range(len(armor_list)):
        possible_robo[i] = []
        for j in range(len(armor_list)-1):
            ij_distance = np.linalg.norm((np.array(armor_list[i].xyz)-np.array(armor_list[j].xyz)),ord=2)
            if ij_distance <= armor_dis:
                possible_robo[i].append(j)
                if armor_list[i].distance < armor_list[j].distance:
                    nearest_point.append(armor_list[i])
                else:
                    nearest_point.append(armor_list[j])
    return set(nearest_point)



