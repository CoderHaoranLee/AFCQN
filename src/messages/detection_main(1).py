# -*- coding:utf-8 -*-
import numpy as np
from scipy.ndimage import filters
import cv2
from pylab import *
import math
import os
import tensorflow as tf
import cnn_model
import profile
import copy
from light_detection import *
army_color = "B"
class Armor_Detection:
    def __init__(self, img):
        self.img = img
        self.army_color = army_color
        self.debug_label = 0
        self.detect_None = False


    def get_light_area_1(self):
        b, g, r = cv2.split(self.img)
        if self.army_color == "R":
            thresh = 50
            result_img = cv2.subtract(r, g)
        else:
            result_img = cv2.subtract(b, g)
            thresh = 20
        # cv2.imshow("color", result_img)
        # cv2.waitKey()
        ret, result_img = cv2.threshold(result_img, thresh, 255, cv2.THRESH_BINARY)
        index_range = np.argwhere(result_img == 255) #根据阈值筛选目标区域
        if len(index_range) == 0:
            self.detect_None = True
            print "no light area"
            return None
        y_down = index_range[:, 0].max()
        y_up = index_range[:, 0].min()
        x_right = index_range[:, 1].max()
        x_left = index_range[:, 1].min()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result_img = cv2.dilate(result_img, kernel)  #对目标区域做膨胀处理
        self.mask_1 = [x_left, y_up, x_right, y_down]
        self.light_area = result_img
        self.w = x_left
        self.h = y_up
        # cv2.imshow("color",result_img)
        # cv2.waitKey()
        return [x_left, y_up, x_right, y_down]

    def get_light_area_2(self, sub_img):
        if army_color == "B":
            lower_blue = np.array([0, 0, 200])
            upper_blue = np.array([190, 70, 255])
        else:
            # todo:红色数据待改
            lower_blue = np.array([20, 0, 200])
            upper_blue = np.array([190, 70, 255])

        # cv2.imshow('Capture', img)  # change to hsv model
        hsv = cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)  # get mask
        mask = cv2.inRange(hsv, lower_blue, upper_blue, )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # mask = cv2.erode(mask, kernel)
        self.mask_2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def get_light(self):
        box_point = self.get_light_area_1()
        if self.detect_None:
            return None
        else:
            self.light_img = self.img[box_point[1]:box_point[3], box_point[0]:box_point[2]]
        im = self.get_light_area_2(self.light_img)
        # try:
        #     im = result_1[box_point[1]:box_point[3], box_point[0]:box_point[2]]
        #     self.light_img = self.img[box_point[1]:box_point[3], box_point[0]:box_point[2]]
        # except:
        #     self.detect_None = True
        #     return None
        if self.detect_None:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        sure_bg = cv2.dilate(im, kernel, iterations=1)
        binary, contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #分出单个灯条
        if len(contours) == 0:
            self.detect_None = True
            return None
        if self.debug_label == 1:
            cv2.drawContours(im, contours, -1, (152, 152, 125), 5)
        light_list = []
        for c in contours:
            #框出灯条box,用列表light_list 存放
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            light_list.append(box.tolist())
            # if self.debug_label == 1:
            #     cv2.drawContours(sure_bg, [box], 0, (152, 152, 125),3)
        armor_lights = []
        for light_i_points in light_list:
            light_i = Light(light_i_points)
            light_type = light_i.light_judge()
            if light_type == 1:
                armor_lights.append(light_i)
                # todo:对light_list从左到右排序
                if self.debug_label == 1:
                    cv2.drawContours(sure_bg, [np.array(light_i.point)], 0, (152, 152, 125), 3)
        if len(armor_lights) == 0:
            self.detect_None = 1
            print "no lights "
            return None
        # return armor_lights, sure_bg
        return armor_lights

    def armor_filter(self):
        armor_lights = self.get_light()
        if self.detect_None == 1:
            return None
        armors = []
        for i in range(len(armor_lights) - 1):
            for j in range(i + 1, len(armor_lights)):
                armor_ij = Armor([armor_lights[i], armor_lights[j]])
                armor_ij.debug_label = self.debug_label
                armor_label = armor_ij.armor_judge() #判断可能的装甲是否满足条件

                if armor_label == 1:
                     armors.append(armor_ij)
                     cv2.drawContours(self.light_img, [np.array(armor_ij.points)], 0, (152, 152, 125), 3)
                     # cv2.imshow("tt", img)
                     # cv2.waitKey()


                else:
                    if self.debug_label == 1:
                        points = armor_lights[i].middle_line + armor_lights[j].middle_line[::-1]
                        print "armor，ij:", i, j, armor_ij.points
                    pass

        if len(armors) == 0:
            self.detect_None = 1
            print "no armor"
            return None
        return armors

def detection_main(im):
        armor_detect = Armor_Detection(im)
        result = armor_detect.armor_filter()
        if armor_detect.detect_None == 1:
            return []
        else:
            armors = result

        # # if len(armors)>=1:
        armor_j = 0
        armors_2 = []
        for armor in armors:
            for i in range(len(armor.points)):
                armor.points[i] += np.array([armor_detect.w, armor_detect.h])
            cv2.drawContours(im, [np.array(armor.points)], 0, (152, 152, 125), 3)
            im_ij = im[armor.line_area[1] +  armor_detect.h:armor.line_area[3] + armor_detect.h,
                    armor.line_area[0] +  armor_detect.w:armor.line_area[2] +  armor_detect.w]
            if 0 not in np.shape(im_ij):
                # points = [np.array([armor.line_area[0] +  armor_detect.w, armor.line_area[1] +  armor_detect.h]),
                #           np.array([armor.line_area[0] +  armor_detect.w, armor.line_area[3] +  armor_detect.h]),
                #           np.array([armor.line_area[2] +  armor_detect.w, armor.line_area[3] +  armor_detect.h]),
                #           np.array([armor.line_area[2] +  armor_detect.w, armor.line_area[1] +  armor_detect.h]),
                #           ]

                # cv2.drawContours(im, [np.array(points)], 0, (222, 0, 0), 3)
                # im_ij = cv2.resize(im_ij,(30,30))
                # if cnn_label == 1:
                #     cv2.imwrite("r2/" + str(img_i) + "_" + str(armor_j) + ".jpg", im_ij)
                t = cv2.resize(im_ij, (30, 30))
                # print np.max(tt)
                # t = np.array(Image.open("r2/" + str(img_i)+"_"+str(armor_j) + ".jpg"))
                # print np.max(t)
                # t = t.resize((30, 30))
                # # cv2.imshow("tt",tt)
                # # cv2.waitKey(0)
                # cv2.imshow("t", t)
                # cv2.waitKey(0)
                # cv2.imwrite("r2/" + str(img_i) + "_" + str(armor_j) + "_" + ".jpg", t)
                t = np.array([t])
                # a = sess.run(y_result, feed_dict={x: t, keep_prob: 1.})
                if 1:
                    armors_2.append(armor)
            else:
                armors_2.append(armor)
            armor_j += 1
        return im, armors_2


if __name__ == "__main__":

    weights = cnn_model.weights
    biases = cnn_model.biases
    x = tf.placeholder(tf.float32, [None, 30, 30, 3])
    # y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    pred = cnn_model.alex_net(x, weights, biases, keep_prob)
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "/home/nvidia/catkin_ws/src/messages/Model2/model.ckpt")
        graph = tf.get_default_graph()
        prediction = graph.get_tensor_by_name('add_2:0')
        y_result = tf.nn.softmax(prediction)
        name_list = os.listdir("temp_pics/")
        for imname in name_list:
        # for imname in range( 64,80):
            # imname ='output/' + str(img_i) + '.jpg'
            # imname ="434.jpg"
            filename = 'temp_pics/' + imname
            print imname, "-----------------"
            im = cv2.imread(filename)
            if im is None:
                print "read none"

            else:
                # get_hsv(imname)
                profile.run("detection_main(im)")

                # result = detection_main(im)
                #
                # if len(result) != 0:
                #     im1 = result[0]
                #     armors = result[1]
                #     centre_points = []
                #     for armor in armors:
                #
                #         distant = Distance()
                #         print "armor_points:", armor.points
                #         distant.get_distance(armor.points)
                #         centre_points.append(distant)
                #     # cv2.imshow("tt", im1)
                #     # cv2.imwrite("r3/" + str(img_i) + ".jpg", im1)
                #     cv2.imwrite("temp_result/" + imname, im1)
                #     print 2
                #
                #     final_distance = filter_robo(centre_points, 30)#返回的distance对象
                # else:
                #     cv2.imwrite("temp_result/" + imname, im)
                #
                # #
                #





