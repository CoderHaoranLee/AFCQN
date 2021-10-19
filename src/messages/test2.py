# -*- coding:utf-8 -*-
from pylab import *
import numpy as np
# from PIL import Image
from scipy.ndimage import filters
import cv2
import math
from skimage import morphology,data,color
import os
import tensorflow as tf
import sys
print sys.path
import cnn_model
import profile
import copy
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
        self.middle_line = None#两个点
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
        # self.left_line = [x_order[0],x_order[1]]
        # self.right_line = [x_
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
            if  debug_light == 1:
                print "light:",max_edge,min_edge
            return False
        edge_vector = []
        edge_vector.append(np.array(point_box[3])-np.array(point_box[0]))
        edge_length.append(np.linalg.norm(np.array(point_box[3])-np.array(point_box[0]), ord=2))
        if self.x_length == 0:
            return False
        # if self.x_length <= self.y_length:
        #     ratio = float(self.y_length)/float(self.x_length)
        #     if ratio > 1.5:

        self.middle_line = [np.array(middle_point_1), np.array(middle_point_3)]
        self.middle_length = np.linalg.norm(np.array(middle_point_1)-np.array(middle_point_3), ord=2)
        # self.middle_line = [np.array(x_order[]), np.array(middle_point_3)]

        return 1 #armor_light
        # else:
        #     #TODO:血量条检测规则
        #     return False



class Armor:
    def __init__(self):
        self.points = None
        self.edge_ratio = 3.5
        self.min_edge = 10
        self.max_angle = 137
        self.min_angle = 55
        self.min_edge_ratio = 0.33 #对边的比例
        self.max_edge_ratio = 3
        self.middle_line_angle = 30 # 装甲板横中线与水平线的夹角
        self.middle_line_ratio = 3
        self.second_edge_ratio = 2.5

        # self.standard_dic = {
        #     "edge_ratio":4, #最长边与最短边的比
        #     "min_edge":0,
        #     "max_angle" :130,
        #     "min_angle" : 55,
        #     "min_edge_ratio" : 0.7,#对边的长度比
        #     "max_edge_ratio":1.5,
        #     "middle_line_angle":40,
        #     "middle_line_ratio":3
        # }
        self.line_area = None



    def armor_judge(self, light_pair):
        # 判断规则是中心的线与水平线的夹角不超过60度，中心最长边与最短边的比例不会超过三
        # ，最大边与最短边不会超过2-3倍，相邻两条边的夹角（60,120）

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
                if debug_label == 1:
                    print "1、edge_ratio:", edge_ratio
                return 0
        else:
            if debug_label == 1:
                print "2、min_edge = 0"
            return 0
        second_edge_ratio = self.edge_length[2]/self.edge_length[1]
        if second_edge_ratio >3:
            if debug_label == 1:
                print "9、second_edge_ratio:", second_edge_ratio

            return 0

        edge_angle = []
        for i in range(3):
            angle = np.arccos(np.dot(edge_vector[i],
                              (edge_vector[i+1])/(self.edge_length[i]*self.edge_length[i+1])))
            angle =(angle*180)/math.pi
            if angle > self.max_angle or angle < self.min_angle:
                if debug_label == 1:
                    print "3、angle:", angle
                return 0

        for i in range(2):
            edge_ratio_2 = self.edge_length[i+2]/self.edge_length[i]
            if edge_ratio_2 > self.max_edge_ratio or edge_vector < self.min_edge_ratio :
                if debug_label == 1:
                    print "4、edge_ratio_2:", edge_ratio_2
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
                if debug_label == 1:
                    print "6、middle_line_angle:", line_angle
                return 0
        else:
            if debug_label == 1:
                print "5、middle_line_ratio:", ratio
            return 0
        # TODO:确定要裁切的下方区域四个坐标
        if light_1.point[1][0]> light_2.point[1][0]:
            self.left_light = light_2.middle_line
            self.right_light = light_1.middle_line
        else:
            self.left_light = light_1.middle_line
            self.right_light = light_2.middle_line

        # if self.left_light[1][1] >= self.right_light[1][1]:
        #     under_point_1 = self.left_light[1][0]
        #     under_point_2 = self.right_light[1][1]
        #     under_point_3 = self.left_light[1][0]+x_raw_length
        #     under_point_4 =min(self.right_light[1][1]+x_raw_length,640)
        # else:
        #     under_point_1 = self.left_light[1][0]
        #     under_point_2 = self.left_light[1][1]
        #     under_point_3 = self.left_light[1][0] + x_raw_length
        #     under_point_4 = min(self.left_light[1][1] + x_raw_length, 640)
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




def light_cross(light_1, light_2):
    #判断两灯光有无交叉，返回左右两灯光对象
    pass


def get_armor(lights_point,img):
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
            return [],img

    armors = []
    for i in range(len(armor_lights)-1):

        for j in range(i+1, len(armor_lights)):

            armor_ij = Armor()
            armor_label = armor_ij.armor_judge([armor_lights[i], armor_lights[j]])
            points = armor_lights[i].middle_line + armor_lights[j].middle_line[::-1]
            if armor_label == 1:
                # armor_ij.points = armor_lights[i]. + armor_lights[j]

                armors.append(armor_ij)
            else:
                if debug_label == 1:
                    print "armor，ij:",i,j, points
    return armors,img





def line_detection(img,points,i):

    #TODO:线检测
    region = img.crop(points)
    cv2.imwrite("region/"+str(i)+".jpg")
    pass


def sobel_filter(imag):
    imx = zeros(imag.shape)
    filters.sobel(imag, -1, imx)

    imy = zeros(imag.shape)
    filters.sobel(imag, -1, imy)
    img = sqrt(imx ** 2 + imy ** 2)
    return img


def line_houghline(img_name):

    img = cv2.imread(img_name)
    # img = cv2.resize(img,(640,512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120)
    minLineLength = 80
    maxLineGap = 5
    lines  = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength, maxLineGap )
    for i in range(len(lines)):
        for x1,y1,x2,y2 in lines[i]:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("edges", edges)
    cv2.imshow("lines",img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def get_hsv(imgname):
    image = cv2.imread(imgname)
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def getpos(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(HSV[y, x])
            # th2=cv2.adaptiveThreshold(imagegray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    cv2.imshow("imageHSV", HSV)
    cv2.imshow('image', image)
    cv2.setMouseCallback("imageHSV", getpos)
    cv2.waitKey(0)

def spilt_light(im):
    # cv2.imshow("light_im", im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # im = cv2.imread(imgname)
    # mask = morphology.skeletonize(im)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    sure_bg = cv2.dilate(im, kernel, iterations = 1)
    # cv2.imshow("raw_light", sure_bg)
    # show()
    #
    # dist_transform = cv2.distanceTransform(im, cv2.DIST_L2, 5)
    # #
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    # sure_fg = np.uint8(sure_fg)
    # unknown = cv2.subtract(sure_bg, sure_fg)
    # ret, markers = cv2.connectedComponents(sure_fg)
    # markers = markers + 1
    # markers[unknown == 255] = 0
    # im = unknown
    #
    # markers = cv2.watershed(im,markers)
    # # im[markers == -1] = [255,0,0]

    binary, contours, hierarchy = cv2.findContours(sure_bg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(im, contours, -1, (152, 152, 125), 5)
    light_list = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(im, (x,y), (x+w,y+h), (0,255,0),2)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        light_list.append(box.tolist())
        # if debug_label == 1:
        #     cv2.drawContours(sure_bg, [box], 0, (152, 152, 125),3)
    armors, sure_bg = get_armor(light_list, sure_bg)
    # for armor in armors:
    #     # print armor
    #     cv2.drawContours(sure_bg, [np.array(armor.points)], 0, (152, 152, 125), 3)






    # im = sure_bg
    return sure_bg,armors
    #
    #




def get_light(img):
    # im = imread(imgname)
    # im = cv2.resize(im, (640, 512))
    if army_color == "B":
        lower_blue = np.array([0, 0, 200])
        upper_blue = np.array([190, 70, 255])
    else:
        #todo:红色数据待改
        lower_blue = np.array([20, 0, 200])
        upper_blue = np.array([190, 70, 255])

    #cv2.imshow('Capture', img)  # change to hsv model
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # get mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue, )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # mask = cv2.erode(mask, kernel)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # ret, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )
    # mask, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour = contours[0]
    # cv2.drawContours(mask, [contour], 0, (0, 255, 0), 2)
    # mask = cv2.minAreaRect(mask)
    # box = cv2.cv.BoxPoints(mask)

    # cv2.imshow('Mask', mask)  # detect blue
    # res = cv2.bitwise_and(im, im, mask=mask)
    return mask

def get_light_2(img):
    b, g, r = cv2.split(img)
    thresh = 0
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,gray_binary = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    # cv2.imshow("gray", gray_binary)
    if army_color == "R":
        thresh = 50
        result_img = cv2.subtract(r, g)
    else:
        result_img = cv2.subtract(b, g)
        thresh = 90


    # cv2.imshow("color", result_img)
    # cv2.imshow("color", b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret, result_img = cv2.threshold(result_img, thresh, 255, cv2.THRESH_BINARY)
    index_range = np.argwhere(result_img == 255)
    # print np.shape(index_range)
    try:
        y_down = index_range[:, 0].max()
        y_up = index_range[:, 0].min()
        x_right = index_range[:, 1].max()
        x_left = index_range[:, 1].min()
    except Exception as e:
        return False
        print e
    #y_down = index_range[:,0].max()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    result_img = cv2.dilate(result_img,kernel)
    cc = result_img & gray_binary
    # cv2.imshow("Blue", cc)
    # cv2.imshow("Red", g)
    # cv2.imshow("Green", b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return result_img,[x_left,y_up,x_right,y_down]

def get_light_3(img):
    result_light = get_light_2(img)
    if result_light:
        result_2 = result_light[0]
        box_point = result_light[1]
    # print box_point
        result_1 = get_light(img)
        result_3 = result_1[box_point[1]:box_point[3],box_point[0]:box_point[2]]
        # cv2.imshow("result_3", result_1)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return result_3,box_point[0],box_point[1]
    else:
        return False






# def get_point(image_name):
#     im = np.array(Image.open(image_name).convert("L"))
#     imshow(im)
#     x = ginput(4)
#     for i in x:
#         print im[int(i[0])][int(i[1])]
#     print x
#     show()

def area_judge(sub_img):

    # 构建模型
    weights = cnn_model.weights
    biases = cnn_model.biases
    x = tf.placeholder(tf.float32, [None, 30, 30, 3])
    # y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    pred = cnn_model.alex_net(x, weights, biases, keep_prob)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.getcwd()+"/Model/model.ckpt")
        graph = tf.get_default_graph()
        prediction = graph.get_tensor_by_name('add_2:0')
        y_result = tf.nn.softmax(prediction)
        # print("v1:", sess.run(weights))  # 打印v1、v2的值和之前的进行对比
        # print("v2:", sess.run(biases))
        # print(sess.run(prediction, feed_dict={x: sub_img, keep_prob: 1.}))
        a = sess.run(y_result, feed_dict={x: sub_img, keep_prob: 1.})
        print a
        print(a.shape)
        if a[0][1] == 1:
            return True
        else:
            return False

        # [ 3.]


def detection_main(im ):
        #print np.shape(im)
        # get_point(imname)
        # get_hsv(imname)
        result_3 = get_light_3(im)
        if result_3:
            im1 = result_3[0]
            w = result_3[1]
            h = result_3[2]
            im1, armors = spilt_light(im1)
            # # if len(armors)>=1:
            armor_j = 0
            armors_2 = []
            for armor in armors:
                for i in range(len(armor.points)):
                    armor.points[i] += np.array([w,h])
                if debug_label_armor == 1:
                    # print armor
                    cv2.drawContours(im1, [np.array(armor.points)], 0, (152, 152, 125), 3)
                im_ij = im[armor.line_area[1]+h:armor.line_area[3]+h,
                        armor.line_area[0]+w:armor.line_area[2]+w]
                if 0 not in np.shape(im_ij):
                    points = [np.array([armor.line_area[0]+w, armor.line_area[1]+h]),
                              np.array([armor.line_area[0]+w, armor.line_area[3]+h]),
                              np.array([armor.line_area[2]+w, armor.line_area[3]+h]),
                              np.array([armor.line_area[2]+w, armor.line_area[1]+h]),
                              ]

                    # cv2.drawContours(im, [np.array(points)], 0, (222, 0, 0), 3)
                    # im_ij = cv2.resize(im_ij,(30,30))
                    if cnn_label == 1:
                        cv2.imwrite("r2/" + str(img_i) + "_" + str(armor_j) + ".jpg", im_ij)
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
                    a = sess.run(y_result, feed_dict={x: t, keep_prob: 1.})
                    if 1:
                        armors_2.append(armor)
                else:
                    armors_2.append(armor)
                armor_j += 1

                # judge_result = area_judge(t)
                # if judge_result:
                #     armors_2.append(armor)

                # saver = tf.train.import_meta_graph(os.getcwd()+"/Model/model.ckpt.meta")
                #
                # with tf.Session() as sess:
                #     saver.restore(sess, "./Model/model.ckpt")  # 注意路径写法
                #
                #     x = tf.placeholder(tf.float32, [None, 30, 30, 3])
                #     y = tf.placeholder(tf.float32, [None, 2])
                #     keep_prob = tf.placeholder(tf.float32)
                #     ll = np.array([[0,0]])
                #     print t.shape,ll.shape
                #
            # # # im1 = np.array(Image.open(imname).convert("L"))
            # # # im1 = sobel_filter(im1)
            # # line_houghline(imname)
            # print 3

            for armor in armors_2:
                # print armor
                cv2.drawContours(im, [np.array(armor.points)], 0, (152, 152, 125), 3)
            if debug_label == 1:
                imshow(im1)
                show()
            if cnn_label == 1:
                cv2.imwrite("r3/" + str(img_i) + ".jpg", im)
            return im, armors_2
        else:
            return None,[]

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


    def get_distance(self, image_points):
        imagePoints = np.array(image_points, dtype=np.float64).reshape((4, 2, 1))
        print np.shape(imagePoints),imagePoints
        print np.shape(self.armor_points),np.shape(np.array(image_points)),np.shape(self.camera_matrix)
        print np.shape(self.camera_distortion)
        _,_,tvec = cv2.solvePnP(self.armor_points, imagePoints,self.camera_matrix,self.camera_distortion)
        print tvec
        fly_time = tvec[1]/float(1000*20)
        gravity_offset = 0.5*9.8*fly_time*1000
        xyz = [tvec[0],tvec[1]-gravity_offset+33,tvec[2]]
        pitch = math.atan(-xyz[1]/xyz[2])
        yaw = math.atan2(xyz[0], xyz[2])
        pitch = pitch*180/np.pi
        yaw = yaw*180/np.pi
        dis = (xyz[0]**2+xyz[2]**2)**0.5
        print pitch, yaw, dis
        return pitch, yaw, dis[0]



     

# if __name__ == "__main__":
#
#     weights = cnn_model.weights
#     biases = cnn_model.biases
#     x = tf.placeholder(tf.float32, [None, 30, 30, 3])
#     # y = tf.placeholder(tf.float32, [None, 2])
#     keep_prob = tf.placeholder(tf.float32)
#     pred = cnn_model.alex_net(x, weights, biases, keep_prob)
#
#     init = tf.initialize_all_variables()
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         saver.restore(sess, "/home/drl/catkin_ws/Model2/model.ckpt")
#         graph = tf.get_default_graph()
#         prediction = graph.get_tensor_by_name('add_2:0')
#         y_result = tf.nn.softmax(prediction)
#         for img_i in range(5, 6):
#             imname ='output/' + str(img_i) + '.jpg'
#
#             print imname, "-----------------"
#             im = cv2.imread(imname)
#             if im is None:
#                 print "read none"
#
#             else:
#                 # get_hsv(imname)
#                 # profile.run("detection_main(im)")
#
#                 _,armors = detection_main(im)
#                 if len(armors) != 0:
#                     for armor in armors:
#
#                         distant = Distance()
#                         print "armor", armor.points
#                         distant.get_distance(armor.points)
#                 else:
#                      print 0,0,0
#                 # im_re = get_light_3(im)
#                 # imshow(im_re)
#                 # show()


