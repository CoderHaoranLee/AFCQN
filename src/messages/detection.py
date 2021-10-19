#! /usr/bin/env python
import sys
import rospy
import cv2
from math import *
# from timeit import timeit as timeit
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from messages.msg import EnemyPos
#from messages.msg import Odometry
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import PIL 
from test2 import *
from cnn_model import *
import profile
import time
weights = cnn_model.weights
biases = cnn_model.biases
x = tf.placeholder(tf.float32, [None, 30, 30, 3])

keep_prob = tf.placeholder(tf.float32)
pred = cnn_model.alex_net(x, weights, biases, keep_prob)
init = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, "/home/drl/ros_codes/catkin_ws/src/messages/Model2/model.ckpt")
graph = tf.get_default_graph()
prediction = graph.get_tensor_by_name('add_2:0')
y_result = tf.nn.softmax(prediction)


class circle_detetction(object):
    def __init__(self):
        self.i=0
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera_0", Image, self.callback)
        #self.chassis_sub = rospy.Subscriber("odom", Odometry, self.callback)
        self.pub_gim = rospy.Publisher('enemy_pos', EnemyPos, queue_size=1)
        self.pub_vel = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.pub_img = rospy.Publisher('detection', Image, queue_size=1)
        self.acc_yaw = 0.0
        self.debug = 1
        self.acc_pitch = 0.0


    def armor_detection(self,im):
        im1 = get_light(im)
        im1, armors = spilt_light(im1)
        armor_j = 0
        armors_2 = []
        for armor in armors:
            im_ij = im[armor.line_area[1]:armor.line_area[3],\
                    armor.line_area[0]:armor.line_area[2]]
            if 0 not in np.shape(im_ij):
        
                t = cv2.resize(im_ij, (30, 30))
                t = np.array([t])
                a = sess.run(y_result, feed_dict={x: t, keep_prob: 1.})
                if a[0][1]:
                    armors_2.append(armor)
            armor_j += 1
        # for armor in armors_2:
        #     cv2.drawContours(im, [np.array(armor.points)], 0, (152, 152, 125), 3)
        # cv2.imwrite('/home/drl/armor/region/' + str(self.i) + '.jpg', im)
        return armors_2

    def armor_detection_2(self, im):
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

                    cv2.drawContours(im1, [np.array(armor.points)], 0, (152, 152, 125), 3)
                im_ij = im[armor.line_area[1]+h:armor.line_area[3]+h,
                        armor.line_area[0]+w:armor.line_area[2]+w]
                if 0 not in np.shape(im_ij):
                    # points = [np.array([armor.line_area[0]+w, armor.line_area[1]+h]),
                    #           np.array([armor.line_area[0]+w, armor.line_area[3]+h]),
                    #           np.array([armor.line_area[2]+w, armor.line_area[3]+h]),
                    #           np.array([armor.line_area[2]+w, armor.line_area[1]+h]),
                    #           ]


                    # if cnn_label == 1:
                    #     cv2.imwrite("r2/" + str(img_i) + "_" + str(armor_j) + ".jpg", im_ij)
                    t = cv2.resize(im_ij, (30, 30))
                    t = np.array([t])
                    a = sess.run(y_result, feed_dict={x: t, keep_prob: 1.})
                    if 1:
                        armors_2.append(armor)
                else:
                    armors_2.append(armor)
                armor_j += 1

            # if debug_label == 1:
            #     imshow(im1)
            #     show()
            # if cnn_label == 1:
            #     cv2.imwrite("r3/" + str(img_i) + ".jpg", im)
            return im, armors_2
        else:
            return None,[]



    def callback(self, data):
        cost_time = 0

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            print e

        if self.debug == 1:
            self.i += 1
            print "start detection",self.i
            a = int(time.time()*1000)
        # print a

        #armors = self.armor_detection(cv_image)



        # profile.run("self.armor_detection(cv_image)")
        _, armors = self.armor_detection_2(cv_image)
        distant_list = []
        if len(armors) != 0:
            for armor in armors:
                distant = Distance()
                if self.debug == 1:
                    print armor.points
                    cv2.drawContours(cv_image, [np.array(armor.points)], 0, (152, 152, 125), 3)
                    #cv2.imwrite('/home/drl/armor_pics/'+str(self.i)+'.jpg',cv_image)
                    # cv2.imshow("image", cv_image)
                    # cv2.waitKey(15)
                    b = int(time.time() * 1000)
                    cost_time = float(b - a)/1000.0
                    print cost_time
                    # todo:choose one distant
                self.pub_img.publish(CvBridge().cv2_to_imgmsg(cv_image, 'bgr8'))
                return distant.get_distance(armor.points)
        else:
            # if self.debug == 1:
                # cv2.imwrite('/home/drl/armor_pics/' + str(self.i) + '.jpg', cv_image)
                # cv2.imshow("image", cv_image)
                # cv2.waitKey(15)
            self.pub_img.publish(CvBridge().cv2_to_imgmsg(cv_image, 'bgr8'))
            return 0, 0, 0


        #         else:
        #             min_x = x
        #             min_y = y
        #             min_r = r

        #     pix_distance = ref_distance * (min_r / 10.0)
        #     # yaw : arctan(x_off / pix_distance)
        #     # pitch: arctan(y_off / pix_distance)

        #     # self.acc_yaw += 0.005*(canvas_center[0]-min_x)
        #     # self.acc_pitch += 0.0025*(min_y-canvas_center[1])

        #     self.acc_yaw += 0.0001*(canvas_center[0]-min_x)
        #     self.acc_pitch += 0.0001*(min_y-canvas_center[1])

        #     print('yaw:')
        #     print(self.acc_yaw)
        #     print('pitch:')
        #     print(self.acc_pitch)

        #     pos = EnemyPos()
        #     pos.enemy_dist = 0.0
        #     pos.enemy_yaw = self.acc_yaw
        #     pos.enemy_pitch = self.acc_pitch
        #     self.pub_gim.publish(pos)

        #     twist = Twist()
        #     twist.linear.x = 0.001*(70-min_r); twist.linear.y = 0; twist.linear.z = 0;
        #     twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        #     self.pub_vel.publish(twist)

        #     cv2.circle(image_, (min_x, min_y), min_r, (0, 255, 0), 4)
        #     cv2.rectangle(image_, (min_x - 5, min_y - 5), (min_x + 5, min_y + 5), (0, 128, 255), -1)
        # else:
        #     pos = EnemyPos()
        #     pos.enemy_dist = 0.0
        #     pos.enemy_yaw  = self.acc_yaw
        #     pos.enemy_pitch  = self.acc_pitch
        #     self.pub_gim.publish(pos)
        #     twist = Twist()
        #     twist.linear.x += 0 ; twist.linear.y = 0; twist.linear.z = 0;
        #     twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0
        #     self.pub_vel.publish(twist)

        #     pass
        #cv2.imwrite('/home/drl/armor_pic/' + str(self.i) + '.jpg', cv_image)
        #cv2.imshow("image", cv_image)
        #cv2.waitKey(1)

def main(args):
  #  global_1=global_s()

    rospy.init_node('circle_detection', anonymous=True)
    detect = circle_detetction()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "shutting down"

if __name__ == '__main__':
	main(sys.argv)
