#!/usr/bin/env python

import rospy
import tf
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from tf_tools import *
import matplotlib.pyplot as plt
import time
import thread
import cv2

class lidar_scan(object):
    def __init__(self):
        self.scan_sub_ = rospy.Subscriber("scan", LaserScan, self.scan_callback)
        self.laser_project_ = LaserProjection()

        self.scan_x = 0.0
        self.scan_y = 0.0
        self.scan_a = 0.0
        self.scan_tf_listener_ = tf.TransformListener()

        self.points = np.array([])

        self.enemy_pos = []
        self.enemy_pos_sub_ = None
        self.dist_th = 0.5
        self.enemy_tf_listener_ = tf.TransformListener()

        self.lidar_est = []

    def scan_callback(self, data):
        # update scan pose

        try:
            (trans, rot) = self.scan_tf_listener_.lookupTransform("map", "laser", rospy.Time(0))
            self.scan_x = trans[0]
            self.scan_y = trans[1]
            roll, pitch, yaw = euler_from_quaternion((rot[0], rot[1], rot[2], rot[3]))
            self.scan_a = yaw
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print "update scan pose faild! use last pose"
        cloud = self.laser_project_.projectLaser(scan_in=data)
        gen = pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
        points = []
        for p in gen:
            points.append(p)
        points = np.asarray(points).transpose()
        points = points[:2, :]
        # print points.shape

        R = np.array([[math.cos(self.scan_a), -math.sin(self.scan_a)],
                      [math.sin(self.scan_a), math.cos(self.scan_a)]])
        points = np.dot(R, points)
        tvec = np.asarray([[self.scan_x], [self.scan_y]])
        self.points = points + tvec
        self.update_enemy()
        self.estimate_enemy_pos()

    def estimate_enemy_pos(self):
        enemy_poses = []
        points = np.copy(self.points.transpose())
        for p in self.enemy_pos:
            p_enemy = np.asarray(p)
            delta = points - p_enemy
            distance = np.sqrt(delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1])
            enemy_points = points[distance < self.dist_th]
            # print enemy_points
            if enemy_points.shape[0] != 0:
                enemy_pose = np.mean(enemy_points, axis=0)
                enemy_poses.append([enemy_pose[0], enemy_pose[1]])
                self.lidar_est = [enemy_pose[0], enemy_pose[1]]
        return enemy_poses

    def update_enemy(self):
        try:
            (trans, rot) = self.enemy_tf_listener_.lookupTransform("map", "armor0", rospy.Time(0))
            self.enemy_pos = [[trans[0], trans[1]]]
            # print(self.enemy_pos)
            # return self.estimate_enemy_pos()
            # print self.enemy_pos
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass


def update_thread(lidar):
    while not rospy.is_shutdown():
        lidar.update_enemy()
        p = lidar.estimate_enemy_pos()
        print "laser est: ", p, " camera est: ", lidar.enemy_pos
        time.sleep(0.05)


if __name__=="__main__":
    rospy.init_node("laser_pose")
    map_file = "/home/drl/ros_codes/RoboRTS/tools/map/icra.pgm"
    map_im = cv2.imread(map_file)
    lidar = lidar_scan()
    thread.start_new_thread(update_thread, (lidar,))
    while not rospy.is_shutdown():
        if lidar.points.shape[0] != 0 and len(lidar.enemy_pos) != 0 and len(lidar.lidar_est)!=0:
            plt.plot(lidar.points[0], lidar.points[1], '.')
            plt.plot(lidar.enemy_pos[0][0], lidar.enemy_pos[0][1], '*', markersize = 20)
            plt.plot(lidar.lidar_est[0], lidar.lidar_est[1], '*', markersize=15)
            plt.imshow(map_im, extent=[-1.2, 6.8, -0.8, 4.2])
            plt.draw()
            plt.pause(0.01)
            plt.clf()
            # time.sleep(0.1)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "shutting down"
