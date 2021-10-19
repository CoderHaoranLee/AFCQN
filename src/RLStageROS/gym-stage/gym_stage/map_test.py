from nav_msgs.msg import OccupancyGrid
import rospy
import numpy as np
import matplotlib.pyplot as plt
import cv2

class MapTest():
    def __init__(self):
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.last_map = []
        self.true_map = cv2.imread("/home/drl/ros_codes/catkin_ws/src/DQNStageROS/world/robocup.png")
        self.true_map = np.asarray(self.true_map)
        self.true_H, self.true_W, _ = self.true_map.shape
        self.true_half_H, self.true_half_W = int(self.true_H/2.0), int(self.true_W/2.0)
        print self.true_map.shape

    def map_callback(self, data):
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
        map_data[idx_less0] = 128
        map_data[idx_greater0] = 255

        map_data = map_data.reshape(H, W)
        map_data = np.flipud(map_data)

        copy_map_data = np.ones((self.true_H, self.true_W)) * 128

        copy_map_data[max(0, self.true_half_H-map_pixel_y):min(self.true_H, self.true_half_H+(H-map_pixel_y)),
        max(0, self.true_half_W - map_pixel_x):min(self.true_W, self.true_half_W + (W - map_pixel_x))] = map_data[max(0, map_pixel_y-self.true_half_H):min(H, map_pixel_y+self.true_half_H),
                                                                                                         max(0,map_pixel_x - self.true_half_W):min(W,map_pixel_x + self.true_half_W)]
        plt.subplot(1, 2, 1)
        plt.imshow(self.true_map)
        plt.subplot(1, 2, 2)
        plt.imshow(copy_map_data)
        plt.draw()
        plt.pause(0.1)
        plt.clf()


if __name__=="__main__":
    rospy.init_node("map_test")
    t = MapTest()
    rospy.spin()
