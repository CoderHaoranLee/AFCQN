#!/usr/bin/env python
from detection_main import *
from light_detection import *
import rospy
from messages.msg import EnemyPos
from sensor_msgs.msg import Image
from messages.msg import RoboState
import time
from cv_bridge import CvBridge, CvBridgeError

class GimbalControl(object):

    def __init__(self):
        self.cmd_pitch = 0.0
        self.cmd_yaw = 0.0
        self.cmd_pub_ = rospy.Publisher('/enemy_pos', EnemyPos, queue_size=1)
        self.img_sub_ = rospy.Subscriber('/camera_0', Image, self.detection_callback)
        self.gimbal_state_sub_ = rospy.Subscriber('/robo_state', RoboState, self.update_gimbal_state)

        self.bridge = CvBridge()

        # self.model_path = "/home/drl/catkin_ws/src/messages/Model2/model.ckpt"
        # self.keep_prob = tflow.placeholder(tflow.float32)
        # self.pred = cnn_model.alex_net(self.x, self.weights, self.biases, self.keep_prob)
        # self.init = tflow.initialize_all_variables()
        # self.saver = tflow.train.Saver()
        # self.sess = tflow.Session()
        # self.saver.restore(self.sess, self.model_path)
        # self.graph = tflow.get_default_graph()
        # self.prediction = self.graph.get_tensor_by_name('add_2:0')
        # self.y_result = tflow.nn.softmax(self.prediction)

    def detection_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            print e

        start = int(time.time() * 1000)
        result = detection_main(cv_image)
        end = int(time.time() * 1000)
        cost_time = float(end - start) / 1000.0
        print "cost time:", cost_time
        if len(result) != 0:
            im1 = result[0]
            armors = result[1]
            center_points = []
            for armor in armors:
                distant = Distance()
                # print "armor_points:", armor.points
                distant.get_distance(armor.points)
                center_points.append(distant)

    def update_gimbal_state(self, data):
        pass

if __name__ == "__main__":
    rospy.init_node("detection_and_control")
    gimbal_control = GimbalControl()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "shutting down"