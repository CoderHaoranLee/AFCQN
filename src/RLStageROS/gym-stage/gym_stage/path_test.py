import rospy
from nav_msgs.msg import Path

def print_path(data):
    print len(data.poses)

if __name__=="__main__":
    rospy.init_node("path_test")
    sub_path = rospy.Subscriber("/global_planner_path", Path, print_path)
    rospy.spin()