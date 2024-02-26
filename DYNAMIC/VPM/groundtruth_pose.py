#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import LinkStates, ModelStates

# /gazebo/link_states
# /gazebo/model_states

rospy.init_node('groundtruth_pose')

# pose_pub = rospy.Publisher('groundtruth', PoseStamped, queue_size=1)
# uncomment to loop groundtruth pose to vision pose input:
pose_pub = rospy.Publisher('mavros/vision_pose/pose', PoseStamped, queue_size=1)

pose_msg = PoseStamped()
pose_msg.header.frame_id = 'map'

def link_states_callback(msg):
    try:
        index = msg.name.index('clover::base_link')
    except ValueError:
        return
    pose_msg.pose = msg.pose[index]
    pose_msg.header.stamp = rospy.Time.now()
    pose_pub.publish(pose_msg)

rospy.Subscriber('gazebo/link_states', LinkStates, link_states_callback)

rospy.spin()
