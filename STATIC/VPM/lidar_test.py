#!/usr/bin/python

# * Simplified complex trajectory tracking in OFFBOARD mode
# *
# * Author: Sean Smith <s.smith@dal.ca>
# *
# * Distributed under MIT License (available at https://opensource.org/licenses/MIT).
# * The above copyright notice and this permission notice shall be included in all
# * copies or substantial portions of the Software.


import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from gazebo_msgs.msg import ModelState, LinkStates
import tf
from std_msgs.msg import String
from sensor_msgs.msg import Imu, LaserScan

from panel_functions import CLOVER_COMPONENTS, CLOVER_STREAM_GEOMETRIC_INTEGRAL, CLOVER_KUTTA, CLOVER_STREAMLINE

#from scipy.interpolate import griddata

from tf.transformations import euler_from_quaternion
from scipy.spatial.transform import Rotation  #[w,x,y,z]

import numpy as np

import time

# Could plot the stored data in SITL (not hardware) if desired:
import matplotlib.pyplot as plt
from matplotlib import path

from time import sleep
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

rospy.init_node('clover_panel') # Figure8 is the name of the ROS node

# Define the Clover service functions, the only ones used in this application are navigate and land.

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

# Release service is used to allow for complex trajectory publishing i.e it stops the navigate service from publishing setpoints because you dont want two sources of publishing at the same time.
release = rospy.ServiceProxy('simple_offboard/release', Trigger)

# Deine math parameter
PI_2 = math.pi/2

# Debugging and logging
xf = []  # This gathers the 
yf = []
xdispf=[]
ydispf=[]
xa = []
ya = []
YawL = []
YawF = []
YawC = []

# Check local/before global transformation
xloc = []
yloc = []
xloc_orig = []
yloc_orig = []

# 3D transform logging
x3 = []
y3 = []

# Unmodified measurements
x_orig = []
y_orig = []

# Analyze control input (see if error is being minimized )
velfx=[]
velfy=[]
velcx=[]
velcy=[]
U_infx = []
V_infy=[]
evx=[]
evy=[]
eyaw=[]


# Initialize variables to keep track of time
start = time.time()
log_interval = 2  # Log data every 5 seconds		

		
lidar_angles = np.linspace(-180*(math.pi/180), 180*(math.pi/180), 360) # generate the array of lidar angles
# lidar_angles = np.arange(-3.141590118408203, 3.141590118408203, 0.017501894384622574)

def link_states_callback(msg):
	global clover_pose
	try:
		index = msg.name.index('clover::base_link')
	except ValueError:
		return

	clover_pose = msg.pose[index]


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to rotation matrix.
    """
    # Convert degrees to radians
    # roll = math.radians(roll)
    # pitch = math.radians(pitch)
    # yaw = math.radians(yaw)
    
    # Compute rotation matrices
    R_roll = np.array([[1, 0, 0],
                       [0, math.cos(roll), -math.sin(roll)],
                       [0, math.sin(roll), math.cos(roll)]])
    
    R_pitch = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
    
    R_yaw = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                      [math.sin(yaw), math.cos(yaw), 0],
                      [0, 0, 1]])
    
    # Combine rotation matrices
    R = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    
    return R

		
		
def lidar_read(data):

	global start  # Declare start_time as global
	
	
		# The angles and ranges start at -45 degrees i.e. at the right side, then go counter clockwise up to the top i.e. 45 degrees
	ranges = data.ranges
	

	# Ensure there are actually lidar readings, no point in doing calculations if
		# nothing is detected:
	if any(not np.isinf(range_val) for range_val in ranges):
	
		angles = lidar_angles
		print(ranges)
		print(angles)
		
		# telem = get_telemetry(frame_id='map')
			
		x_clover = clover_pose.position.x
		y_clover = clover_pose.position.y
		z_clover = clover_pose.position.z
		quaternion = [clover_pose.orientation.w,clover_pose.orientation.x, clover_pose.orientation.y, clover_pose.orientation.z ]
		euler_angles = euler_from_quaternion(quaternion)
		roll = euler_angles[2] 
		yaw = -euler_angles[0]+math.pi #- PI_2
		pitch = euler_angles[1]

		r = Rotation.from_quat(quaternion)

		roll_pitch_yaw = r.as_euler('xyz')#, degrees=True)
		
		yaw_s = roll_pitch_yaw[0]
		pitch_s = roll_pitch_yaw[1]
		yaw_s = roll_pitch_yaw[2]

		print(euler_angles)
		print(roll_pitch_yaw)

		# Convert ranges to a NumPy array if it's not already
		ranges = np.array(ranges)
		ranges_orig = np.array(ranges) # store the unmodified ranges
			
			# Polar to Cartesion transformation for all readings (assuming angles are in standard polar coordinates).
		x_local = ranges*np.cos(angles)
		y_local = ranges*np.sin(angles)

		# This transforms to the local Lidar frame where the y-axis is forward
			# and the x-axis is pointing right:
		# x_local = ranges*np.sin(angles)
		# x_local = np.multiply(x_local,-1)
		# y_local = ranges*np.cos(angles)

		x_local_orig = ranges_orig*np.cos(angles)
		y_local_orig = ranges_orig*np.sin(angles)
	#---------------- Safety factor-----------------------------------------------
		# # put a safety factor on the detected obstacle
		# 		# Reduce the range by a scaling factor beta for each real range (set as diameter of the clover)
		# beta = 1.3 # Scale object and shift
		# # Combine xdata and ydata into a single array of points
		# points = np.column_stack((x_local, y_local))

		# # Find the point closest to the origin
		# min_distance_index = np.argmin(np.linalg.norm(points, axis=1))
		# closest_point = points[min_distance_index]

		# # Step 2: Shift all points so that the closest point becomes the origin
		# shifted_points = points - closest_point

		# # Step 3: Scale all points by a factor beta
		# scaled_points = shifted_points * beta

		# # Step 4: Shift all points back to their original positions
		# final_points = scaled_points + closest_point

		# # Calculate the distance to move the closest point
		# desired_distance = 0.5

		# # Calculate the current distance to the origin for the closest point
		# current_distance = np.linalg.norm(closest_point)

		# # Calculate the unit vector in the direction of the closest point
		# unit_vector = closest_point / current_distance

		# # Calculate the new position for the closest point
		# new_closest_point = unit_vector * (current_distance - desired_distance)

		# # Calculate the difference vector
		# shift_vector = closest_point - new_closest_point

		# # Shift all points including the closest point
		# shifted_points = final_points - shift_vector


		# # translate the shape equally to the origin (To clover)
		# x_local = shifted_points[:,0]
		# y_local = shifted_points[:,1]
		

	#---------------------------------------------------------------------------------
			# Homogenous transformation matrix for 2D
		R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) # rotation matrix
		T = np.vstack([np.hstack([R, np.array([[x_clover], [y_clover]])]),[0,0,1]]) # Homogenous transformation matrix


		# Lidar readings in homogenous coordinates
		readings_local = np.vstack([x_local, y_local, np.ones_like(x_local)])
		readings_local_orig = np.vstack([x_local_orig, y_local_orig, np.ones_like(x_local_orig)])
			
			
			# Transform all lidar readings to global coordinates
		readings_global = np.dot(T, readings_local)
		readings_global_orig = np.dot(T, readings_local_orig)
			
			# Extract the tranformed positions
		readings_global = readings_global[:2,:].T
		readings_global_orig = readings_global_orig[:2,:].T

		#print(readings_global)
		xa = readings_global[:,0]
		ya = readings_global[:,1]

		xa_orig = readings_global_orig[:,0]
		ya_orig = readings_global_orig[:,1]

		

	#-----------------improve with 3D transofmation-----------------------------------

		# Compute 2D transformation matrix
		# Compute rotation matrix
		R_3D = euler_to_rotation_matrix(roll, pitch, yaw)
		# Homogeneous transformation matrix
		T_3D = np.vstack([np.hstack([R_3D, np.array([[x_clover], [y_clover], [z_clover]])]), [0, 0, 0, 1]])
		# T_3D = np.vstack([np.hstack([R_3D, np.array([[x_clover], [y_clover], [0]])]), [0, 0, 1]])

		# Lidar readings in homogeneous coordinates
		readings_local_3D = np.vstack([x_local, y_local, np.ones_like(x_local), np.ones_like(x_local)])
	# Transform lidar readings to global coordinates
		readings_global_3D = np.dot(T_3D, readings_local_3D)
		
		# Extract the transformed positions
		readings_global_3D = readings_global_3D[:2, :].T

		xa1 = readings_global[:,0]
		ya1 = readings_global[:,1]

		# Check if it's time to log data
		if time.time() - start >= log_interval:
		
			# Append row after row of data (to log readings)
			xf.append(xa.tolist())
			yf.append(ya.tolist())
			xloc.append(x_local.tolist())
			yloc.append(y_local.tolist())
			xloc_orig.append(x_local_orig.tolist())
			yloc_orig.append(y_local_orig.tolist())


			x_orig.append(xa_orig.tolist())
			y_orig.append(ya_orig.tolist())

			# Append row after row of data (to log readings)
			x3.append(xa1.tolist())
			y3.append(ya1.tolist())
			# Reset start time
			start = time.time()

	
    
		
		

def main():

	# Subscribe to the Lidar readings
	lidar = rospy.Subscriber('/ray_scan',LaserScan,lidar_read)

	# Subscribe directly to the ground truth
	ground_truth = rospy.Subscriber('gazebo/link_states', LinkStates, link_states_callback)

	# navigate(x=0.0,y=0.0,z = 0.5, frame_id='map',auto_arm=True) 
	# rospy.sleep(6)
	# navigate(x=10,y=10,z = 0.5,speed=0.7, frame_id='map')
	# rospy.sleep(20)
	# navigate(x=0,y=0,z = 0.5,speed=0.7, frame_id='map')
	# rospy.sleep(15)
	# land()
	
		
		

	rospy.spin()

		
			
			
	

		

if __name__ == '__main__':
	try:
		
		
		main()

		plt.figure(1)
		for x_row, y_row in zip(xf, yf):
			plt.plot(x_row,y_row, '-o',label=f'Reading {len(plt.gca().lines)}')
			#plt.fill(xa,ya,'k')
		plt.grid(True)
		plt.axis('equal')
		for x_row, y_row in zip(x_orig, y_orig):
			plt.plot(x_row,y_row, '-o',label=f'Orig {len(plt.gca().lines)}')
		plt.legend()

		plt.figure(2)
		for x_row, y_row in zip(x3, y3):
			plt.plot(x_row,y_row, '-o',label=f'Reading {len(plt.gca().lines)}')
			#plt.fill(xa,ya,'k')
		plt.grid(True)
		plt.axis('equal')
		plt.legend()

		plt.figure(3)
		for x_row, y_row in zip(xloc, yloc):
			plt.plot(x_row,y_row, '-o',label=f'Reading {len(plt.gca().lines)}')
			#plt.fill(xa,ya,'k')
		for x_row, y_row in zip(xloc_orig, yloc_orig):
			plt.plot(x_row,y_row, '-o',label=f'Orig {len(plt.gca().lines)}')
		plt.grid(True)
		plt.legend()
		plt.axis('equal')
		plt.show()
		
		plt.show()
		
		
	except rospy.ROSInterruptException:
		pass

		
	

