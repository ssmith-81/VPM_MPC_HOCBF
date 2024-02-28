#!/usr/bin/python3

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
from gazebo_msgs.srv import SetModelState
import tf
from std_msgs.msg import String
from sensor_msgs.msg import Imu, LaserScan
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse

# Custom message for object state publishing
from custom_message.msg import ObjectPub

#from panel_functions import CLOVER_COMPONENTS, CLOVER_STREAM_GEOMETRIC_INTEGRAL, CLOVER_KUTTA, CLOVER_STREAMLINE, CLOVER_noOBSTACLE

from scipy.interpolate import griddata
from scipy.optimize import curve_fit, least_squares

import h5py  # use for data logging of larges sets of data
import os

from tf.transformations import euler_from_quaternion

import numpy as np

# Could plot the stored data in SITL (not hardware) if desired:
import matplotlib.pyplot as plt
from matplotlib import path

from time import sleep
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

rospy.init_node('AHOSMO_node') # Figure8 is the name of the ROS node

# Set the use_sim_time parameter to true
rospy.set_param('use_sim_time', True)

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
xf = []  # This gathers the x-lidar points
yf = []
# Variables for lines l_1, l_2, l_3, lidar_line_1, lidar_line_2
A1 = []
B1 = []
LidarA1 = []
LidarB1 = []
A2 = []
B2 = []
LidarA2 = []
LidarB2 = []
A3 = []
B3 = []

# Intersection points and estimated center
mu1x = []
mu1y = []
mu2x = []
mu2y = []
zhatx = []
zhaty = []

xdispf=[]
ydispf=[]
xa = []
ya = []
YawL = []
YawF = []
YawC = []

# Normalized omega center estimation
znormx = []
znormy = []

# sci ellipse
cenx = []
ceny = []
a_elipse = []
b_elipse = []
beta = []

# Circle obstacle plot
xa = []
ya = []

# updated velocity field plot
u_field = []
v_field = []

# Analyze observer for obstacle 1
x_obs=[]
vx_obs=[]
ax_obs=[]

y_obs=[]
vy_obs=[]
ay_obs=[]

# actual obstacle pose
x_obsr=[]
vx_obsr=[]
ax_obsr=[]

y_obsr=[]
vy_obsr=[]
ay_obsr=[]

# Time for obs1
time_obs1 = []

# Analyze observer for obstacle 2
x_obs2=[]
vx_obs2=[]
ax_obs2=[]

y_obs2=[]
vy_obs2=[]
ay_obs2=[]

# actual obstacle pose
x_obsr2=[]
vx_obsr2=[]
ax_obsr2=[]

y_obsr2=[]
vy_obsr2=[]
ay_obsr2=[]

# Time for obs1
time_obs2 = []

# Log the time for the position control variables
time_now = []

# Get the path to the directory containing the Python script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Create a HDF5 file name
# Open the HDF5 file globally
file_name = 'AHOSMO.h5'

# Construct the absolute path to the HDF5 file
absolute_file_path = os.path.join(script_dir, file_name)


 # Open the HDF5 file for writing
with h5py.File(absolute_file_path, 'a') as hf:

			
	# This class categorizes all of the functions used for complex rajectory tracking
	class clover:

		def __init__(self, FLIGHT_ALTITUDE, RATE, RADIUS, CYCLE_S, REF_FRAME): 


			# -------------------------------------------------------------------------------------------------------------------------------

			# global static variables
			self.FLIGHT_ALTITUDE = FLIGHT_ALTITUDE          # fgdf
			self.RATE            = RATE                     # loop rate hz
			self.FRAME           = REF_FRAME                # Reference frame for complex trajectory tracking
			
			self.last_timestamp = None # make sure there is a timestamp

			# Subscribe directly to the ground truth
			self.ground_truth = rospy.Subscriber('gazebo/link_states', LinkStates, self.link_states_callback)
			
			# Generate the array of lidar angles
			self.lidar_angles = np.linspace(-180*(math.pi/180), 180*(math.pi/180), 360) # Adjust this number based on number defined in XACRO!!!!!

			# Initialize the ranges array for the controller as no object detected (i.e. an array of inf)
			self.obs_detect = np.full_like(self.lidar_angles, np.inf)

			# Subscribe to the Lidar readings
			self.lidar = rospy.Subscriber('/ray_scan',LaserScan,self.lidar_read)

			# Subscribe to the 'Obstacle_state' topic with the message type ObjectPub
			#self.obs_state = rospy.Subscriber('Obstacle_state', ObjectPub, self.obstacle)

			# Create two subscribers for /Obstacle_state1 and /Obstacle_state2 topics
			self.obs_state1 = rospy.Subscriber('/Obstacle_state1', ObjectPub, self.obstacle1)
			self.obs_state2 = rospy.Subscriber('/Obstacle_state2', ObjectPub, self.obstacle2)
			
			# Create a topic to publish the estimated obstacle dynamics on, from the observer
			# Create a publisher for the obstacle dynamics topic
			self.obstacle_dynamics_pub = rospy.Publisher('/AHOSMO_est', ObjectPub, queue_size=10)

			# Create a publisher object
			self.AHOSMO = ObjectPub()

			# This flag will be used to determine which obstacle is being detected and what pose to act on fr estimation for MPC solver ACADOS
			self.obstacle_counter = 0 

			# Set a flag, that will be used as a logic operator to adjust how the HOCBF is set, which depends on whether an obstacle is detected or not
			self.object_detect = False


			# Set a flag, that will track if a change in envirnment occurred i.e. an object was detected
			# therefore if an object was not detected previously then there is no reason in re calculating
			# the velocity field based on source/sink strengths
			self.flag = False

			# Define the observer variables
			self.x1hat_cur = 0
			self.x2hat_cur = 0
			self.x3hat_cur = 0

			self.y1hat_cur = 0
			self.y2hat_cur = 0
			self.y3hat_cur = 0


			
			

			self.current_state = State()
			self.rate = rospy.Rate(20)

		def link_states_callback(self, msg):
			try:
				index = msg.name.index('clover::base_link')
			except ValueError:

				return

			self.clover_pose = msg.pose[index]

		def obstacle1(self,data):

			self.obs_x1 = data.x
			self.obs_y1 = data.y
			self.obs_vx1 = data.vx
			self.obs_vy1 = data.vy
			self.obs_ax1 = data.ax
			self.obs_ay1 = data.ay
			# print(data.x)
		
		def obstacle2(self,data):

			self.obs_x2 = data.x
			self.obs_y2 = data.y
			self.obs_vx2 = data.vx
			self.obs_vy2 = data.vy
			self.obs_ax2 = data.ax
			self.obs_ay2 = data.ay

		
		def lidar_read(self,data):


			# Update the obstacle detect array
			self.obs_detect = data.ranges

			# Get current state of this follower 
            # telem = get_telemetry(frame_id='map')
			x_clover = self.clover_pose.position.x
			y_clover = self.clover_pose.position.y
			z_clover = self.clover_pose.position.z
			quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
			euler_angles = euler_from_quaternion(quaternion)
			roll = euler_angles[2] #+ math.pi
			yaw = -euler_angles[0]+math.pi 
			pitch = euler_angles[1]

			# Ensure there are actually lidar readings, no point in doing calculations if
				# nothing is detected:
				# Need to add a lidar detection threshold for panel method case (4), so if we have like 1-2 or low detection we could get singular matrix
			if sum(not np.isinf(range_val) for range_val in self.obs_detect) >= 4 and z_clover >= 0.65: # want the drone to be at some altitude so we are not registering ground detections
		
				# Going to need to subscribe to the set_obstacle_state node in this function to extract
				# obstacle position (as the ellipse estimation will most likely be too computationally heavy :/)

				# The angles and ranges start at -45 degrees i.e. at the right side, then go counter clockwise up to the top i.e. 45 degrees
				ranges = data.ranges
				
			
				angles = self.lidar_angles
			
				# Get current state of this follower 
				# telem = get_telemetry(frame_id='map')
				x_clover = self.clover_pose.position.x
				y_clover = self.clover_pose.position.y
				z_clover = self.clover_pose.position.z
				quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
				euler_angles = euler_from_quaternion(quaternion)
				roll = euler_angles[2] #+ math.pi
				yaw = -euler_angles[0]+math.pi 
				pitch = euler_angles[1]
				
				# This transforms to the local Lidar frame where the y-axis is forward
				# and the x-axis is pointing right (the same reference frame used in the RM tracker paper, so this is the one we will use here):
				#x_local2 = ranges*np.sin(angles)
				#x_local2 = np.multiply(x_local2,-1)
				#y_local2 = ranges*np.cos(angles)
				
				# Polar to Cartesion transformation for all readings (assuming angles are in standard polar coordinates) y-axis is left and x-axis is directly forward.
				x_local2 = ranges*np.cos(angles)
				y_local2 = ranges*np.sin(angles)
				
				
				# Filter out the inf values in the data point arrays
				valid_indices = np.isfinite(x_local2) & np.isfinite(y_local2)

				
				# Filter x_local2, y_local2, and angles based on valid_indices
				self.x_local2_filtered = x_local2[valid_indices]
				self.y_local2_filtered = y_local2[valid_indices]
				angles_filtered = angles[valid_indices]
				ranges_np = np.array(ranges)
				ranges_filtered = ranges_np[valid_indices]
				
	#-------------------Ellipse estimation section-------------------------------------------------------------------
				# # The least squares at each iteration is too computationally intensive for this VM system, it sucks up to much CPU 
				# # percentage. Therefore we wont calculate here, we will log the lidar readings and calculate all of the ellipse estimations
				# # after the simulation has run for plotting (when using on hardware, we wont have gazebo etc running so there should be enough resources)

				# if self.ellipse_first:
				# 	# if an obstacle is aleady detected, do nothing
				# 	pass
				# else:

				# 	 # If no obstacle is currently detected, perform the calculation
				# 	# This calculation will only be performed once per new obstacle

				# 	# Ellipse model-------------------------
				# 	xy = np.column_stack((self.xa[1::2],self.ya[1::2]))
				# 	ellipse_model = EllipseModel()
				# 	ellipse_model.estimate(xy)

				# 	# Extract parameters of the fitted ellipse
				# 	self.xc = ellipse_model.params[0]
				# 	self.yc = ellipse_model.params[1]
				# 	self.a_fit = ellipse_model.params[2]
				# 	self.b_fit = ellipse_model.params[3]
				# 	self.theta = ellipse_model.params[4]

				# 	self.ellipse_first = True

				# 	# print(self.xc)

				# 	#------compare--------


				# 	# reg = LsqEllipse().fit(xy)

				# 	# center_t, width, height, phi = reg.as_parameters()

				# 	# print(center_t[0])

				# 	#--------------------

				# 	xc.append(self.xc)
				# 	yc.append(self.yc)
	#-----------------------------------------------------------------------------------------------------------------
				# # Append row after row of data (to log readings)
				# xa.append(self.xa.tolist())
				# ya.append(self.ya.tolist())

				# # Log the Clover position and velocity at the same time we are logging this lidar data reading
				# # do this so we can calculate psi_0 and psi_1 based on the ellipse readings. Hopefully will calculate this
				# # in real time when we get to hardware in the lab
				# telem = get_telemetry(frame_id='map')
				# x_ce.append(self.clover_pose.position.x)
				# vx_ce.append(telem.vx)
				# y_ce.append(self.clover_pose.position.y)
				# vy_ce.append(telem.vy)
					

		

				
				xf.append(self.x_local2_filtered)
				yf.append(self.y_local2_filtered)
				
				# Homogeneous transformation matrix for 2D (adjust global yaw by PI/2 because local Lidar frame is rotated 90 degrees relative to it):
				R_2 = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])  # rotation matrix
				
				# Combine rotation and translation into a single homogeneous transformation matrix
				T_2 = np.vstack([np.hstack([R_2, np.array([[x_clover], [y_clover]])]), [0, 0, 1]])  # Homogeneous transformation matrix
				
				# Lidar readings in homogenous coordinates
				readings_local2 = np.vstack([x_local2, y_local2, np.ones_like(x_local2)])
				
				# Transform all lidar readings to global coordinates
				readings_global2 = np.dot(T_2, readings_local2)
				
				# Extract the tranformed positions
				readings_global2 = readings_global2[:2,:].T

				# Retrieve the current timestamp in seconds
				current_timestamp = data.header.stamp.to_sec()

				# Subscribe to the obstacle state (position published from the set obstacle node)...

				if not self.object_detect:
					self.obstacle_counter +=1 # this only increments when an obstacle was detected once
												# so it will be equal to 1 for the cylinder and equal to 2 for the prism
					self.object_detect = True # Update object detected flag

				# Calculate the obstacles position, velocity, and acceleration in the drones reference frame using the AHOSMO
				# An obstacle was detected, use the obstacle_counter number
				# to determine if it was the cylinder or the prism, and send on 
				# the values accordingly
				if self.obstacle_counter == 1:
					# This is the first moving cylinder, which has a radius of (1.5m??)
					center = np.array([self.obs_x1,self.obs_y1])
				else:
					# This is the second moving cylinder, which has a radius of (1.5m??)
					center = np.array([self.obs_x2,self.obs_y2])

				if self.last_timestamp is not None:

					# Calculate the change in time since the last time step (this is needed
					# for euler integration within this funcion)
					dt = current_timestamp - self.last_timestamp

					# Temporary constant gains
					L1 = 15
					L2 = 25
					L3 = 15

					# Update the observer for the x-dynamic direction
					x1hat = self.x1hat_cur + dt*(self.x2hat_cur + L1*(abs(center[0] - self.x1hat_cur)**(2/3))*np.sign(center[0]-self.x1hat_cur))
					x2hat = self.x2hat_cur + dt*(self.x3hat_cur + L2*(abs(center[0] - self.x1hat_cur)**(1/3))*np.sign(center[0]-self.x1hat_cur))
					x3hat = self.x3hat_cur + dt*(L3*np.sign(center[0]-self.x1hat_cur))

					# Update the observer for the x-dynamic direction
					y1hat = self.y1hat_cur + dt*(self.y2hat_cur + L1*(abs(center[1] - self.y1hat_cur)**(2/3))*np.sign(center[1]-self.y1hat_cur))
					y2hat = self.y2hat_cur + dt*(self.y3hat_cur + L2*(abs(center[1] - self.y1hat_cur)**(1/3))*np.sign(center[1]-self.y1hat_cur))
					y3hat = self.y3hat_cur + dt*(L3*np.sign(center[1]-self.y1hat_cur))

					if self.obstacle_counter == 1:
					# This is the first moving cylinder, which has a radius of (1.5m??)
						x_obs.append(x1hat)
						vx_obs.append(x2hat)
						ax_obs.append(x3hat)
						y_obs.append(y1hat)
						vy_obs.append(y2hat)
						ay_obs.append(y3hat)
						 # Get the actual time:
						current_time = rospy.Time.now()
						x_obsr.append(self.obs_x1)
						vx_obsr.append(self.obs_vx1)
						ax_obsr.append(self.obs_ax1)
						y_obsr.append(self.obs_y1)
						vy_obsr.append(self.obs_vy1)
						ay_obsr.append(self.obs_ay1)
						time_obs1.append(current_time.to_sec())

						# Publish state estimation to the main node file for use in the MPC
						self.AHOSMO.x = self.obs_x1 # send in the actual position values to the main script, not the estimation
						self.AHOSMO.y = self.obs_y1 
						self.AHOSMO.vx = x2hat
						self.AHOSMO.vy = y2hat
						self.AHOSMO.ax = x3hat
						self.AHOSMO.ay = y3hat
					
					else:
					# This is the second moving cylinder, which has a radius of (1.5m??)
						x_obs2.append(x1hat)
						vx_obs2.append(x2hat)
						ax_obs2.append(x3hat)
						y_obs2.append(y1hat)
						vy_obs2.append(y2hat)
						ay_obs2.append(y3hat)
						 # Get the actual time:
						current_time = rospy.Time.now()
						x_obsr2.append(self.obs_x2)
						vx_obsr2.append(self.obs_vx2)
						ax_obsr2.append(self.obs_ax2)
						y_obsr2.append(self.obs_y2)
						vy_obsr2.append(self.obs_vy2)
						ay_obsr2.append(self.obs_ay2)
						time_obs2.append(current_time.to_sec())

						# Publish state estimation to the main node file for use in the MPC
						self.AHOSMO.x = self.obs_x2 # send in the actual position values to the main script, not the estimation
						self.AHOSMO.y = self.obs_y2 
						self.AHOSMO.vx = x2hat
						self.AHOSMO.vy = y2hat
						self.AHOSMO.ax = x3hat
						self.AHOSMO.ay = y3hat
					

				

					self.obstacle_dynamics_pub.publish(self.AHOSMO)

					# Update current values
					self.x1hat_cur = x1hat
					self.x2hat_cur = x2hat
					self.x3hat_cur = x3hat

					self.y1hat_cur = y1hat
					self.y2hat_cur = y2hat
					self.y3hat_cur = y3hat

					

				# update last_timestamp to current timestamp
				self.last_timestamp = current_timestamp
			else:
				self.object_detect = False # Update object detected flag

				# Re-anitialize the observer calculations to zero when nothing is calculated
				self.x1hat_cur = 0.0
				self.x2hat_cur = 0.0
				self.x3hat_cur = 0.0

				self.y1hat_cur = 0.0
				self.y2hat_cur = 0.0
				self.y3hat_cur = 0.0

				# When nothing is being detected, publish zeros for the obstacle state
				# Publish state estimation to the main node file for use in the MPC
				self.AHOSMO.x = self.x1hat_cur # send in the actual position values to the main script, not the estimation
				self.AHOSMO.y = self.y1hat_cur
				self.AHOSMO.vx = self.x2hat_cur
				self.AHOSMO.vy = self.y2hat_cur
				self.AHOSMO.ax = self.x3hat_cur
				self.AHOSMO.ay = self.y3hat_cur

				self.obstacle_dynamics_pub.publish(self.AHOSMO)
				
				
				
				

		


		def main(self):
		
			
			
			while not rospy.is_shutdown():

					
				
				
				
				# Get current state of this follower 
				#telem = get_telemetry(frame_id='map')
				
				# logging/debugging
				
				#evx.append(self.u-telem.vx)
				#evy.append(self.v-telem.vy)
				#eyaw.append(self.omega-telem.yaw)
				#YawC.append(self.omega*(180/math.pi))
				#YawF.append(telem.yaw*(180/math.pi))
				#xdispf.append(self.d_xi)
				#ydispf.append(self.d_yi)
				#velfx.append(telem.vx)
				#velfy.append(telem.vy)
				#velcx.append(self.u)
				#velcy.append(self.v)
				# U_infx.append(self.U_inf)
				# V_infy.append(self.V_inf)
				
				rospy.spin()
				
				

	if __name__ == '__main__':
		try:
			# Define the performance parameters here which starts the script
			q=clover(FLIGHT_ALTITUDE = 1.0, RATE = 50, RADIUS = 3.3, CYCLE_S = 25, REF_FRAME = 'map')
			
			q.main()

			#-------------External LOG------------------
            # Create a group to store velocity field for this iteration/time
			iteration_group = hf.create_group('observe_obs_1')
			iteration_group.create_dataset('x_obs', data=x_obs)
			iteration_group.create_dataset('y_obs', data=y_obs)
			iteration_group.create_dataset('vx_obs', data=vx_obs)
			iteration_group.create_dataset('vy_obs', data=vy_obs)
			iteration_group.create_dataset('ax_obs', data=ax_obs)
			iteration_group.create_dataset('ay_obs', data=ay_obs)

			iteration_group.create_dataset('xc', data=x_obsr)
			iteration_group.create_dataset('yc', data=y_obsr)
			iteration_group.create_dataset('vxc', data=vx_obsr)
			iteration_group.create_dataset('vyc', data=vy_obsr)
			iteration_group.create_dataset('axc', data=ax_obsr)
			iteration_group.create_dataset('ayc', data=ay_obsr)
			
			#time_now = np.array(time_now)
			iteration_group.create_dataset('time_now', data=time_obs1)

			# Create a group to store velocity field for this iteration/time
			group2 = hf.create_group('observe_obs_2')
			group2.create_dataset('x_obs', data=x_obs2)
			group2.create_dataset('y_obs', data=y_obs2)
			group2.create_dataset('vx_obs', data=vx_obs2)
			group2.create_dataset('vy_obs', data=vy_obs2)
			group2.create_dataset('ax_obs', data=ax_obs2)
			group2.create_dataset('ay_obs', data=ay_obs2)

			group2.create_dataset('xc', data=x_obsr2)
			group2.create_dataset('yc', data=y_obsr2)
			group2.create_dataset('vxc', data=vx_obsr2)
			group2.create_dataset('vyc', data=vy_obsr2)
			group2.create_dataset('axc', data=ax_obsr2)
			group2.create_dataset('ayc', data=ay_obsr2)
			
			#time_now = np.array(time_now)
			group2.create_dataset('time_now', data=time_obs2)
                #------------------------------------------


			# Debug section, need matplotlib to plot the results for SITL
			plt.figure(10)
			plt.subplot(311)
			plt.plot(time_obs1,x_obs,'r',label='x-pos-est')
			plt.plot(time_obs1,x_obsr,'b--',label='x-pos')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Position [m]')
			plt.subplot(312)
			plt.plot(time_obs1,vx_obs,'r',label='x-vel-est')
			plt.plot(time_obs1,vx_obsr,'b--',label='x-vel')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Velocity [m/s]')
			plt.subplot(313)
			plt.plot(time_obs1,ax_obs,'r',label='ax-est')
			plt.plot(time_obs1,ax_obsr,'b--',label='ax-real')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Position [m]')
			plt.xlabel('Time [s]')
			
			
			# # observer plot
			# plt.figure(2)
			# plt.subplot(211)
			# plt.plot(cenx,'r',label='obs_center')
			# plt.plot(x_obs,'b',linewidth = 2,linestyle='--',label='obs_hat')
			# plt.ylabel('pos[m]')
			# plt.xlabel('Time [s]')
			# plt.legend()
			# plt.grid(True)
			# plt.subplot(212)
			# plt.plot(ceny,'r',label='obs_center')
			# plt.plot(y_obs,'b',linewidth = 2,linestyle='--',label='obs_hat')
			# plt.ylabel('pos[m]')
			# plt.xlabel('Time [s]')
			# plt.legend()
			# plt.grid(True)
			# # plt.plot(velfy,'r',label='vy-vel')
			# # plt.plot(velcy,'b--',label='vy-com')
			# # plt.legend()
			# # plt.grid(True)
			# # plt.ylabel('Position [m]')
			# # plt.subplot(313)
			# # plt.plot(evx,'r',label='evx')
			# # plt.plot(evy,'b',label='evy')
			# # plt.plot(eyaw,'g',label='eyaw')
			# # plt.ylabel('Error[m]')
			# # plt.xlabel('Time [s]')
			# # plt.legend()
			# plt.figure(3)
			# plt.subplot(311)
			# plt.plot(cenx,'r',label='obs_center-x')
			# plt.plot(x_obs,'b',linewidth = 2,linestyle='--',label='obs_hat-x')
			# plt.plot(x_obsr,'g',linewidth = 2,linestyle='--',label='real_obs-center-x')
			# plt.ylabel('pos[m]')
			# plt.legend()
			# plt.grid(True)
			# plt.subplot(312)
			# plt.plot(vx_obs,'r',label='obs_vel-hat')
			# plt.plot(vx_obsr,'b',linewidth = 2,linestyle='--',label='obs_vel-real')
			# plt.ylabel('vel [m/s]')
			# plt.legend()
			# plt.grid(True)
			# plt.subplot(313)
			# plt.plot(ax_obs,'r',label='obs_ax-hat')
			# plt.plot(ax_obsr,'b',linewidth = 2,linestyle='--',label='obs_ax-real')
			# plt.ylabel('pos [m]')
			# plt.xlabel('Time [s]')
			# plt.legend()
			# plt.grid(True)

			# plt.figure(3)
			# for x_row, y_row in zip(xa, ya):
			# 	plt.plot(x_row,y_row, '-o',label=f'Reading {len(plt.gca().lines)}')
			# 	#plt.fill(xa,ya,'k')
			# plt.grid(True)
			# plt.legend()
			
			# plt.figure(3)
			# plt.plot(U_infx,'r',label='x_inf')
			# plt.plot(V_infy,'b',label='y_inf')
			# #plt.plot(velcx,'g',label='velcx')
			# #plt.plot(velcy,'m',label='velcy')
			# #plt.plot(velLx,'k',label='velLx')
			# #plt.plot(velLy,'y',label='velLy')
			# plt.ylabel('SMC Commands [m/s]')
			# plt.xlabel('Time [s]')
			# plt.legend()
			# plt.grid(True)
			
			#plt.figure(4)
			#plt.subplot(211)
			#plt.plot(ex,'r',label='ex-fol')
			#plt.plot(ey,'b--',label='ey-lead')
			#plt.plot(
			
			#plt.subplot(212)
			#plt.plot(YawL,'b--',label='yaw')
			
			plt.show()
			
		except rospy.ROSInterruptException:
			pass
