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

from tf.transformations import euler_from_quaternion

import numpy as np

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

# Analyze observer
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
		self.obs_state = rospy.Subscriber('Obstacle_state', ObjectPub, self.obstacle)

		# Set a timer for the velocity field update function (runs periodically)
		# updates every 8 seconds (0.125 Hz)

		#rospy.Timer(rospy.Duration(3), self.velocity_field_update)

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

	def ellipse(self,x,y,a,b):
		# Define the general ellipse function that will be fit
		return ((x-self.z_hatx)**2 / a**2) + ((y-self.z_haty)**2 / b**2) - 1

	def link_states_callback(self, msg):
		try:
			index = msg.name.index('clover::base_link')
		except ValueError:

			return

		self.clover_pose = msg.pose[index]

	def obstacle(self,data):

		self.obs_x = data.x
		self.obs_y = data.y
		self.obs_vx = data.vx
		self.obs_vy = data.vy
		self.obs_ax = data.ax
		self.obs_ay = data.ay

	
	def lidar_read(self,data):


		# Update the obstacle detect array
		self.obs_detect = data.ranges

		# Ensure there are actually lidar readings, no point in doing calculations if
		# nothing is detected:
		if any(not np.isinf(range_val) for range_val in self.obs_detect):
	
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
			
			# Filter out the inf values in the data point arrays
			# x_local2 = x_local2[np.isfinite(x_local2)]
			# y_local2 = y_local2[np.isfinite(y_local2)]
			
			# The array of local data points starts at position 0 with the right most detected point, and ends with the left most detected point.
			# Therefore we will use the same setup as the RM paper and build the equation for l_1 off of the left most point (end point of the array)
			# In the local lidar reference frame, the lidar is at position (0,0)
			x_L = 0
			y_L = 0
			# Coordinate equation for stratight line l_1
			self.A_1 = -1*(self.x_local2_filtered[-1]-x_L)/(self.y_local2_filtered[-1]-y_L)
			self.B_1 = self.y_local2_filtered[-1] - self.A_1*self.x_local2_filtered[-1]
			# Coordinate equation for straight line l_2
			self.A_2 = -1*(self.x_local2_filtered[0] - x_L)/(self.y_local2_filtered[0] - y_L)
			self.B_2 = self.y_local2_filtered[0] - self.A_2*self.x_local2_filtered[0]

			# print(x_local2)
			# print(x_local2_filtered)
			# print(valid_indices)

#---------------------Just use Z to estimate l3, probably the easiest thing to do for now------------------------------------------

# Like the lidar measurements are already divided up into sectors, why would I need to redivide it up into more sectors and then take the average?
# seems strange, lets just get this working this way, also we will have more n paired data points this way.

			# Calculate the midpoints Z_bar
			# Initialize an empty list to store the averaged values
			self.x_bar = []
			self.y_bar = []
			self.Z_bar = []

			# Calculate the number of elements in the array
			n = len(self.x_local2_filtered)

			# Calculate the number of pairs (assuming an even number of elements)
			num_pairs = n // 2

			# Iterate through the array to average pairs of values
			for i in range(num_pairs):
    		# Calculate the indices of the pair
				first_index = i
				second_index = n - 1 - i
    
    		# Calculate the average of the pair and append it to x_bar
				average_valuex = (self.x_local2_filtered[first_index] + self.x_local2_filtered[second_index]) / 2
				average_valuey = (self.y_local2_filtered[first_index] + self.y_local2_filtered[second_index]) / 2
				self.x_bar.append(average_valuex)
				self.y_bar.append(average_valuey)
				self.Z_bar.append([self.x_bar,self.y_bar])

			self.x_bar = self.x_bar[::-1] # reverse so we have the points 'closest together' (look at fig in paper) at the beginning of the array
			self.y_bar = self.y_bar[::-1]	
			self.Z_bar = self.Z_bar[::-1]
			
			column_1 = np.array(self.x_bar)
			column_2 = np.ones_like(column_1)

			# set up to solve least squares problem with Cx = d where x = [A_3, B_3]
			C = np.column_stack((column_1,column_2)) 
			d = np.column_stack(np.array(self.y_bar)).T
			# solve the least squares problem
			solution = np.linalg.lstsq(C, d, rcond=None)[0]
			self.A_3 = solution[0]
			self.B_3 = solution[1]
			# print(x_local2)
			# print(self.x_local2_filtered)
			# print(self.x_bar)
			# print(C)
			# print(d)
			# print(solution) # Verify this some how, make sure there are many data points. plot the lines l_1, l_2 and l_3 in desmos to verify maybe. plot the points local_filtered and Z_bar and compare
			# maybe try plotting using matplot lib in here to verify shape and things. If the line l_3 matches up with center points and all looks good then move on...
			# print([A_1,B_1])
			# print([A_2,B_2])

			# Calculate the intersection points
			x_mu1 = (self.B_3 - self.B_1)/(self.A_1-self.A_3)
			y_mu1 = self.A_1*(self.B_3-self.B_1)/(self.A_1-self.A_3) + self.B_1

			x_mu2 = (self.B_3 - self.B_2)/(self.A_2-self.A_3)
			y_mu2 = self.A_2*(self.B_3-self.B_2)/(self.A_2-self.A_3) + self.B_2

			self.z_hatx = (x_mu1 +x_mu2)/2
			self.z_haty = (y_mu1 + y_mu2)/2 

			# Estimate the obstacle with an ellipse
			# Initial guess for major and minor axis (a,b)
			a0 = 1
			b0 = 1

			p0 = a0,b0

			# Perform unweighted curve fitting
			# popt, pcov = curve_fit(self.ellipse,self.x_local2_filtered, self.y_local2_filtered, p0)

			# If n_bar <= threshold-----------------------------

			# Calculate omega(i)
			nk = len(self.x_local2_filtered)
			omega = np.zeros(nk)

			for i in range(nk):
				if i == nk-1:
					j = nk-2
				else:
					j = i+1
				zi = np.array([self.x_local2_filtered[i], self.y_local2_filtered[i]])
				zj = np.array([self.x_local2_filtered[j], self.y_local2_filtered[j]])

				dij = min(ranges_filtered[i], ranges_filtered[j])
				rij = np.abs(angles_filtered[i]-angles_filtered[j])
				
				
				omega[i] = np.dot(zi - zj, zi - zj) / ((dij**2) * (2 - 2 * np.cos(np.abs(rij))))
				

			# Calculate normalization value tau
			tau = nk / np.sum(omega)
			
			# Normalized omega
			omega_norm = tau*omega

			self.znorm_x = np.sum(omega_norm*self.x_local2_filtered)
			self.znorm_y = np.sum(omega_norm*self.y_local2_filtered)

			# Ellipse model
			xy = np.column_stack((self.x_local2_filtered,self.y_local2_filtered))
			ellipse_model = EllipseModel()
			ellipse_model.estimate(xy)

			# Extract parameters of the fitted ellipse
			center = ellipse_model.params[0:2]
			a_fit = ellipse_model.params[2]
			b_fit = ellipse_model.params[3]
			theta = ellipse_model.params[4]

			cenx.append(center[0])
			ceny.append(center[1])
			a_elipse.append(a_fit)
			b_elipse.append(b_fit)
			beta.append(theta)


			
			xf.append(self.x_local2_filtered)
			yf.append(self.y_local2_filtered)
			A1.append(self.A_1)
			B1.append(self.B_1)
			LidarA1.append(-1/self.A_1)
			LidarB1.append(0)
			A2.append(self.A_2)
			B2.append(self.B_2)
			LidarA2.append(-1/self.A_2)
			LidarB2.append(0)
			A3.append(self.A_3)
			B3.append(self.B_3)

			mu1x.append(x_mu1)
			mu1y.append(y_mu1)
			mu2x.append(x_mu2)
			mu2y.append(y_mu2)
			zhatx.append(self.z_hatx)
			zhaty.append(self.z_haty)

			znormx.append(self.znorm_x)
			znormy.append(self.znorm_y)

			
			
			
			
			
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

			# Calculate the obstacles position, velocity, and acceleration in the drones reference frame using the AHOSMO
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

				x_obs.append(x1hat)
				vx_obs.append(x2hat)
				ax_obs.append(x3hat)
				y_obs.append(y1hat)
				vy_obs.append(y2hat)
				ay_obs.append(y3hat)

				x_obsr.append(self.obs_x)
				vx_obsr.append(self.obs_vx)
				ax_obsr.append(self.obs_ax)
				y_obsr.append(self.obs_y)
				vy_obsr.append(self.obs_vy)
				ay_obsr.append(self.obs_ay)

				# Update current values
				self.x1hat_cur = x1hat
				self.x2hat_cur = x2hat
				self.x3hat_cur = x3hat

				self.y1hat_cur = y1hat
				self.y2hat_cur = y2hat
				self.y3hat_cur = y3hat

				

			# update last_timestamp to current timestamp
			self.last_timestamp = current_timestamp
			
			
			
			
			#--------------------------------------Transformation assuming stadard polar coordinates and local y-left, local x-right---------------------
			
			# # Polar to Cartesion transformation for all readings (assuming angles are in standard polar coordinates) y-axis is left and x-axis is directly forward.
			# x_local = ranges*np.cos(angles)
			# y_local = ranges*np.sin(angles)
			
			# # Filter out the inf values in the data point arrays
			# x_local = x_local[np.isfinite(x_local)]
			# y_local = y_local[np.isfinite(y_local)]
			
			
			
			
			# # Homogenous transformation matrix for 2D
			# R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) # rotation matrix
			

			# T = np.vstack([np.hstack([R, np.array([[x_clover], [y_clover]])]),[0,0,1]]) # Homogenous transformation matrix
			
			
			
			# # Lidar readings in homogenous coordinates
			# readings_local = np.vstack([x_local, y_local, np.ones_like(x_local)])
			# readings_local2 = np.vstack([x_local2, y_local2, np.ones_like(x_local2)])
			
			# # Transform all lidar readings to global coordinates
			# self.readings_global = np.dot(T, readings_local)
			# readings_global2 = np.dot(T_2, readings_local2)
			
			# # Extract the tranformed positions
			# self.readings_global = self.readings_global[:2,:].T
			# readings_global2 = readings_global2[:2,:].T
			# print(self.readings_global)
			# print(readings_global2)

			# # Dont want to update these here!! we are using in controller and if the values are updated mid calculation for velocity itll cause many issues
			# # This is updating faster then the velocity controller calculations
			# # self.xa = readings_global[:,0].T
			# # self.ya = readings_global[:,1].T

	


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
		q=clover(FLIGHT_ALTITUDE = 1.0, RATE = 50, RADIUS = 2.0, CYCLE_S = 13, REF_FRAME = 'map')
		
		q.main()
		#print(xa)
		#print(xf)
		x = np.linspace(-1.5,1.5,100)
		# Calculate y values for lines
		y1 = A1[0]*x + B1[0]
		y_lidar1 = LidarA1[0]*x
		y2 = A2[0]*x + B2[0]
		y_lidar2 = LidarA2[0]*x
		y3 = A3[0]*x + B3[0]
		# Plot logged data for analyses and debugging
		plt.figure(1)
		# plt.scatter(xf,yf,color='r',label='lidar-data')
		plt.plot(x,y1,'b--',label='l_1')
		plt.plot(x,y_lidar1,'g',label='lidar1')
		plt.plot(x,y2,'m--',label='l_2')
		plt.plot(x,y_lidar2,'g',label='lidar2')
		plt.plot(x,y3,'r',label='l_3')
		plt.scatter(mu1x,mu1y,color='c',label='mu1')
		plt.scatter(mu2x,mu2y,color='c',label='mu2')
		plt.scatter(zhatx,zhaty,color='k',label='z_hat')
		plt.scatter(znormx,znormy,color='y',label='znorm_hat')
		plt.legend()
		plt.grid(True)
		plt.axis('equal')
		plt.xlim(-1.5,1.5)
		plt.ylim(0,3.5)
		# create an ellipse patch
		# plt.figure(2)
		ellipse = Ellipse(xy=(cenx[0],ceny[0]), width=2*a_elipse[0], height=2*b_elipse[0], angle=np.rad2deg(beta[0]),
		edgecolor='b', fc='None', lw=2)
		plt.gca().add_patch(ellipse)
		# plt.axis('equal')
	
		#plt.subplot(312)
		#plt.plot(yf,'r',label='y-fol')
		#plt.plot(ya,'b--',label='y-obs')
		#plt.legend()
		#plt.grid(True)
		#plt.ylabel('Position [m]')
		
		
		# observer plot
		plt.figure(2)
		plt.subplot(211)
		plt.plot(cenx,'r',label='obs_center')
		plt.plot(x_obs,'b',linewidth = 2,linestyle='--',label='obs_hat')
		plt.ylabel('pos[m]')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)
		plt.subplot(212)
		plt.plot(ceny,'r',label='obs_center')
		plt.plot(y_obs,'b',linewidth = 2,linestyle='--',label='obs_hat')
		plt.ylabel('pos[m]')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)
		# plt.plot(velfy,'r',label='vy-vel')
		# plt.plot(velcy,'b--',label='vy-com')
		# plt.legend()
		# plt.grid(True)
		# plt.ylabel('Position [m]')
		# plt.subplot(313)
		# plt.plot(evx,'r',label='evx')
		# plt.plot(evy,'b',label='evy')
		# plt.plot(eyaw,'g',label='eyaw')
		# plt.ylabel('Error[m]')
		# plt.xlabel('Time [s]')
		# plt.legend()
		plt.figure(3)
		plt.subplot(311)
		plt.plot(cenx,'r',label='obs_center-x')
		plt.plot(x_obs,'b',linewidth = 2,linestyle='--',label='obs_hat-x')
		plt.plot(x_obsr,'g',linewidth = 2,linestyle='--',label='real_obs-center-x')
		plt.ylabel('pos[m]')
		plt.legend()
		plt.grid(True)
		plt.subplot(312)
		plt.plot(vx_obs,'r',label='obs_vel-hat')
		plt.plot(vx_obsr,'b',linewidth = 2,linestyle='--',label='obs_vel-real')
		plt.ylabel('vel [m/s]')
		plt.legend()
		plt.grid(True)
		plt.subplot(313)
		plt.plot(ax_obs,'r',label='obs_ax-hat')
		plt.plot(ax_obsr,'b',linewidth = 2,linestyle='--',label='obs_ax-real')
		plt.ylabel('pos [m]')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)

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
