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
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import tf

import numpy as np

# Could plot the stored data in SITL (not hardware) if desired:
import matplotlib.pyplot as plt

import h5py  # use for data logging of larges sets of data
import os

# Custom message for object state publishing
from custom_message.msg import ObjectPub


from time import sleep
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

rospy.init_node('set_obstacle_dynamic') # Figure8 is the name of the ROS node



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

set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState) # Set the state of the obstacle

# Release service is used to allow for complex trajectory publishing i.e it stops the navigate service from publishing setpoints because you dont want two sources of publishing at the same time.
release = rospy.ServiceProxy('simple_offboard/release', Trigger)

# Deine math parameter
PI_2 = math.pi/2

# Define global variables to log the obstacle dynamics

# Figure-8 obstacle
xf = []
yf = []
vxf = []
vyf = []
axf = []
ayf = []

# Circle obstacle
xc = []
yc = []
vxc = []
vyc = []
axc = []
ayc = []


# Log the time for the position control variables
time_now = []

# Get the path to the directory containing the Python script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Create a HDF5 file name
# Open the HDF5 file globally
file_name = 'set_obs.h5'

# Construct the absolute path to the HDF5 file
absolute_file_path = os.path.join(script_dir, file_name)


 # Open the HDF5 file for writing
with h5py.File(absolute_file_path, 'a') as hf:


			
	# This class categorizes all of the functions used for complex rajectory tracking
	class clover:

		def __init__(self, FLIGHT_ALTITUDE, RATE, RADIUS, CYCLE_S, REF_FRAME, N_horizon, T_horizon): 
			
			# Publisher which will publish to the topic '/mavros/setpoint_velocity/cmd_vel'.
			self.velocity_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size=10)

			# Create a topic to publish the estimated obstacle dynamics on, from the observer
		# Create a publisher for the obstacle dynamics topic
			self.obstacle_dynamics_pub = rospy.Publisher('/Obstacle_state1', ObjectPub, queue_size=10)
			self.obstacle_dynamics_pub2 = rospy.Publisher('/Obstacle_state2', ObjectPub, queue_size=10)

			# MPC variables
			self.N_horizon = N_horizon # Define prediction horizone in terms of optimization intervals
			self.T_horizon = T_horizon # Define the prediction horizon in terms of time (s) --> Limits time and improves efficiency
			# This will have time step within prediction horizon as dt = T/N, would probably like it to be close to

			# Set rate fast here, it doesnt need to be slow and line up with the N_horizon and T_horizon here
			# global static variables
			self.FLIGHT_ALTITUDE = FLIGHT_ALTITUDE          # fgdf
			self.RATE            =  RATE#self.N_horizon/self.T_horizon # i think this makes sense, because dt = self.T_horizon/self.N_horizon
			self.RADIUS          = RADIUS                   # radius of figure 8 in meters
			self.CYCLE_S         = CYCLE_S                  # time to complete one figure 8 cycle in seconds
			self.CYCLE_C         = 15              # Cycle time for the circle
			self.STEPS           = int( self.CYCLE_S * self.RATE )
			self.STEPS_C           = int( self.CYCLE_C * self.RATE )
			self.FRAME           = REF_FRAME                # Reference frame for complex trajectory tracking

			# sink location
			self.xsi = 20
			self.ysi = 20


			self.current_state = State()
			self.rate = rospy.Rate(20)
			

		def main(self):
		
			
			i    = 0                        # Set the counter
			dt   = 1.0/self.RATE		# Set the sample time step
			dadt = math.pi*2 / self.CYCLE_S # first derivative of angle with respect to time
			dadt_c = math.pi*2 / self.CYCLE_C # first derivative of angle with respect to time
			r    = self.RADIUS		# Set the radius of the figure-8
			rc = 2.7 # radius of the circle
			path = []
			

			# Wait for 5 seconds
			rospy.sleep(3)
			
			
			rospy.loginfo('start figure8')          # Print a notification to the screen when beginning the figure-8
			PI=math.pi
			start = get_telemetry()                 # Use this Clover function to get the current drone state
			start_stamp = rospy.get_rostime()       # Get the current ROS time
			
			# create random time array with enough elements to complete the entire figure-8 sequence
			t = np.arange(0,self.STEPS,1)
			# create random time array with enough elements to complete the entire circle sequence
			tc = np.arange(0,self.STEPS_C,1)
			
			# Create arrays for each variable we want to feed information to (for the cylinder, big box 2
			# which will follow the figure 8 trajectory):
			posx = [1]*len(t)
			posy = [1]*len(t)
			posz = [1]*len(t)
			velx = [1]*len(t)
			vely = [1]*len(t)
			velz = [1]*len(t)
			afx = [1]*len(t)
			afy = [1]*len(t)
			afz = [1]*len(t)
			self.rotate = 0#math.pi/2 # rotate figure-8
			# Trajectory for the prism or other cylinder which will follow the circular trajectory
			posx2 = [1]*len(tc)
			posy2 = [1]*len(tc)
			posz2 = [1]*len(tc)
			velx2 = [1]*len(tc)
			vely2 = [1]*len(tc)
			velz2 = [1]*len(tc)
			afx2 = [1]*len(tc)
			afy2 = [1]*len(tc)
			afz2 = [1]*len(tc)
			
			for i in range(0, self.STEPS):
			
				# calculate the parameter 'a' which is an angle sweeping from -pi/2 to 3pi/2
				# through the figure-8 curve or circular curve
				a = (-math.pi/2) + i*(math.pi*2/self.STEPS)
				# These are definitions that will make position, velocity, and acceleration calulations easier:
				c = math.cos(a)
				c2a = math.cos(2.0*a)
				c4a = math.cos(4.0*a)
				c2am3 = c2a-3.0
				c2am3_cubed = c2am3*c2am3*c2am3
				s = math.sin(a)
				cc = c*c
				ss = s*s
				sspo = (s*s)+1.0 # sin squared plus one
				ssmo = (s*s)-1.0 # sin squared minus one
				sspos = sspo*sspo
				
				# For more information on these equations, refer to the GitBook Clover documentation:
				
				# Position (figure-8)
				# https:#www.wolframalpha.com/input/?i=%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
				posx[i] = -(r*c*s) / sspo #+4.5
				# https:#www.wolframalpha.com/input/?i=%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
				posy[i] =  (r*c)   / sspo #+ 4.5
				posz[i] =  self.FLIGHT_ALTITUDE

				# Transform the figure-8, where we rotate it by 45 degrees or pi/4
				posx[i] = posx[i]*math.cos(self.rotate) - posy[i]*math.sin(self.rotate) + 8
				posy[i] = posx[i]*math.sin(self.rotate) + posy[i]*math.cos(self.rotate) + 8


				# Velocity (figure-8)
				# https:#www.wolframalpha.com/input/?i=derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				velx[i] =   dadt*r* ( ss*ss + ss + (ssmo*cc) ) / sspos
				# https:#www.wolframalpha.com/input/?i=derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				vely[i] =  -dadt*r* s*( ss + 2.0*cc + 1.0 )    / sspos
				velz[i] =  0.0

				# Transform the figure-8, where we rotate it by 45 degrees or pi/4
				velx[i] = velx[i]*math.cos(self.rotate) - vely[i]*math.sin(self.rotate)
				vely[i] = velx[i]*math.sin(self.rotate) + vely[i]*math.cos(self.rotate)

				# Acceleration (figure-8)
				# https:#www.wolframalpha.com/input/?i=second+derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				afx[i] =  -dadt*dadt*8.0*r*s*c*((3.0*c2a) + 7.0)/ c2am3_cubed
				# https:#www.wolframalpha.com/input/?i=second+derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				afy[i] =  dadt*dadt*r*c*((44.0*c2a) + c4a - 21.0) / c2am3_cubed
				afz[i] =  0.0

				# Transform the figure-8, where we rotate it by 45 degrees or pi/4
				afx[i] = afx[i]*math.cos(self.rotate) - afy[i]*math.sin(self.rotate)
				afy[i] = afx[i]*math.sin(self.rotate) + afy[i]*math.cos(self.rotate)

			
			for i in range(0, self.STEPS_C):

				# calculate the parameter 'a' which is an angle sweeping from -pi/2 to 3pi/2
				# through the figure-8 curve or circular curve
				a = (-math.pi/2) + i*(math.pi*2/self.STEPS_C)
				# These are definitions that will make position, velocity, and acceleration calulations easier:
				c = math.cos(a)
				s = math.sin(a)
				

				# Position (circele)
				# https:#www.wolframalpha.com/input/?i=%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
				posx2[i] = rc*c + 15
				# https:#www.wolframalpha.com/input/?i=%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
				posy2[i] = rc*s + 15
				posz2[i] =  self.FLIGHT_ALTITUDE

				# Velocity (circle)
				# https:#www.wolframalpha.com/input/?i=derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				velx2[i] =  -dadt_c*rc*s
				# https:#www.wolframalpha.com/input/?i=derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				vely2[i] =  dadt_c*rc*c
				velz2[i] =  0.0

				# Acceleration (circle)
				# https:#www.wolframalpha.com/input/?i=second+derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				afx2[i] =  -dadt_c*dadt_c*rc*c
				# https:#www.wolframalpha.com/input/?i=second+derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
				afy2[i] =  -dadt_c*dadt_c*rc*s
				afz2[i] =  0.0

			
			# Define object that will be published
			state_msg_8 = ModelState() # figure-8 publisher object
			state_msg_circle = ModelState()

			state_msg_8.model_name = 'Big box 2'
			state_msg_circle.model_name = 'Big box 3'
			rr = rospy.Rate(self.RATE)
			k=0
			q = 0
			# define obstacle state objects for publishing them (this will be read by the AHOSMO)
			state_pub1 = ObjectPub()
			state_pub2 = ObjectPub()
			while not rospy.is_shutdown():
				
				# Update the position of the object (figure-8)
				state_msg_8.pose.position.x = posx[k]
				state_msg_8.pose.position.y = posy[k]
				state_msg_8.pose.position.z = posz[k]
				state_msg_8.pose.orientation.x = 0
				state_msg_8.pose.orientation.y = 0
				state_msg_8.pose.orientation.z = 0
				state_msg_8.pose.orientation.w = 0
				
				# Publish and give pose to observer node
				state_pub1.x = posx[k]
				state_pub1.y = posy[k]
				state_pub1.vx = velx[k]
				state_pub1.vy = vely[k]
				state_pub1.ax = afx[k]
				state_pub1.ay = afy[k]

				# Update the position of the object (circle)
				state_msg_circle.pose.position.x = posx2[q]
				state_msg_circle.pose.position.y = posy2[q]
				state_msg_circle.pose.position.z = posz2[q]
				state_msg_circle.pose.orientation.x = 0
				state_msg_circle.pose.orientation.y = 0
				state_msg_circle.pose.orientation.z = 0
				state_msg_circle.pose.orientation.w = 0

				# Publish and give pose to observer node
				state_pub2.x = posx2[q]
				state_pub2.y = posy2[q]
				state_pub2.vx = velx2[q]
				state_pub2.vy = vely2[q]
				state_pub2.ax = afx2[q]
				state_pub2.ay = afy2[q]

				xf.append(posx[k])
				yf.append(posy[k])
				vxf.append(velx[k])
				vyf.append(vely[k])
				axf.append(afx[k])
				ayf.append(afy[k])

				# Circle obstacle
				xc.append(posx2[q])
				yc.append(posy2[q])
				vxc.append(velx2[q])
				vyc.append(vely2[q])
				axc.append(afx2[q])
				ayc.append(afy2[q])

				 # Get the actual time:
				current_time = rospy.Time.now()
				time_now.append(current_time.to_sec())

				get_time = rospy.get_rostime()

				#print('time_obs:', current_time.to_sec(), get_time.to_sec())

								
				# Set the obstacles state
				figure_8 = set_state( state_msg_8 )

				circle = set_state(state_msg_circle)

				# Publish the obstacles state for the AHOSMO
				self.obstacle_dynamics_pub.publish(state_pub1)
				self.obstacle_dynamics_pub2.publish(state_pub2)
			
				
			
			
				# This loop initializes when the figure-8 is complete, therefore it will navigate back to the origin and setpoint publishing will be continued as needed to avoid entering failsafe mode.
				k = k+1
				q = q+1
				if k >= self.STEPS: 
					k = 0 # Reset the counter
					# break
				if q >= self.STEPS_C:
					q = 0

				telem = get_telemetry('map')
				if math.sqrt((telem.x-self.xsi) ** 2 + (telem.y-self.ysi) ** 2) < 0.6: # 0.4
					break
				rr.sleep()

			# Wait for 3 seconds
			rospy.sleep(3)
			# Perform landing
			


			
			
			# rospy.spin() # press control + C, the node will stop.

	if __name__ == '__main__':
		try:
			# Define the performance parameters here which starts the script
			q=clover(FLIGHT_ALTITUDE = 1.749502, RATE = 50, RADIUS = 3.3, CYCLE_S = 25, REF_FRAME = 'aruco_map', N_horizon = 25, T_horizon = 5)
			
			q.main()

			#-------------External LOG------------------
            # Create a group to store velocity field for this iteration/time
			iteration_group = hf.create_group('Control_log')
			iteration_group.create_dataset('xf', data=xf)
			iteration_group.create_dataset('yf', data=yf)
			iteration_group.create_dataset('vxf', data=vxf)
			iteration_group.create_dataset('vyf', data=vyf)
			iteration_group.create_dataset('axf', data=axf)
			iteration_group.create_dataset('ayf', data=ayf)

			iteration_group.create_dataset('xc', data=xc)
			iteration_group.create_dataset('yc', data=yc)
			iteration_group.create_dataset('vxc', data=vxc)
			iteration_group.create_dataset('vyc', data=vyc)
			iteration_group.create_dataset('axc', data=axc)
			iteration_group.create_dataset('ayc', data=ayc)
			
			#time_now = np.array(time_now)
			iteration_group.create_dataset('time_now', data=time_now)
                #------------------------------------------

			# Debug section, need matplotlib to plot the results for SITL
			plt.figure(8)
			plt.subplot(211)
			plt.plot(time_now,xf,'r',label='x-pos')
			plt.plot(time_now,yf,'b--',label='y-pos')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Position [m]')
			plt.subplot(212)
			plt.plot(time_now,vxf,'r',label='x-vel')
			plt.plot(time_now,vyf,'b--',label='y-vel')
			plt.legend()
			plt.grid(True)
			plt.ylabel('Velocity [m/s]')
			plt.xlabel('Time [s]')

			plt.figure(9)
			# plt.subplot(211)
			plt.plot(time_now, axf,'r',label='x-acc')
			plt.plot(time_now, ayf,'b--',label='y-acc')
			plt.ylabel('af [m/s^2]')
			plt.legend()
			plt.grid(True)
			# plt.subplot(212)
			# plt.plot(t,yawc,'r',label='yaw')
			# plt.ylabel('Magnitude')
			# plt.xlabel('Time [s]')
			# plt.plot(t,yaw_ratec,'b--',label='yaw_rate')
			# plt.legend()
			# plt.grid(True)
			plt.figure(10)
			plt.plot(xf,yf, 'b')
			plt.show()
			
		except rospy.ROSInterruptException:
			pass
		




