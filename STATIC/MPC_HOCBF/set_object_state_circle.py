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
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import tf

# Custom message for object state publishing
from custom_message.msg import ObjectPub

import numpy as np

# Could plot the stored data in SITL (not hardware) if desired:
import matplotlib.pyplot as plt


from time import sleep
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

rospy.init_node('box') # Figure8 is the name of the ROS node

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

			
# This class categorizes all of the functions used for complex rajectory tracking
class clover:

	def __init__(self, FLIGHT_ALTITUDE, RATE, RADIUS, CYCLE_S, REF_FRAME): 
 		
 		# Publisher which will publish to the topic '/mavros/setpoint_velocity/cmd_vel'.
		self.velocity_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size=10)

		# global static variables
		self.FLIGHT_ALTITUDE = FLIGHT_ALTITUDE          # fgdf
		self.RATE            = RATE                     # loop rate hz
		self.RADIUS          = RADIUS                   # radius of figure 8 in meters
		self.CYCLE_S         = CYCLE_S                  # time to complete one figure 8 cycle in seconds
		self.STEPS           = int( self.CYCLE_S * self.RATE )
		self.FRAME           = REF_FRAME                # Reference frame for complex trajectory tracking

		# Create a publisher to provide the state of the object
		self.state = rospy.Publisher('Obstacle_state', ObjectPub, queue_size=10)

		# MPC variables (hardcode horizon variables from figure8 script)
		self.N_horizon = 10 # Define prediction horizone in terms of optimization intervals (of the Clover)
		self.T_horizon = 1 # Define the prediction horizon in terms of time (s) --> Limits time and improves efficiency
		# This will have time step within prediction horizon as dt = T/N, would probably like it to be close to
		# self.CYCLE_S/self.STEPS = dt

		# Compute the prediction horizon length in terms of steps in the reference trajectory
		self.N_steps = int(self.T_horizon*self.RATE)
		self.dt_ref = int(self.N_steps/self.N_horizon) # This is the amount of steps ahead within the reference trajectory array, per iteration in the prediction horizon


		self.current_state = State()
		self.rate = rospy.Rate(20)
        

	def main(self):
	
		
		i    = 0                        # Set the counter
		dt   = 1.0/self.RATE		# Set the sample time step
		dadt = math.pi*2 / self.CYCLE_S # first derivative of angle with respect to time
		r    = self.RADIUS		# Set the radius of the circle
		path = []
		

		# Wait for 5 seconds
		rospy.sleep(3)
		
		
		rospy.loginfo('start figure8')          # Print a notification to the screen when beginning the figure-8
		PI=math.pi
		start = get_telemetry()                 # Use this Clover function to get the current drone state
		start_stamp = rospy.get_rostime()       # Get the current ROS time
		
		# create random time array with enough elements to complete the entire figure-8 sequence
		t = np.arange(0,self.STEPS,1)
		
		# Create arrays for each variable we want to feed information to:
		posx = [1]*len(t)
		posy = [1]*len(t)
		posz = [1]*len(t)
		velx = [1]*len(t)
		vely = [1]*len(t)
		velz = [1]*len(t)
		afx = [1]*len(t)
		afy = [1]*len(t)
		afz = [1]*len(t)
		yawc = [1]*len(t)
		pitchc = [1]*len(t)
		rollc = [1]*len(t)
		yaw_ratec = [1]*len(t)
		
		for i in range(0, self.STEPS):
		
			# calculate the parameter 'a' which is an angle sweeping from -pi/2 to 3pi/2
			# through the circle curve. 
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
			
			# Position
			# https:#www.wolframalpha.com/input/?i=%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
			posx[i] = r*c
			# https:#www.wolframalpha.com/input/?i=%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29
			posy[i] = r*s
			posz[i] =  self.FLIGHT_ALTITUDE

			# Velocity
			# https:#www.wolframalpha.com/input/?i=derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
			velx[i] =  -dadt*r*s
			# https:#www.wolframalpha.com/input/?i=derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
			vely[i] =  dadt*r*c
			velz[i] =  0.0

			# Acceleration
			# https:#www.wolframalpha.com/input/?i=second+derivative+of+%28-r*cos%28a%29*sin%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
			afx[i] =  -dadt*dadt*c
			# https:#www.wolframalpha.com/input/?i=second+derivative+of+%28r*cos%28a%29%29%2F%28%28sin%28a%29%5E2%29%2B1%29+wrt+a
			afy[i] =  -dadt*dadt*r*s
			afz[i] =  0.0

			# calculate yaw as direction of velocity vector:
			yawc[i] = math.atan2(vely[i], velx[i])
			
			# calculate Pitch and Roll angles, if publishing acceleration isnt possible could send these low level commands
			pitchc[i] = math.asin(afx[i]/9.81)
			rollc[i] = math.atan2(afy[i], afz[i])
		
		# calculate yaw_rate by dirty differentiating yaw
		for i in range(0, self.STEPS):
			next = yawc[(i+1)%self.STEPS] # 401%400 = 1 --> used for this reason (when it goes over place 400 or whatever STEPS is)
			curr = yawc[i]
      			# account for wrap around +- PI
			if((next-curr) < -math.pi):
				next = next + math.pi*2
			if((next-curr) >  math.pi):
				next = next - math.pi*2
			yaw_ratec[i] = (next-curr)/dt
		
		# Define object that will be published
		state_msg = ModelState()
		state_msg.model_name = 'Big box'
		rr = rospy.Rate(self.RATE)
		k=0
		# Create an object for state publishing
		pose = ObjectPub()
		while not rospy.is_shutdown():
			
			# Update the position of the object
			state_msg.pose.position.x = posx[k]
			state_msg.pose.position.y = posy[k]
			state_msg.pose.position.z = posz[k]
			state_msg.pose.orientation.x = 0
			state_msg.pose.orientation.y = 0
			state_msg.pose.orientation.z = 0
			state_msg.pose.orientation.w = 0
			
			resp = set_state( state_msg )

			# Prelocatted lists
			pose.x = [0.0]*11
			pose.y = [0.0]*11
			pose.vx = [0.0]*11
			pose.vy = [0.0]*11
			pose.ax = [0.0]*11
			pose.ay = [0.0]*11

			#Publish the current state and the predicted state over the prediction horizon
			j = 0 # array wrap around index
			for i in range(11): # up to prediction horizon=10 for now (so n+1 spots)
				#index = k + j
				index = k + self.dt_ref # Taking into account the amount of time per prediction horizon interval relative to the sampling rate of the reference trajectory
				# Check if index is within the bounds of the arrays
				if index < len(posx):
					pose.x[i] = posx[index]
					pose.y[i] = posy[index]
					pose.vx[i] = velx[index]
					pose.vy[i] = vely[index]
					pose.ax[i] = afx[index]
					pose.ay[i] = afy[index]
				else:
					# If index exceeds the length of the arrays, restart data setting from the beginning of the arrays
					pose.x[i] = posx[j]
					pose.y[i] = posy[j]
					pose.vx[i] = velx[j]
					pose.vy[i] = vely[j]
					pose.ax[i] = afx[j]
					pose.ay[i] = afy[j]
					#j+=1
					j+=self.dt_ref # This considers the amount of steps in the reference trajectory, considering time of prediction horizon and rate of reference trajectory (sampled)

			self.state.publish(pose)
			
		
		
			# This loop initializes when the figure-8 is complete, therefore it will navigate back to the origin and setpoint publishing will be continued as needed to avoid entering failsafe mode.
			k = k+1
			if k >= self.STEPS: 
				k = 0 # Reset the counter
			rr.sleep()

		# Wait for 3 seconds
		rospy.sleep(3)
		# Perform landing
		


		# Debug section, need matplotlib to plot the results for SITL
		plt.figure(1)
		plt.subplot(211)
		plt.plot(t,posx,'r',label='x-pos')
		plt.plot(t,posy,'b--',label='y-pos')
		plt.plot(t,posz,'g:',label='z-pos')
		plt.legend()
		plt.grid(True)
		plt.ylabel('Position [m]')
		plt.subplot(212)
		plt.plot(t,velx,'r',label='x-vel')
		plt.plot(t,vely,'b--',label='y-vel')
		plt.plot(t,velz,'g:',label='z-vel')
		plt.legend()
		plt.grid(True)
		plt.ylabel('Velocity [m/s]')
		plt.xlabel('Time [s]')

		plt.figure(2)
		plt.subplot(211)
		plt.plot(t, afx,'r',label='x-acc')
		plt.plot(t, afy,'b--',label='y-acc')
		plt.plot(t, afz,'g:',label='z-acc')
		plt.ylabel('af [m/s^2]')
		plt.legend()
		plt.grid(True)
		plt.subplot(212)
		plt.plot(t,yawc,'r',label='yaw')
		plt.ylabel('Magnitude')
		plt.xlabel('Time [s]')
		plt.plot(t,yaw_ratec,'b--',label='yaw_rate')
		plt.legend()
		plt.grid(True)
		plt.show()
		
		# rospy.spin() # press control + C, the node will stop.

if __name__ == '__main__':
	try:
		# Define the obstacle performance parameters here which starts the script (of the obstacle reference trajectory)
		q=clover(FLIGHT_ALTITUDE = 1.74, RATE = 50, RADIUS = 2.5, CYCLE_S = 18, REF_FRAME = 'aruco_map') # flight altitude = 0 for big box 2 i think
		
		q.main()
		
	except rospy.ROSInterruptException:
		pass
    




