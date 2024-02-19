#!/usr/bin/python

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from CLOVER_MODEL import export_clover_model
from FUNCTIONS import CloverCost, acados_settings
from casadi import SX, vertcat, sin, cos, norm_2, diag, sqrt 

import rospy
from clover import srv
from std_srvs.srv import Trigger
import math
from geometry_msgs.msg import Point, PoseStamped, TwistStamped
from gazebo_msgs.msg import ModelState, LinkStates
import tf
from sensor_msgs.msg import Imu, LaserScan

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from clover import srv
from custom_message import srv as surf
from time import sleep
# import all mavros messages and services
from mavros_msgs.msg import *
from mavros_msgs.srv import *

import h5py  # use for data logging of larges sets of data
from skimage.measure import EllipseModel

from tf.transformations import euler_from_quaternion

rospy.init_node('Helix')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
land = rospy.ServiceProxy('land', Trigger)

# Add in external position of target object service
object_loc = rospy.ServiceProxy('ObjectPose',surf.ObjectPose)

# Release service to allow for complex trajectory publishing i.e stopping navigate service publishing because you dont want two sources of publishing at the same time.
release = rospy.ServiceProxy('simple_offboard/release', Trigger)

PI_2 = math.pi/2

# Global lof variables
X = []
VX = []
Y = []
Ux = []
Uy = []
Uz = []
Z = []
VZ = []

class fcuModes:

	def __init__(self):
		pass

	def setArm(self):
		rospy.wait_for_service('mavros/cmd/arming')
		try:
			armService = rospy.ServiceProxy('mavros/cmd/arming', mavros_msgs.srv.CommandBool)
			armService(True)
		except rospy.ServiceException as e:
			print ("Service arming call failed: %s")

	def setOffboardMode(self):
		rospy.wait_for_service('mavros/set_mode')
		try:
			flightModeService = rospy.ServiceProxy('mavros/set_mode', mavros_msgs.srv.SetMode)
			flightModeService(custom_mode='OFFBOARD')
		except rospy.ServiceException as e:
			print ("service set_mode call failed: %s. Offboard Mode could not be set.")

class clover:

	def __init__(self, FLIGHT_ALTITUDE = 1.0, RATE = 50, RADIUS = 3.5, V_des = 0.6, N_horizon=10, T_horizon=1.0, REF_FRAME = 'map'): # rate = 50hz radius = 5m cycle_s = 25
        # If you change prediction horizon intervals or time, need to hardcode changes in set_object_state
 		
 		# Publisher which will publish to the topic '/mavros/setpoint_velocity/cmd_vel'.
		self.velocity_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size=10)

		self.xVals  = [0, 20];  # X-grid extents [min, max]
		self.yVals  = [0, 20];  # Y-grid extents [min, max]

		distance = math.sqrt((self.xVals[1]-self.xVals[0])**2 + (self.yVals[1]-self.yVals[0])**2)

		 # global static variables
		self.FLIGHT_ALTITUDE = FLIGHT_ALTITUDE          # fgdf
		self.RATE            = RATE                     # loop rate hz
		self.FRAME           = REF_FRAME                # Reference frame for complex trajectory tracking
		self.V_des           = V_des
		self.CYCLE_S         = distance / self.V_des     # time to complete linear trajectory in seconds
		self.STEPS           = int( self.CYCLE_S * self.RATE ) # Total number of steps in the trajectory
		
		# MPC variables
		self.N_horizon = N_horizon # Define prediction horizone in terms of optimization intervals
		self.T_horizon = T_horizon # Define the prediction horizon in terms of time (s) --> Limits time and improves efficiency
		# This will have time step within prediction horizon as dt = T/N, would probably like it to be close to
		# self.CYCLE_S/self.STEPS = dt

		# Compute the prediction horizon length in terms of steps in the reference trajectory
		self.N_steps = int(self.T_horizon*self.RATE)
		self.dt_ref = int(self.N_steps/self.N_horizon) # This is the amount of steps ahead within the reference trajectory array, per iteration in the prediction horizon

		i    = 0
		dt   = 1.0/self.RATE

		# create random time array
		t = np.arange(0,self.STEPS,1)

		# Calculate the x and y velocity components (inertial of course)
		angle_degrees = 45     # Angle measured counterclockwise from the positive x-axis
		angle_radians = np.deg2rad(angle_degrees)

		# Calculate x and y components of the desired velocity
		velocity_x = self.V_des * np.cos(angle_radians)
		velocity_y = self.V_des * np.sin(angle_radians)

		
		self.posx = np.linspace(self.xVals[0], self.xVals[1], len(t))
		self.posy = np.linspace(self.yVals[0], self.yVals[1], len(t))
		self.posz = np.ones(len(t)) * self.FLIGHT_ALTITUDE
		self.velx = np.ones(len(t)) * velocity_x
		self.vely = np.ones(len(t)) * velocity_y
		self.velz = np.zeros(len(t))
		self.afx = np.zeros(len(t))
		self.afy = np.zeros(len(t))
		self.afz = np.zeros(len(t))
		# Calculate yaw as direction of velocity
		self.yawc = np.arctan2(self.vely, self.velx)
		self.yaw_ratec = [1]*len(t)
		# calculate yaw_rate by dirty differentiating yaw
		for i in range(0, self.STEPS):
			next = self.yawc[(i+1)%self.STEPS] # 401%400 = 1 --> used for this reason (when it goes over place 400 or whatever STEPS is)
			curr = self.yawc[i]
      			# account for wrap around +- PI
			if((next-curr) < -math.pi):
				next = next + math.pi*2
			if((next-curr) >  math.pi):
				next = next - math.pi*2
			self.yaw_ratec[i] = (next-curr)/dt

		#--------------Obstacle parameters-----------------------------
		self.SF = 0.3 # safety factor distance from the obstcle (set as the width of the Clover)


		#--------------------------------------------------------------
		
		
		
		# Publisher which will publish to the topic '/mavros/setpoint_raw/local'.
		self.publisher = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
		
		# Subscribe to drone state
		self.state = rospy.Subscriber('mavros/state', State, self.updateState)

		# Subscribe directly to the ground truth
		self.ground_truth = rospy.Subscriber('gazebo/link_states', LinkStates, self.link_states_callback)

		# Generate the array of lidar angles
		self.lidar_angles = np.linspace(-180*(math.pi/180), 180*(math.pi/180), 360) # Adjust this number based on number defined in XACRO!!!!!

		# Initialize the ranges array for the controller as no object detected (i.e. an array of inf)
		self.obs_detect = np.full_like(self.lidar_angles, np.inf)

		# Subscribe to the Lidar readings
		self.lidar = rospy.Subscriber('/ray_scan',LaserScan,self.lidar_read)

		# Set a flag, that will be used as a logic operator to adjust how the HOCBF is set, which depends on whether an obstacle is detected or not
		self.object_detect = False

		self.current_state = State()
		self.rate = rospy.Rate(20)

	def updateState(self, msg):
		self.current_state = msg

	def link_states_callback(self, msg):
		try:
			index = msg.name.index('clover::base_link')
		except ValueError:

			return

		self.clover_pose = msg.pose[index]

	def lidar_read(self,data):

		# Read the current timestamp of the readings
		self.lidar_timestamp = data.header.stamp.to_sec()

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
		# Need to add a lidar detection threshold for ellipsoid estimation, so if we have like 1-2 or low detection we could get bad results
		if sum(not np.isinf(range_val) for range_val in self.obs_detect) >= 4 and z_clover >= 0.65: # want the drone to be at some altitude so we are not registering ground detections


			# The angles and ranges start at -180 degrees i.e. at the right side, then go counter clockwise up to the top i.e. 180 degrees
			self.ranges = data.ranges

			angles = self.lidar_angles


			# Convert ranges to a NumPy array if it's not already
			self.ranges = np.array(self.ranges)


			# Polar to Cartesion transformation for all readings (assuming angles are in standard polar coordinates) y-axis is left and x-axis is directly forward.
			self.x_local = self.ranges*np.cos(angles)
			y_local = self.ranges*np.sin(angles)

			# This transforms to the local Lidar frame where the y-axis is forward
			# and the x-axis is pointing right:
			# x_local = ranges*np.sin(angles)
			# x_local = np.multiply(x_local,-1)
			# y_local = ranges*np.cos(angles)

			#------------------2D transformations--------------------
			# Homogenous transformation matrix for 2D
			R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) # rotation matrix
			T = np.vstack([np.hstack([R, np.array([[x_clover], [y_clover]])]),[0,0,1]]) # Homogenous transformation matrix

			# Lidar readings in homogenous coordinates
			readings_local = np.vstack([self.x_local, y_local, np.ones_like(self.x_local)])

			# Transform all lidar readings to global coordinates
			self.readings_global = np.dot(T, readings_local)

			# Extract the tranformed positions
			self.readings_global = self.readings_global[:2,:].T

			# Update the lidar detection readings
			self.xa = self.readings_global[:,0].T
			self.ya = self.readings_global[:,1].T

			# Filter out the inf values in the data point arrays
			self.xa = self.xa[np.isfinite(self.xa)]
			self.ya = self.ya[np.isfinite(self.ya)]

			# Ellipse model
			xy = np.column_stack((self.xa,self.ya))
			ellipse_model = EllipseModel()
			ellipse_model.estimate(xy)

			# Extract parameters of the fitted ellipse
			self.xc = ellipse_model.params[0]
			self.yc = ellipse_model.params[1]
			self.a_fit = ellipse_model.params[2]
			self.b_fit = ellipse_model.params[3]
			self.theta = ellipse_model.params[4]

			self.object_detect = True # Update object detected flag
		else:
			self.object_detect = False # Update object detected flag

               

	def main(self):#,x=0, y=0, z=2, yaw = float('nan'), speed=1, frame_id ='',auto_arm = True,tolerance = 0.2):
		
		
		 # load model
		acados_solver, acados_integrator, model = acados_settings(self.N_horizon, self.T_horizon)
		# dimensions
		nx = model.x.size()[0]
		nu = model.u.size()[0]
		ny = nx + nu
		

		# Wait for 3 seconds
		rospy.sleep(3)
		
		# Takeoff with Clovers navigate function
		navigate(x=0, y=0, z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.5, frame_id = 'map', auto_arm = True)
		
		# Give the Clover time to reach the takeoff altitude
		rospy.sleep(15)

		# Define object that will be published
		target = PositionTarget()
		rr = rospy.Rate(self.RATE)
		k=0    # Define a counter for the reference arrays
		release() # stop navigate service from publishing

		# update initial condition (current pose of MPC problem)
		telem = get_telemetry(frame_id = 'map')
		x_clover = self.clover_pose.position.x
		y_clover = self.clover_pose.position.y
		z_clover = self.clover_pose.position.z
		quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
		euler_angles = euler_from_quaternion(quaternion)
		roll = euler_angles[2] #+ math.pi
		yaw = -euler_angles[0]+math.pi 
		pitch = euler_angles[1]

		x0 = np.array([x_clover, telem.vx, y_clover, telem.vy, z_clover, telem.vz])
		acados_solver.set(0, "lbx", x0) # Update the zero shooting node position
		acados_solver.set(0, "ubx", x0) # update the zero shooting node control input

		
		while not rospy.is_shutdown():
		
			 # Trajectory publishing-----------------------------------------------
			target.header.frame_id = self.FRAME  # Define the frame that will be used
			
			target.coordinate_frame = 1 #MAV_FRAME_LOCAL_NED  # =1
			
			target.type_mask = 1+2+4+8+16+32 # Use everything!
			#target.type_mask =  3576 # Use only position #POSITION_TARGET_TYPEMASK_VX_IGNORE | POSITION_TARGET_TYPEMASK_VY_IGNORE | POSITION_TARGET_TYPEMASK_VZ_IGNORE | POSITION_TARGET_TYPEMASK_AX_IGNORE | POSITION_TARGET_TYPEMASK_AY_IGNORE |POSITION_TARGET_TYPEMASK_AZ_IGNORE | POSITION_TARGET_TYPEMASK_FORCE_IGNORE | POSITION_TARGET_TYPEMASK_YAW_IGNORE | POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE # = 3576
			#target.type_mask =  3520 # Use position and velocity
			#target.type_mask =  3072 # Use position, velocity, and acceleration
			#target.type_mask =  2048 # Use position, velocity, acceleration, and yaw

			# obj = object_loc(frame_id = 'map')
			# test2 = np.array([obj.x[0],0,0.0,obj.y[self.N_horizon],0,0.0])
			# print(type(test2))
			# print(test2.dtype)
			# print(obj.y.dtype)
			# print(obj.y)
			# print(obj.y[0])
			# print(obj.y[self.N_horizon])
			# print(obj.x[-1])
			
			# update reference
			for j in range(self.N_horizon): # Up to N-1
				#index = k + j
				index = k + self.dt_ref # Taking into account the amount of time per prediction horizon interval relative to the sampling rate of the reference trajectory
				# Check if index is within the bounds of the arrays
				if index < len(self.posx):
					yref = np.array([self.posx[index], self.velx[index], self.posy[index], self.vely[index], self.posz[index], self.velz[index], 0, 0, 0])
				else:
					# If index exceeds the length of the arrays, use the last elements
					yref = np.array([self.posx[-1], self.velx[-1], self.posy[-1], self.vely[-1], self.posz[-1], self.velz[-1], 0, 0, 0])
    
				acados_solver.set(j, "yref", yref)

				if not self.object_detect:

					# Set the distance from the obstacle
					r = 0
				
					acados_solver.set(j, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
				else:
					# Set the distance from the obstacle
					r = self.SF + max(self.a_fit, self.b_fit)

					acados_solver.set(j, "p", np.array([self.xc,0,0,self.yc,0,0,r])) # Assuming a static obstacle
					# acados_solver.set(j, "p", np.array([2.5,0,0,2.5,0,0])) # State = [x, vx, ax, y, vy, ay]
					# acados_solver.set(j, "p", np.array([5.5678987695432+j*0.00003458792871,0.2,0,obj.y[j],0.2,0])) # Works
					# acados_solver.set(j, "p", np.array([obj.x[j],obj.vx[j],obj.ax[j],obj.y[j],obj.vy[j],obj.ay[j]]))
					# test = np.array([5.5678987695432+j*0.00003458792871,0.2,0,5.67865,0.2,0])
					# print(type(test))
					# print(test.dtype)
					# acados_solver.set(self.N_horizon, "p", np.array([obj.x[j],0,0.0,obj.y[j],0,0.0])) # State = [x, vx, ax, y, vy, ay]

			#index2 = k + self.N_horizon
			index2 = k + self.N_steps # This considers the amount of steps in the reference trajectory, considering time of prediction horizon and rate of reference trajectory
			if index2 < len(self.posx):

				yref_N = np.array([self.posx[k+self.N_horizon],self.velx[k+self.N_horizon],self.posy[k+self.N_horizon],self.vely[k+self.N_horizon],self.posz[k+self.N_horizon],self.velz[k+self.N_horizon]]) # terminal components
			else:
				yref_N = np.array([self.posx[-1], self.velx[-1], self.posy[-1], self.vely[-1], self.posz[-1], self.velz[-1]])

			acados_solver.set(self.N_horizon, "yref", yref_N)
			if not self.object_detect:

				# Set the distance from the obstacle
				r = 0
				acados_solver.set(self.N_horizon, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
			else:
				# Set the distance from the obstacle
				r = self.SF + max(self.a_fit, self.b_fit)

				# Assuming a static obstacle
				acados_solver.set(self.N_horizon, "p", np.array([self.xc,0,0,self.yc,0,0,r])) # State = [x, vx, ax, y, vy, ay]
				# acados_solver.set(self.N_horizon, "p", np.array([2.5,0,0,2.5,0,0])) # State = [x, vx, ax, y, vy, ay]
				# acados_solver.set(j, "p", np.array([5.6,0.2,0,5.6,0.2,0]))
				#acados_solver.set(self.N_horizon, "p", np.array([obj.x[self.N_horizon-1],obj.vx[self.N_horizon-1],obj.ax[self.N_horizon-1],obj.y[self.N_horizon-1],obj.vy[self.N_horizon-1],obj.ay[self.N_horizon-1]])) # State = [x, vx, ax, y, vy, ay]
				# acados_solver.set(self.N_horizon, "p", np.array([obj.x[-1],obj.vx[-1],obj.ax[-1],obj.y[-1],obj.vy[-1],obj.ay[-1]])) # State = [x, vx, ax, y, vy, ay]

			# Solve ocp
			status = acados_solver.solve()

			# get solution
			x0 = acados_solver.get(0, "x")
			u0 = acados_solver.get(0, "u")

			# Gather position for publishing
			# target.position.x = posx[k] +0.15
			# target.position.y = posy[k]
			# target.position.z = posz[k]
			
			# Gather velocity for publishing
			# target.velocity.x = velx[k]
			# target.velocity.y = vely[k]
			# target.velocity.z = velz[k]
			
			# Gather acceleration for publishing
			# target.acceleration_or_force.x = afx[k]
			# target.acceleration_or_force.y = afy[k]
			# target.acceleration_or_force.z = afz[k]

			# Feedforward acceleration setpoints as well
			target.acceleration_or_force.x = u0[0] #+ afx[k]
			target.acceleration_or_force.y = u0[1] #+ afy[k]
			target.acceleration_or_force.z = u0[2] #+ afz[k]
			
			# Might not want to consider yaw and yawrate as it will intefere with MPC and its constraints
			# Gather yaw for publishing
			target.yaw = self.yawc[k]
			
			# Gather yaw rate for publishing
			target.yaw_rate = self.yaw_ratec[k]
			
			
			self.publisher.publish(target)

			# update initial condition
			#x0 = acados_solver.get(1, "x")
			telem = get_telemetry(frame_id = 'map')
			x0 = np.array([x_clover, telem.vx, y_clover, telem.vy, z_clover, telem.vz])
			acados_solver.set(0, "lbx", x0) # Update the zero shooting node position
			acados_solver.set(0, "ubx", x0)

			# logging/debugging
			X.append(telem.x)
			VX.append(telem.vx)
			Y.append(telem.y)
			Ux.append(u0[0])
			Uy.append(u0[1])
			Uz.append(u0[2])
			Z.append(telem.z)
			VZ.append(telem.vz)
			
			
		
			#set_position(x=posx[k], y=posy[k], z=posz[k],frame_id='aruco_map')
			#set_velocity(vx=velx[k], vy=vely[k], vz=velz[k],frame_id='aruco_map')#,yaw = yawc[i]) 
			#set_attitude(yaw = yawc[k],pitch = pitchc[k], roll = rollc[k], thrust = float('nan'),frame_id='aruco_map')
			#set_rates(yaw_rate = yaw_ratec[k],thrust = float('nan'))
		
		
			
			k = k+1
			if k >= self.STEPS: 
				navigate(x=0, y=0, z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.5, frame_id = 'map')
				rospy.sleep(5)
				break
			rr.sleep()

		# Wait for 5 seconds
		rospy.sleep(2)
		# Perform landing
		
		res = land()
		 
		if res.success:
			print('Drone is landing')

	# If we press control + C, the node will stop.
		rospy.sleep(6)
		# debug section
		plt.figure(1)
		# plt.subplot(211)
		#plt.plot(t,velx)
		plt.plot(self.posx,self.posy)
		plt.plot(X,Y)
		plt.axis('equal')
		# plt.subplot(212)
		# plt.plot(t,velx)
		# plt.plot(VX)
		#plt.plot(t,afx)
		#plt.plot(t,yawc)
		#plt.plot(t,yaw_ratec)
		#plt.show()

		plt.figure(2)
		plt.subplot(311)
		plt.plot(Ux,'r')
		plt.legend()
		plt.grid(True)
		plt.ylabel('yaw [deg]')
		plt.xlabel('Time [s]')
		plt.subplot(312)
		plt.plot(Uz,'r')
		plt.grid(True)
		plt.subplot(313)
		plt.plot(VZ,'r')
		plt.grid(True)

		# plt.figure(4)
		# plt.subplot(311)
		# plt.plot(t, Ux,'r')
		# plt.legend()
		# plt.grid(True)
		# plt.ylabel('Ux')
		# plt.subplot(312)
		# plt.plot(t, Uy,'b')
		# plt.legend()
		# plt.grid(True)
		# plt.ylabel('Uy')
		# plt.subplot(313)
		# plt.plot(t, Uz,'r')
		# plt.legend()
		# plt.grid(True)
		# plt.ylabel('T_input')
		# plt.xlabel('Time [s]')
		plt.show()
		
		
		rospy.spin()

if __name__ == '__main__':
	try:
		# Desired velocity of the Clover:
		v_des = 0.6 # m/s
		q=clover()
		
		q.main()#x=0,y=0,z=1,frame_id='aruco_map')
		
	except rospy.ROSInterruptException:
		pass

