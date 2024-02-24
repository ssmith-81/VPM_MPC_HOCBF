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

from panel_functions import CLOVER_COMPONENTS, CLOVER_STREAM_GEOMETRIC_INTEGRAL, CLOVER_KUTTA, CLOVER_STREAMLINE, CLOVER_noOBSTACLE
from scipy.interpolate import griddata


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
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse


from tf.transformations import euler_from_quaternion

rospy.init_node('MPC')

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

# Debugging and logging
xf = []  # This gathers the clover position
yf = []
# position command from MPC dyamic output
xcom = []
ycom = []
YawF = []
YawC = []

# readings obstacle (for the cylinder)
xa = []
ya = []
# obstacle readings (for the prism)
xap = []
yap = []

#ellipse estimation from lidar data
a_fit = []
b_fit = []
theta = []
xc = []# for the ellipse
yc = []
xcp = [] # for the prism
ycp = []
# Correspond to cylinder
x_ce = [] # these ones are for logging position/velocity corresponding to the lidar measurements
y_ce = [] # so we can calculate psi_0 and psi_1 offline or after the simulation
vx_ce = []
vy_ce = []
# Correspond to prism
x_pe = [] # these ones are for logging position/velocity corresponding to the lidar measurements
y_pe = [] # so we can calculate psi_0 and psi_1 offline or after the simulation
vx_pe = []
vy_pe = []


# Analyze control input (see if error is being minimized )
velfx=[]
velfy=[]
# velocity command from MPC dynamic output
velcx=[]
velcy=[]
# log velocity command from the VPM velocity field
uVPM = []
vVPM = []

# updated velocity field plot
nGridX = 20  # 20                                                         # X-grid for streamlines and contours
nGridY = 20  # 20                                                       # Y-grid for streamlines and contours
x_field = np.zeros((nGridX, nGridY))# np.zeros((30, 30))
y_field = np.zeros((nGridX, nGridY)) # np.zeros((30, 30))
u_field = np.zeros((nGridX, nGridY))
v_field = np.zeros((nGridX, nGridY))
lidar_x = []
lidar_y = []

# log the invarient sets for cylinder
psi_0 = []
psi_1 = []
# for prism
psi_0p = []
psi_1p = []

U_infx = []
V_infy=[]
evx=[]
evy=[]
eyaw=[]

# Global lof variables
X = []
VX = []
Y = []
Ux = []
Uy = []
Uz = []
Z = []
VZ = []

class ellipse:

	def ellipse_calc(self,xa,ya, x_ce, vx_ce, y_ce, vy_ce, xap,yap, x_pe, vx_pe, y_pe, vy_pe):

		# Because the ellipse estimation is too computationally expensive for this VM 
		# for real time estimation, we will use it for offline ellipse estimation
		# (wont be able to do this for hardware)

		#--------------Obstacle parameters-----------------------------
		self.SF = 0.8 # safety factor distance from the obstcle (set as the width of the Clover)
		self.SFp = 1.2 # safety factor for prism
		self.cyl_rad = 1.5 # [m] radius of the cylinder
		self.rec_rad = 1.25 # [m] half of the longest side of the prism

		#--------------------------------------------------------------

		# Doing the cylinder and prism seperately because the prism is giving straneg results on psi_1

		# Iterate over rows of data (for the prism)
		for i, (xa_row, ya_row) in enumerate(zip(xap, yap)):
			# Ellipse model-------------------------
			xy = np.column_stack((xa_row, ya_row))
			ellipse_model = EllipseModel()
			ellipse_model.estimate(xy)

			# If ellipse_model.params is None, skip this iteration
			if ellipse_model.params is None:
				continue
			
			if ellipse_model.params[0] <= 11.5 - 1.25 or ellipse_model.params[0] >= 11.5 + 1.25:
				continue
			if ellipse_model.params[1] <= 15.2 - 1.25 or ellipse_model.params[1] >= 15.2 + 1.25:
				continue

			#----------- Calculate delta_p, delta_v, and delta_a-----------------------------------
			delta_p = np.array([x_pe[i]-ellipse_model.params[0], y_pe[i]-ellipse_model.params[1]])
			delta_v = np.array([vx_pe[i]-0, vy_pe[i]-0]) # static obstacle assumption for now

			# Calculate norms
			norm_delta_p = np.linalg.norm(delta_p, ord=2)  # Euclidean norm
			norm_delta_v = np.linalg.norm(delta_v, ord=2)  # Euclidean norm

			# constants
			q1 = 12#15
			q2 = 9#10
			#print(max(max(ellipse_model.params[2], ellipse_model.params[3]),0.8)) # woow, some pretty unstacle shapes come out of this on the prism
			r = self.SFp + self.rec_rad #max(max(ellipse_model.params[2], ellipse_model.params[3]),0.8) # make sure it is at least giving 0.8, where the max prism radius is 1.25 i believe

			self.psi_0p = norm_delta_p - r
			self.psi_1p = (np.dot(delta_p, delta_v)) / norm_delta_p + q1*(self.psi_0p)

			psi_0p.append(self.psi_0p)
			psi_1p.append(self.psi_1p)
			# Extract parameters of the fitted ellipse
			xcp.append(ellipse_model.params[0])
			ycp.append(ellipse_model.params[1])
	#--------------------------------------------------------------------------------------------

		# Iterate over rows of data (for the cylinder)
		for i, (xa_row, ya_row) in enumerate(zip(xa, ya)):#, x_ce, vx_ce, y_ce, vy_ce):
			# Ellipse model-------------------------
			xy = np.column_stack((xa_row, ya_row))
			ellipse_model = EllipseModel()
			ellipse_model.estimate(xy)

			# If ellipse_model.params is None, skip this iteration
			if ellipse_model.params is None:
				continue

			#----------- Calculate delta_p, delta_v, and delta_a-----------------------------------
			delta_p = np.array([x_ce[i]-ellipse_model.params[0], y_ce[i]-ellipse_model.params[1]])
			delta_v = np.array([vx_ce[i]-0, vy_ce[i]-0]) # static obstacle assumption for now

			# Calculate norms
			norm_delta_p = np.linalg.norm(delta_p, ord=2)  # Euclidean norm
			norm_delta_v = np.linalg.norm(delta_v, ord=2)  # Euclidean norm

			# constants
			q1 = 12#15
			q2 = 9#10
			r = self.SF + max(ellipse_model.params[2], ellipse_model.params[3])

			self.psi_0 = norm_delta_p - r
			self.psi_1 = (np.dot(delta_p, delta_v)) / norm_delta_p + q1*(self.psi_0)

			psi_0.append(self.psi_0)
			psi_1.append(self.psi_1)
	#--------------------------------------------------------------------------------------------


			# Extract parameters of the fitted ellipse
			xc.append(ellipse_model.params[0])
			yc.append(ellipse_model.params[1])
			a_fit.append(ellipse_model.params[2])
			b_fit.append(ellipse_model.params[3])
			theta.append(ellipse_model.params[4])
		
		# Return lists of ellipse parameters for each row of data
		return xc, yc, a_fit, b_fit, theta, psi_0, psi_1


class clover:

	def __init__(self, FLIGHT_ALTITUDE = 0.7, RATE = 50, RADIUS = 3.5, V_des = 0.6, N_horizon=25, T_horizon=5.0, REF_FRAME = 'map'): # rate = 50hz radius = 5m cycle_s = 25 N_h = 25, T_h = 5

		# Note on N_horizon and T_horizon: with how I have things now, set T_horizon higher to react quicker to obstacles, but if you increase, this increases the time ahead of the first shooting node some 
		# which is which may increase speed some. N_horizon, increase to slow drone down some and react faster to obstacles. Lower, increases time to first shoting node and increases speed

		# Define the sink location and strength
		self.g_sink = 2.00
		self.xsi = 20 # 20
		self.ysi = 20 # 20

		# Define the source strength and location
		self.g_source = 0.8
		self.xs = -0.1
		self.ys = -0.1
		# self.xs = 5
		# self.ys = 2.5


		# Define the source strength placed on the actual drone
		self.g_clover = 0.0

		# Could possible just put a source on the drone and not bother with the two different sources


		# Free flow constant
		self.U_inf = 0
		self.V_inf = 0

		self.alpha = 70*(math.pi/180)

	# iNITIALIZE INPUT VELOCITY VALUES
		self.u = 0
		self.v = 0

		# Define the max velocity allowed for the Clover
		self.vel_max = 0.65 # [m/s] 0.65

		# #-------------------- Offline Panel Calculations---------------------------------------------------

		# An object was not detected yet, so use the source and sink for velocity based navigation

		## Compute Streamlines with stream function velocity equation
		# Too many gridpoints is not good, it will cause the control loop to run too slow
		# Grid parameters
		self.nGridX = 20;  # 20 is good                                                         # X-grid for streamlines and contours
		self.nGridY = 20;  # 20 is good                                                    # Y-grid for streamlines and contours
		self.xVals  = [-1, 21];  # ensured it is extended past the domain incase the clover leaves domain             # X-grid extents [min, max]
		self.yVals  = [-1, 21];  #-0.3;0.3                                                 # Y-grid extents [min, max]

		# Define Lidar range (we will set parameters to update grid resolution within detection range):
		self.lidar_range = 3.5 # [m]

		
		# Create an array of starting points for the streamlines
		x_range = np.linspace(0, 10, int((10-0)/0.5) + 1)
		y_range = np.linspace(0, 10, int((10-0)/0.5) + 1)

		x_1 = np.zeros(len(y_range))
		y_1 = np.zeros(len(x_range))
		Xsl = np.concatenate((x_1, x_range))
		Ysl = np.concatenate((np.flip(y_range), y_1))
		XYsl = np.vstack((Xsl,Ysl)).T

		# Generate the grid points
		Xgrid = np.linspace(self.xVals[0], self.xVals[1], self.nGridX)
		Ygrid = np.linspace(self.yVals[0], self.yVals[1], self.nGridY)
		self.XX, self.YY = np.meshgrid(Xgrid, Ygrid)

		self.Vxe = np.zeros((self.nGridX, self.nGridY))
		self.Vye = np.zeros((self.nGridX, self.nGridY))

		# Get the current state/position of th clover for its source
		telem = get_telemetry(frame_id='map')


		for m in range(self.nGridX):
			for n in range(self.nGridY):
				XP, YP = self.XX[m, n], self.YY[m, n]
				u, v = CLOVER_noOBSTACLE(XP, YP, self.U_inf, self.V_inf, self.xs, self.ys, self.xsi, self.ysi, self.g_source, self.g_sink, self.g_clover, telem.x, telem.y)

				self.Vxe[m, n] = u
				self.Vye[m, n] = v

		# Flatten the grid point matices and velocity matrices into vector arrays for the griddata function
		self.XX_f = self.XX.flatten()
		self.YY_f = self.YY.flatten()
		self.Vxe_f = self.Vxe.flatten()
		self.Vye_f = self.Vye.flatten()

		
 		
 		# Publisher which will publish to the topic '/mavros/setpoint_velocity/cmd_vel'.
		self.velocity_publisher = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel',TwistStamped, queue_size=10)

		#-------MPC formulation----------------------------------------
		# MPC variables
		self.N_horizon = N_horizon # Define prediction horizone in terms of optimization intervals
		self.T_horizon = T_horizon # Define the prediction horizon in terms of time (s) --> Limits time and improves efficiency
		# This will have time step within prediction horizon as dt = T/N, would probably like it to be close to
		# self.CYCLE_S/self.STEPS = dt

		# Compute the prediction horizon length in terms of steps in the reference trajectory
		# self.N_steps = int(self.T_horizon*self.RATE)
		# self.dt_ref = int(self.N_steps/self.N_horizon) # This is the amount of steps ahead within the reference trajectory array, per iteration in the prediction horizon

		# load model
		self.acados_solver, acados_integrator, model = acados_settings(self.N_horizon, self.T_horizon)
		# dimensions
		self.nx = model.x.size()[0]
		self.nu = model.u.size()[0]
		ny = self.nx + self.nu

		#---------------------------------------------------------------------------------------------------
		
		
		 # global static variables
		self.FLIGHT_ALTITUDE = FLIGHT_ALTITUDE          # fgdf
		# self.RATE            = RATE                     # loop rate hz
		self.RATE            =  self.N_horizon/self.T_horizon # i think this makes sense, because dt = self.T_horizon/self.N_horizon
		self.FRAME           = REF_FRAME                # Reference frame for complex trajectory tracking
		self.CYCLE_S         = 15 #distance / self.V_des     # time to complete linear trajectory in seconds
		self.STEPS           = int( self.CYCLE_S * self.RATE ) # Total number of steps in the trajectory

		self.last_timestamp = None # make sure there is a timestamp
		

		i    = 0
		dt   = 1.0/self.RATE # = dt = self.T_horizon/self.N_horizon

		

		#--------------Obstacle parameters-----------------------------
		self.SF = 0.8 # safety factor distance from the obstcle (set as the width of the Clover)
		self.cyl_rad = 1.5 # [m] radius of the cylinder
		# Center of the cylinder location for experiment
		self.x_cyl = 6.0
		self.y_cyl = 5.1
		self.SFp = 1.2 # safety factor for prism
		self.rec_rad = 1.25 # [m] half of the longest side of the prism
		# Center of the prism for experiment
		self.x_rec = 11.5
		self.y_rec = 15.2

		#--------------------------------------------------------------
		
		
		# Publisher which will publish to the topic '/mavros/setpoint_raw/local'.
		self.publisher = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)
		
		# Subscribe to drone state
		self.state = rospy.Subscriber('mavros/state', State, self.updateState)

		# Subscribe directly to the ground truth
		self.ground_truth = rospy.Subscriber('gazebo/link_states', LinkStates, self.link_states_callback)

		 # Use the data coming in through this subscription, subscribing is faster than service, and we are using ground_truth from gazebo for better pose estimations:
		self.pose_call = rospy.Subscriber('/mavros/local_position/velocity_local',TwistStamped, self.controller)

		# Generate the array of lidar angles
		self.lidar_angles = np.linspace(-180*(math.pi/180), 180*(math.pi/180), 360) # Adjust this number based on number defined in XACRO!!!!!

		# Initialize the ranges array for the controller as no object detected (i.e. an array of inf)
		self.obs_detect = np.full_like(self.lidar_angles, np.inf)

		# Subscribe to the Lidar readings
		self.lidar = rospy.Subscriber('/ray_scan',LaserScan,self.lidar_read)

		# Set a flag, that will be used as a logic operator to adjust how the HOCBF is set, which depends on whether an obstacle is detected or not
		self.object_detect = False

		# Set this flag to calculate the ellipse shape on the first detection (cant iteraivly do it in this envirnment as the least squared
		# calculation at each iteration is too computationally intensive. It sucks up all the CPU usage and slows the lidar callback loop down)
		self.ellipse_first = False

		# This flag will be used to determine which obstacle is coming next and what pose to send to the MPC solver ACADOS
		self.obstacle_counter = 0 

		 #-----Define velocity field plot logging variables--------------
		self.count = True

		self.current_state = State()
		self.rate = rospy.Rate(20) # 20



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
			

			

			# now that the obstacle is within the lidar range, we can start calculating psi_0 and psi_1 for logging
			# i.e. we now have an estimation on the obstacles location (for real time ellipse estimation we would do this here)

	#----------- Calculate delta_p, delta_v, and delta_a-----------------------------------
			# telem = get_telemetry(frame_id='map')
			# delta_p = np.array([self.clover_pose.position.x-self.xc, self.clover_pose.position.y-self.yc])
			# delta_v = np.array([telem.vx-0, telem.vy-0]) # static obstacle assumption for now

			# # Calculate norms
			# norm_delta_p = np.linalg.norm(delta_p, ord=2)  # Euclidean norm
			# norm_delta_v = np.linalg.norm(delta_v, ord=2)  # Euclidean norm

			# # constants
			# q1 = 12#15
			# q2 = 9#10
			# r = self.SF + max(self.a_fit, self.b_fit)

			# self.psi_0 = norm_delta_p - r
			# self.psi_1 = (np.dot(delta_p, delta_v)) / norm_delta_p + q1*(self.psi_0)

			# psi_0.append(self.psi_0)
			# psi_1.append(self.psi_1)
	#--------------------------------------------------------------------------------------------

			# Log the fist velocity field update reading
			if self.count: 
				x_field[:,:] = self.XX # Assign XX to x_field, assuming XX and x_field have the same shape
				y_field[:,:] = self.YY
				u_field[:,:] = self.Vxe
				v_field[:,:] = self.Vye
				lidar_x.append(self.xa)
				lidar_y.append(self.ya)
				

				# update the flag variable (turn off so we only log the first update/obstacle reading)
				self.count = False
			
			if not self.object_detect:
				self.obstacle_counter +=1 # this only increments when an obstacle was detected once
										  # so it will be equal to 1 for the cylinder and equal to 2 for the prism
				self.object_detect = True # Update object detected flag

			# Log the Clover position and velocity at the same time we are logging this lidar data reading
			# do this so we can calculate psi_0 and psi_1 based on the ellipse readings. Hopefully will calculate this
			# in real time when we get to hardware in the lab
			telem = get_telemetry(frame_id='map')
			# Append row after row of data (to log readings)
			if self.obstacle_counter == 1:
				# append the readings of the cylinder
				xa.append(self.xa.tolist())
				ya.append(self.ya.tolist())
				x_ce.append(self.clover_pose.position.x)
				vx_ce.append(telem.vx)
				y_ce.append(self.clover_pose.position.y)
				vy_ce.append(telem.vy)
			else:
				# append the readings of the cylinder
				xap.append(self.xa.tolist())
				yap.append(self.ya.tolist())
				x_pe.append(self.clover_pose.position.x)
				vx_pe.append(telem.vx)
				y_pe.append(self.clover_pose.position.y)
				vy_pe.append(telem.vy)


			
			
			
			for j in range(self.N_horizon): # Up to N-1
				# An obstacle was detected, use the obstacle_counter number
				# to determine if it was the cylinder or the prism, and send on 
				# the values accordingly
				if self.obstacle_counter == 1:
					# This is the static cylinder, which has a radius of (1.5m??)
					# 	# Set the distance from the obstacle
					r = self.SF + self.cyl_rad
					
					self.acados_solver.set(j, "p", np.array([self.x_cyl,0,0,self.y_cyl,0,0,r])) # Assuming a static obstacle (xc,yc)

				else:
					# This is the static prism, which has a radius of (1.25m??)
					# 	# Set the distance from the obstacle
					
					r = self.SFp + self.rec_rad
					
					self.acados_solver.set(j, "p", np.array([self.x_rec,0,0,self.y_rec,0,0,r])) # Assuming a static obstacle (xc,yc)
			
			if self.obstacle_counter == 1:
				self.acados_solver.set(self.N_horizon, "p", np.array([self.x_cyl,0,0,self.y_cyl,0,0,r])) # Assuming a static obstacle (xc,yc)
			else:
				self.acados_solver.set(self.N_horizon, "p", np.array([self.x_rec,0,0,self.y_rec,0,0,r])) # Assuming a static obstacle (xc,yc)
				

		else:
			for j in range(self.N_horizon): # Up to N-1
				# an obstacle was not detected, so set the
				# 	# constraint condition to a trivial case

				# 	# Set the distance from the obstacle
				r = 2.5
				
				# 	# self.acados_solver.set(j, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
				self.acados_solver.set(j, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
			
			# 	# self.acados_solver.set(self.N_horizon, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
			self.acados_solver.set(self.N_horizon, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
			self.object_detect = False # Update object detected flag

			self.ellipse_first = False # This will ensure that when a new obstacle is detected, and it goes into
			# the if statement above, a calculation will be run once.
		

	def controller(self,data):

		current_timestamp = data.header.stamp.to_sec()

		if self.last_timestamp is not None:


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



			# #-------------------- Offline Panel Calculations---------------------------------------------------



			# Complete contributions from pre-computed grid distribution
			u = griddata((self.XX_f, self.YY_f),self.Vxe_f,(x_clover,y_clover),method='nearest') #+ self.u_inf #+ u_source #+ u_inf 'linear' is way too computationally expensive
			v = griddata((self.XX_f, self.YY_f),self.Vye_f,(x_clover,y_clover),method='nearest') #+self.v_inf#+ v_source #+self.v_inf #+ v_source #+ v_inf
			# u=0.3
			# v=0.3


			# # Complete contributions from pre-computed grid distribution
			# self.u = u
			# self.v = v

			# normalize velocities
			vec = np.array([[u],[v]],dtype=float) 
			magnitude = math.sqrt(u**2 + v**2)

			if magnitude > 0:
				norm_vel = (vec/magnitude)*self.vel_max
			else:
				norm_vel = np.zeros_like(vec)

			self.u = norm_vel[0,0]
			self.v = norm_vel[1,0]
			
			# determine the yaw
			self.omega = math.atan2(self.v,self.u)


		# Update last_timestamp for the next callback
		self.last_timestamp = current_timestamp

               

	def main(self):#,x=0, y=0, z=2, yaw = float('nan'), speed=1, frame_id ='',auto_arm = True,tolerance = 0.2):
		

		# Wait for 3 seconds
		rospy.sleep(3)
		
		# Takeoff with Clovers navigate function
		navigate(x=0.5, y=0.5, z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.5, frame_id = 'map', auto_arm = True)
		
		# Give the Clover time to reach the takeoff altitude
		rospy.sleep(15)

		# Define object that will be published
		target = PositionTarget()
		rr = rospy.Rate(self.RATE)
		k=0    # Define a counter for the reference arrays
		

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
		self.acados_solver.set(0, "lbx", x0) # Update the zero shooting node position
		self.acados_solver.set(0, "ubx", x0) # update the zero shooting node control input

		release() # stop navigate service from publishing

		while not rospy.is_shutdown():
		
			 # Trajectory publishing-----------------------------------------------
			target.header.frame_id = self.FRAME  # Define the frame that will be used
			
			target.coordinate_frame = 1 #MAV_FRAME_LOCAL_NED  # =1
			
			# Dont use velocity and position at the same time, of the Clover deviates from position then, flight issues will happen
			# because of the mismatch in position and velocity.
			target.type_mask = 8+16+32+64+128+256+2048 # Use everything! 1024-> forget yaw
			#target.type_mask =  3576 # Use only position #POSITION_TARGET_TYPEMASK_VX_IGNORE | POSITION_TARGET_TYPEMASK_VY_IGNORE | POSITION_TARGET_TYPEMASK_VZ_IGNORE | POSITION_TARGET_TYPEMASK_AX_IGNORE | POSITION_TARGET_TYPEMASK_AY_IGNORE |POSITION_TARGET_TYPEMASK_AZ_IGNORE | POSITION_TARGET_TYPEMASK_FORCE_IGNORE | POSITION_TARGET_TYPEMASK_YAW_IGNORE | POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE # = 3576
			#target.type_mask =  3520 # Use position and velocity
			#target.type_mask =  3072 # Use position, velocity, and acceleration
			#target.type_mask =  2048 # Use position, velocity, acceleration, and yaw
			# uint16 IGNORE_VX = 8 # Velocity vector ignore flags
			# uint16 IGNORE_VY = 16
			# uint16 IGNORE_VZ = 32

				
			
			# update reference
			for j in range(self.N_horizon): # Up to N-1
								
				# yref = np.array([self.xsi, self.u, self.ysi, self.v, self.FLIGHT_ALTITUDE, 0, 0, 0, 0])
				yref = np.array([0, self.u, 0, self.v, self.FLIGHT_ALTITUDE, 0, 0, 0, 0])
				
				self.acados_solver.set(j, "yref", yref)

				# if not self.object_detect:
				# 	# an obstacle was not detected, so set the
				# 	# constraint condition to a trivial case

				# 	# Set the distance from the obstacle
				# 	r = 2.5
				
				# 	# self.acados_solver.set(j, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
				# 	self.acados_solver.set(j, "p", np.array([7,0,0,7,0,0, r])) # Assuming a static obstacle
				# else:
				# 	# An obstacle was detected, therefore apply the constraints (assuming a static obstacle for now)
				# 	# Set the distance from the obstacle
				# 	r = 2.5#self.SF + max(self.a_fit, self.b_fit)

				# 	# self.acados_solver.set(j, "p", np.array([self.xc,0,0,self.yc,0,0,r])) # Assuming a static obstacle
				# 	self.acados_solver.set(j, "p", np.array([7,0,0,7,0,0, r])) # Assuming a static obstacle
					
			# Set the terminal reference state
			yref_N = np.array([0,self.u,0,self.v,self.FLIGHT_ALTITUDE,0]) # terminal components
			
			self.acados_solver.set(self.N_horizon, "yref", yref_N)
			# if not self.object_detect:
			# 	# an obstacle was not detected, so set the
			# 	# constraint condition to a trivial case
			# 	# Set the distance from the obstacle
			# 	r = 2.5
			# 	# self.acados_solver.set(self.N_horizon, "p", np.array([40,0,0,40,0,0, r])) # Assuming a static obstacle
			# 	self.acados_solver.set(self.N_horizon, "p", np.array([7,0,0,7,0,0, r])) # Assuming a static obstacle
			# else:
			# 	# An obstacle was detected, therefore apply the constraints (assuming a static obstacle for now)
			# 	# Set the distance from the obstacle
			# 	r = 2.5#self.SF + max(self.a_fit, self.b_fit)

			# 	# Assuming a static obstacle
			# 	# self.acados_solver.set(self.N_horizon, "p", np.array([self.xc,0,0,self.yc,0,0,r])) # State = [x, vx, ax, y, vy, ay]
			# 	self.acados_solver.set(self.N_horizon, "p", np.array([7,0,0,7,0,0, r])) # Assuming a static obstacle
				
			# Solve ocp
			status = self.acados_solver.solve()

			# get solution
			x0 = self.acados_solver.get(0, "x")
			u0 = self.acados_solver.get(0, "u")

			# update initial condition
			x1 = self.acados_solver.get(1, "x")
			x5 = self.acados_solver.get(5, "x")
	

			# Gather position for publishing
			target.position.x = x5[0]
			target.position.y = x5[2]
			target.position.z = self.FLIGHT_ALTITUDE
			
			# Gather velocity for publishing
			# Gather velocity for publishing
			target.velocity.x = x0[1]
			target.velocity.y = x0[3]
			target.velocity.z = 0
			
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
			# target.yaw = self.yawc[k]
			self.omega = math.atan2(x0[3],x0[1])
			target.yaw = self.omega
			
			# Gather yaw rate for publishing
			# target.yaw_rate = self.yaw_ratec[k]
			
			
			self.publisher.publish(target)

			
			# update initial condition (current pose of MPC problem)
			telem = get_telemetry(frame_id = 'map')
			x_clover = self.clover_pose.position.x
			y_clover = self.clover_pose.position.y
			z_clover = self.clover_pose.position.z
			quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
			euler_angles = euler_from_quaternion(quaternion)
			roll = euler_angles[2] #+ math.pi
			yaw = -euler_angles[0]+math.pi 
			# x_new = np.array([x_clover, x1[1], y_clover, x1[3], z_clover, x1[5]])
			x_new = np.array([x_clover, telem.vx, y_clover, telem.vy, z_clover, telem.vz])
			# self.acados_solver.set(0, "lbx", x_new) # Update the zero shooting node position
			# self.acados_solver.set(0, "ubx", x_new)
			self.acados_solver.set(0, "lbx", x1) # Update the zero shooting node position
			self.acados_solver.set(0, "ubx", x1)


			# logging/debugging
			xf.append(x_clover)
			yf.append(y_clover)
			xcom.append(x0[0])
			ycom.append(x0[2])
			# evx.append(self.u-x_clover)
			# evy.append(self.v-telem.vy)
			eyaw.append(self.omega-yaw)
			YawC.append(self.omega*(180/math.pi))
			YawF.append(yaw*(180/math.pi))
			#xdispf.append(self.d_xi)
			#ydispf.append(self.d_yi)
			velfx.append(telem.vx)
			velfy.append(telem.vy)
			velcx.append(x0[1])
			velcy.append(x0[3])
			uVPM.append(self.u)
			vVPM.append(self.v)
			# U_infx.append(self.U_inf)
			# V_infy.append(self.V_inf)
			Ux.append(u0[0])
			Uy.append(u0[1])
			Uz.append(u0[2])
			
		
			#set_position(x=posx[k], y=posy[k], z=posz[k],frame_id='aruco_map')
			#set_velocity(vx=velx[k], vy=vely[k], vz=velz[k],frame_id='aruco_map')#,yaw = yawc[i]) 
			#set_attitude(yaw = yawc[k],pitch = pitchc[k], roll = rollc[k], thrust = float('nan'),frame_id='aruco_map')
			#set_rates(yaw_rate = yaw_ratec[k],thrust = float('nan'))
		
		
			
			k = k+1
			# if k >= self.STEPS: 
			# 	navigate(x=0, y=0, z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.5, frame_id = 'map')
			# 	rospy.sleep(5)
			# 	break
			if math.sqrt((x_clover-self.xsi) ** 2 + (y_clover-self.ysi) ** 2) < 0.6: # 0.4
				set_position(frame_id='body')
				# navigate(x=x_clover,y=y_clover,z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.2, frame_id = self.FRAME)
				
				break
			
			rr.sleep()
			

		# Wait for 6 seconds
		rospy.sleep(6)
		# Perform landing
		
		land()
		 
	
		

if __name__ == '__main__':
	try:
		q=clover()
		
		q.main()#x=0,y=0,z=1,frame_id='aruco_map')

	# Define some preliminary plot limits for streamplot
		xVals  = [-1, 21];  # -0.5; 1.5                                                  # X-grid extents [min, max]
		yVals  = [-1, 21];  #-0.3;0.3                                                 # Y-grid extents [min, max]
		# Create an array of starting points for the streamlines
		# x_range = np.linspace(0, 20, int((20-0)/0.75) + 1)
		# y_range = np.linspace(0, 20, int((20-0)/0.75) + 1)
		# x_1 = np.zeros(len(y_range))
		# y_1 = np.zeros(len(x_range))

		x_range = np.linspace(2, 8, int((8-2)/0.5) + 1)
		y_range = np.linspace(2, 8, int((8-2)/0.5) + 1)
		x_1 = np.ones(len(y_range))*2
		y_1 = np.ones(len(x_range))*2
		# Convert list fields to arrays
		# x_field = np.array(x_field).reshape(2,2)
		# y_field = np.array(y_field).reshape(2,2)
		# u_field = np.array(u_field).reshape(2,2)
		# v_field = np.array(v_field).reshape(2,2)

		
		Xsl = np.concatenate((x_1, x_range))
		Ysl = np.concatenate((np.flip(y_range), y_1))
		XYsl = np.vstack((Xsl,Ysl)).T

		# Send and retrieve the ellipse, and psi_0 and psi_1 values

		w = ellipse()

		xc, yc, a_fit, b_fit, theta, psi_0, psi_1 = w.ellipse_calc(xa,ya, x_ce, vx_ce, y_ce, vy_ce, xap,yap, x_pe, vx_pe, y_pe, vy_pe)


		# debug section
		# Plot logged data for analyses and debugging
		plt.figure(1)
		plt.subplot(211)
		plt.plot(xf,yf,'r',label='pos-clover')
		plt.plot(xcom,ycom,'b',label='MPC-com')
		ellipse = Ellipse(xy=(xc[0],yc[0]), width=2*a_fit[0], height=2*b_fit[0], angle=np.rad2deg(theta[0]),
		edgecolor='b', fc='None', lw=2)
		plt.gca().add_patch(ellipse)
		# plt.plot(self.posx,self.posy,'b--',label='x-com')
		# plt.fill(xa[0],ya[0],'k') # plot first reading
		plt.legend()
		plt.grid(True)
		#plt.subplot(312)
		#plt.plot(yf,'r',label='y-fol')
		#plt.plot(ya,'b--',label='y-obs')
		#plt.legend()
		#plt.grid(True)
		#plt.ylabel('Position [m]')
		plt.subplot(212)
		plt.plot(YawF,'b',label='yaw-F')
		plt.plot(YawC,'g',label='yaw-C')
		plt.legend()
		plt.grid(True)
		plt.ylabel('yaw [deg]')
		plt.xlabel('Time [s]')
		
		# Velocity plot
		plt.figure(2)
		plt.subplot(211)
		plt.plot(velfx,'r',label='vx-vel')
		plt.plot(velcx,'b',label='vx-MPC-com')
		plt.plot(uVPM,'g',label='vx-VPM')
		plt.ylabel('vel[m/s]')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)
		plt.subplot(212)
		plt.plot(velfy,'r',label='vy-vel')
		plt.plot(velcy,'b--',label='vy-MPC-com')
		plt.plot(vVPM,'g',label='vy-VPM')
		plt.legend()
		plt.grid(True)
		plt.ylabel('Position [m]')
		# plt.subplot(313)
		# plt.plot(evx,'r',label='evx')
		# plt.plot(evy,'b',label='evy')
		# plt.plot(eyaw,'g',label='eyaw')
		# plt.ylabel('Error[m]')
		# plt.xlabel('Time [s]')
		# plt.legend()
		plt.grid(True)

		plt.figure(3)
		plt.subplot(311)
		plt.plot(Ux,'r')
		plt.legend()
		plt.grid(True)
		plt.ylabel('Ux')
		plt.subplot(312)
		plt.plot(Uy,'b')
		plt.legend()
		plt.grid(True)
		plt.ylabel('Uy')
		plt.subplot(313)
		plt.plot(Uz,'r')
		plt.legend()
		plt.grid(True)
		plt.ylabel('T_input')
		plt.xlabel('Time [s]')
		

		fig = plt.figure(4)
		plt.cla()
		np.seterr(under="ignore")
		plt.streamplot(x_field,y_field,u_field,v_field,linewidth=1.0,density=40,color='r',arrowstyle='-',start_points=XYsl) # density = 40
		plt.grid(True)
		#plt.plot(XX,YY,marker='o',color='blue')
		plt.axis('equal')
		plt.xlim(xVals)
		plt.ylim(yVals)
		plt.plot(lidar_x, lidar_y,'-o' ,color = 'k',linewidth = 0.25)
		plt.plot(xf,yf,'b',label='x-fol') # path taken by clover
		ellipse = Ellipse(xy=(xc[0],yc[0]), width=2*a_fit[0], height=2*b_fit[0], angle=np.rad2deg(theta[0]),
		edgecolor='b', fc='None', lw=2)
		plt.gca().add_patch(ellipse)
		plt.xlabel('X Units')
		plt.ylabel('Y Units')
		plt.title('Streamlines with Stream Function Velocity Equations')

		plt.figure(5)
		plt.plot(psi_0,'r',label='psi_0')
		plt.plot(psi_1,'b',label='psi_1')
		plt.ylabel('b(x)')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)

		plt.figure(6)
		plt.plot(xc, yc, 'ro', label='ellipse-center', markersize=3)  # 'ro' for red circles
		plt.ylabel('center[m]')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)

		plt.figure(7)
		plt.plot(psi_0p,'r',label='psi_0p')
		plt.plot(psi_1p,'b',label='psi_1p')
		plt.ylim(-2, 30)
		plt.ylabel('b(x)')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)

		plt.figure(8)
		plt.plot(xcp, ycp, 'ro', label='ellipse-center', markersize=3)  # 'ro' for red circles
		plt.ylabel('center[m]')
		plt.xlabel('Time [s]')
		plt.legend()
		plt.grid(True)

		plt.show()
		
	except rospy.ROSInterruptException:
		pass

