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

from panel_functions import CLOVER_COMPONENTS, CLOVER_STREAM_GEOMETRIC_INTEGRAL, CLOVER_KUTTA, CLOVER_STREAMLINE, CLOVER_noOBSTACLE

from scipy.interpolate import griddata

import h5py  # use for data logging of larges sets of data

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
xf = []  # This gathers the 
yf = []
xdispf=[]
ydispf=[]
xa = []
ya = []
YawL = []
YawF = []
YawC = []

# readings of modified/extended obstacle
xa = []
ya = []
# readings of the non-modified/original obstacle
xa_orig = []
ya_orig = []

# updated velocity field plot
nGridX = 25  # 20                                                         # X-grid for streamlines and contours
nGridY = 25  # 20                                                       # Y-grid for streamlines and contours
x_field = np.zeros((nGridX, nGridY))# np.zeros((30, 30))
y_field = np.zeros((nGridX, nGridY)) # np.zeros((30, 30))
u_field = np.zeros((nGridX, nGridY))
v_field = np.zeros((nGridX, nGridY))
lidar_x = []
lidar_y = []

# trail edge kutta condition
trail_x = []
trail_y = []

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

# Create a HDF5 file name
# Open the HDF5 file globally
file_name = 'test4.h5'
 # Open the HDF5 file for writing
with h5py.File(file_name, 'a') as hf:


			
    # This class categorizes all of the functions used for complex rajectory tracking
    class clover:

        def __init__(self, FLIGHT_ALTITUDE, RATE, RADIUS, CYCLE_S, REF_FRAME): 


            # Define the sink location and strength
            self.g_sink = 2.00
            self.xsi = -0.45
            self.ysi = 0.9

            # Define the source strength and location
            self.g_source = 0.8
            self.xs = 0.8
            self.ys = -1.5
            # self.xs = 5
            # self.ys = 2.5


            # Define the source strength placed on the actual drone
            self.g_clover = 0.0

            # Could possible just put a source on the drone and not bother with the two different sources


            # Free flow constant
            self.U_inf = 0
            self.V_inf = 0

            self.alpha = 70*(math.pi/180)



            # Kutta condition flag (decide which one to use)
            self.flagKutta = np.array([0, 0, 0,1])
            # position 0 is for smooth flow off edge for closed object
            # position 1 is for smooth flow off ege for closed object using point extended into space a small amount
            # position 2 is for smooth flow off the edge of a non closed i.e. sail detection with lidar
            # position 3 is for smooth flow off an extended edge of a non closed i.e. sail detection with lidar. Adds safety factor to direct flow away from object



            #---------------------------------------------------------------------------------------------------

            # Set a flag, that will be used as a logic operator to turn the velocity field updator on/off
            # depending on whether the object is detected or not
            self.object_detect = False
            self.timer = None  # To store the timer object, this is used to turn the timer object off and on 
            self.velocity_update_started = False # This is used to avoid constantly trying to start the time at each iteration an object is detected


            # Set a flag that only starts the velocity update when an obstacle is first detected (not continuously)
            self.velocity_update_started = False

            # iNITIALIZE INPUT VELOCITY VALUES
            self.u = 0
            self.v = 0

            # Define the max velocity allowed for the Clover
            self.vel_max = 0.15 # [m/s]

            # #-------------------- Offline Panel Calculations---------------------------------------------------

            # Subscribe directly to the ground truth
            self.ground_truth = rospy.Subscriber('/mocap/clover_pose', PoseStamped, self.link_states_callback)

            # An object was not detected yet, so use the source and sink for velocity based navigation

            ## Compute Streamlines with stream function velocity equation
            # Too many gridpoints is not good, it will cause the control loop to run too slow
            # Grid parameters
            self.nGridX = 25;  # 20 is good                                                         # X-grid for streamlines and contours
            self.nGridY = 25;  # 20 is good                                                    # Y-grid for streamlines and contours
            self.xVals  = [-1, 1.5];  # ensured it is extended past the domain incase the clover leaves domain             # X-grid extents [min, max]
            self.yVals  = [-1.5, 1.5];  #-0.3;0.3                                                 # Y-grid extents [min, max]

            # Define Lidar range (we will set parameters to update grid resolution within detection range):
            self.lidar_range = 3.5 # [m]
            self.lidar_resolution = 0.5 # grid resolution within the lidar ranges

            # Streamline parameters
            stepsize = 0.1;   #0.01                                                     # Step size for streamline propagation
            maxVert  = self.nGridX*self.nGridY*100;                                           # Maximum vertices
            slPct    = 25;                                                          # Percentage of streamlines of the grid


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

            #-------------External LOG------------------
            # Create a group to store velocity field for thisiteration
            iteration_group = hf.create_group('Initial_Field')
            iteration_group.create_dataset('XX', data=self.XX)
            iteration_group.create_dataset('YY', data=self.YY)
            iteration_group.create_dataset('Vxe', data=self.Vxe)
            iteration_group.create_dataset('Vye', data=self.Vye)
            #------------------------------------------

    # -------------------------------------------------------------------------------------------------------------------------------

            # global static variables
            self.FLIGHT_ALTITUDE = FLIGHT_ALTITUDE          # fgdf
            self.RATE            = RATE                     # loop rate hz
            self.FRAME           = REF_FRAME                # Reference frame for complex trajectory tracking

            self.last_timestamp = None # make sure there is a timestamp


            # Publisher which will publish to the topic '/mavros/setpoint_raw/local'. This has a PositionTarget message type: link
            self.publisher = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)



            # Use the data coming in through this subscription, subscribing is faster than calling a service, and we are using ground_truth from gazebo for better pose estimations:
            self.pose_call = rospy.Subscriber('/mavros/local_position/pose',PoseStamped, self.controller)


            # Generate the array of lidar angles
            self.lidar_angles = np.linspace(-180*(math.pi/180), 180*(math.pi/180), 360) # Adjust this number based on number defined in XACRO!!!!!

            # Initialize the ranges array for the controller as no object detected (i.e. an array of inf)
            self.obs_detect = np.full_like(self.lidar_angles, np.inf)

            # Subscribe to the Lidar readings
            self.lidar = rospy.Subscriber('/scan',LaserScan,self.lidar_read)


            #-----Define velocity field plot logging variables--------------
            self.count = True



            #--------------------------------------------------------------



            # Define object to publish for the follower (if not using mocap or central system)
            self.follow = PositionTarget()

            self.follow.header.frame_id = 'map'  # Define the frame that will be used

            self.follow.coordinate_frame = 1 #MAV_FRAME_BODY_NED  # =8

            self.follow.type_mask = 0  # use everything!
            # PositionTarget::IGNORE_VX +
            # PositionTarget::IGNORE_VY +
            # PositionTarget::IGNORE_VZ +
            # PositionTarget::IGNORE_AFX +
            # PositionTarget::IGNORE_AFY +
            # PositionTarget::IGNORE_AFZ +
            # PositionTarget::IGNORE_YAW;

            self.current_state = State()
            self.rate = rospy.Rate(20)

        def euler_to_rotation_matrix(self, roll, pitch, yaw):
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

        def link_states_callback(self, msg):

            self.clover_pose = msg.pose



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
            # Need to add a lidar detection threshold for panel method case (4), so if we have like 1-2 or low detection we could get singular matrix
            if sum(not np.isinf(range_val) for range_val in self.obs_detect) >= 4 and z_clover >= 1.0: # want the drone to be at some altitude so we are not registering ground detections
            # if any(not np.isinf(range_val) for range_val in self.obs_detect):

                # The angles and ranges start at -180 degrees i.e. at the right side, then go counter clockwise up to the top i.e. 180 degrees
                self.ranges = data.ranges

                angles = self.lidar_angles


                # Convert ranges to a NumPy array if it's not already
                self.ranges = np.array(self.ranges)
                ranges_orig = np.array(self.ranges) # store the unmodified ranges


                # Polar to Cartesion transformation for all readings (assuming angles are in standard polar coordinates) y-axis is left and x-axis is directly forward.
                self.x_local = self.ranges*np.cos(angles)
                y_local = self.ranges*np.sin(angles)

                # This transforms to the local Lidar frame where the y-axis is forward
                # and the x-axis is pointing right:
                # x_local = ranges*np.sin(angles)
                # x_local = np.multiply(x_local,-1)
                # y_local = ranges*np.cos(angles)

                x_local_orig = ranges_orig*np.cos(angles)
                y_local_orig = ranges_orig*np.sin(angles)

                #---------------- Safety factor-----------------------------------------------
                # put a safety factor on the detected obstacle
                # Reduce the range by a scaling factor beta for each real range (set as diameter of the clover)
                beta = 2.0 # Scale object and shift
                # Combine xdata and ydata into a single array of points
                points = np.column_stack((self.x_local, y_local))

                # Find the point closest to the origin
                min_distance_index = np.argmin(np.linalg.norm(points, axis=1))
                closest_point = points[min_distance_index]

                # Step 2: Shift all points so that the closest point becomes the origin
                shifted_points = points - closest_point

                # Step 3: Scale all points by a factor beta
                scaled_points = shifted_points * beta

                # Step 4: Shift all points back to their original positions
                final_points = scaled_points + closest_point

                # Calculate the distance to move the closest point
                desired_distance = 0.5

                # Calculate the current distance to the origin for the closest point
                current_distance = np.linalg.norm(closest_point)

                # Calculate the unit vector in the direction of the closest point
                unit_vector = closest_point / current_distance

                # Calculate the new position for the closest point
                new_closest_point = unit_vector * (current_distance - desired_distance)

                # Calculate the difference vector
                shift_vector = closest_point - new_closest_point

                # Shift all points including the closest point
                shifted_points = final_points - shift_vector


                # translate the shape equally to the origin (To clover)
                self.x_local = shifted_points[:,0]
                y_local = shifted_points[:,1]


                #---------------------------------------------------------------------------------

                #------------------2D transformations--------------------
                # Homogenous transformation matrix for 2D
                R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) # rotation matrix
                T = np.vstack([np.hstack([R, np.array([[x_clover], [y_clover]])]),[0,0,1]]) # Homogenous transformation matrix

                # Lidar readings in homogenous coordinates
                readings_local = np.vstack([self.x_local, y_local, np.ones_like(self.x_local)])
                readings_local_orig = np.vstack([x_local_orig, y_local_orig, np.ones_like(x_local_orig)])

                # Transform all lidar readings to global coordinates
                self.readings_global = np.dot(T, readings_local)
                readings_global_orig = np.dot(T, readings_local_orig)

                # Extract the tranformed positions
                self.readings_global = self.readings_global[:2,:].T
                self.readings_global_orig = readings_global_orig[:2,:].T

                #-------------------Velocity field update logic----------------------------------
                self.handle_object_detection() # start the velocity field updating now that an obstacle is detected

            else:

                self.handle_object_loss() # We handle the case where the lidar detect is not happening or doesnt verify the 
                # requirements for the above if statement, in this function, if an object has not been detected yet, it will be skipped,
                # if an object was detected and lost, it will activate the logic inside


            # # Dont want to update these here!! we are using in controller and if the values are updated mid calculation for velocity itll cause many issues
            # # This is updating faster then the velocity controller calculations
            # self.xa = readings_global[:,0].T
            # self.ya = readings_global[:,1].T

            
    #--------------------Define Velocity field update logic functions---------------------------------------------------------------------

        def start_velocity_update(self):
            if not self.velocity_update_started:
                # Begin the velocity field updated now that an obstacle has been detected
                self.timer = rospy.Timer(rospy.Duration(0.75), self.velocity_field_update)
                self.velocity_update_started = True # update this so it doesnt keep starting (only starts when a new obstacle is detected)

        def stop_velocity_update(self):
            if self.timer is not None:
                self.timer.shutdown()  # Stop the timer if it exists
                self.timer = None  # Reset the timer object
                self.velocity_update_started = False # update the timer update status

        def handle_object_detection(self):
            if not self.object_detect:
                # Object detected for the first time
                self.object_detect = True
                self.start_velocity_update()
                # Object was previously detected, no need to take further action

        def handle_object_loss(self):
            if self.object_detect: # Object was detected initially, but was lost according to the else statement in the lidar_function!!!
            # Object is no longer detected

                x_clover = self.clover_pose.position.x
                y_clover = self.clover_pose.position.y
                z_clover = self.clover_pose.position.z
                quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
                euler_angles = euler_from_quaternion(quaternion)
                roll = euler_angles[2] 
                yaw = -euler_angles[0]+math.pi #-PI_2
                pitch = euler_angles[1]

                # # Re-Generate the grid points without modifications from detected obstacle
                # Xgrid = np.linspace(self.xVals[0], self.xVals[1], self.nGridX)
                # Ygrid = np.linspace(self.yVals[0], self.yVals[1], self.nGridY)
                # self.XX, self.YY = np.meshgrid(Xgrid, Ygrid)

                # self.Vxe = np.zeros((self.nGridX, self.nGridY))
                # self.Vye = np.zeros((self.nGridX, self.nGridY))


                self.object_detect = False
                self.stop_velocity_update()
                # Recalculate velocity field for the case without an obstacle
                # An object was not detected yet, so use the source and sink for velocity based navigation
                for m in range(self.nGridX):
                    for n in range(self.nGridY):
                        XP, YP = self.XX[m, n], self.YY[m, n]
                        u, v = CLOVER_noOBSTACLE(XP, YP, self.U_inf, self.V_inf, self.xs, self.ys, self.xsi, self.ysi, self.g_source, self.g_sink, self.g_clover, x_clover, y_clover)

                        self.Vxe[m, n] = u
                        self.Vye[m, n] = v

                # Flatten the grid point matices and velocity matrices into vectory arrays for the griddata function
                # Update the velocity field:
                self.XX_f = self.XX.flatten()
                self.YY_f = self.YY.flatten()
                self.Vxe_f = self.Vxe.flatten()
                self.Vye_f = self.Vye.flatten()




        def velocity_field_update(self, event=None):

            # this needs to run periodically to update the velocity field. But not too much as it is very compuationally expensive, maybe run as a thread or something

            # Get current state of this follower 
            # telem = get_telemetry(frame_id='map')
            x_clover = self.clover_pose.position.x
            y_clover = self.clover_pose.position.y
            z_clover = self.clover_pose.position.z
            quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
            euler_angles = euler_from_quaternion(quaternion)
            roll = euler_angles[2] #+ math.pi
            yaw = -euler_angles[0]+ math.pi
            pitch = euler_angles[1]

            if not self.object_detect:
            # Object not detected, no need to update velocity field
                return


            print('tic')

            # Update the lidar detection readings
            self.xa = self.readings_global[:,0].T
            self.ya = self.readings_global[:,1].T

            self.xa_orig = self.readings_global_orig[:,0].T
            self.ya_orig = self.readings_global_orig[:,1].T

            # Filter every second reading i.e. take every second reading. We want to reduce the computational load on the panel method
            # So take 180 readings over 360degree span instead of 360 readings
            # self.xa = self.xa[::2]
            # self.ya = self.ya[::2]

            # Filter out the inf values in the data point arrays
            self.xa = self.xa[np.isfinite(self.xa)]
            self.ya = self.ya[np.isfinite(self.ya)]
            self.xa_orig = self.xa_orig[np.isfinite(self.xa_orig)]
            self.ya_orig = self.ya_orig[np.isfinite(self.ya_orig)]
            # Keep adding the points into one linear array
            # xa.extend(self.xa)
            # ya.extend(self.ya)

            # Replace with artificial obstacle to test if lidar is working
            # theta1 = np.linspace(270 * np.pi / 180, 135 * np.pi / 180, 30) # CCW rotation
            # self.xa = 9 + 1 * np.cos(theta1)
            # self.ya = 9 + 1 * np.sin(theta1)


            # Append row after row of data (to log readings)
            xa.append(self.xa.tolist())
            ya.append(self.ya.tolist())
            xa_orig.append(self.xa_orig.tolist())
            ya_orig.append(self.ya_orig.tolist())

            # Upate the number of panels
            self.n = len(self.xa)-1

            # in this case an obstacle was detected so apply panel method navigation

            # Get current state of this follower 
            # telem = get_telemetry(frame_id='map')

            # #-------------------- Offline Panel Calculations---------------------------------------------------

            #This function calculates the location of the control points as well as the
            #right hand side of the stream function equation:

            [xmid, ymid, dx, dy, Sj, phiD, rhs] = CLOVER_COMPONENTS(self.xa, self.ya, self.U_inf, self.V_inf, self.g_source, self.g_sink, self.xs, self.ys, self.xsi, self.ysi, self.n, self.g_clover, x_clover, y_clover)



            # Convert angles from [deg] to [rad]
            phi = np.deg2rad(phiD)  # Convert from [deg] to [rad]



            # Evaluate gemoetric integral matrix without the kutta condition equation
            I = CLOVER_STREAM_GEOMETRIC_INTEGRAL(xmid, ymid, self.xa, self.ya, phi, Sj, self.n)

#-----------------------extended kutta condition---------------------------------------------------------
            # Extended point off of the end of the object for kutta condition
            # Calculate the extended edge point for the sail extended kutta condition
            ext_dist = 1.0


            finite_indices = np.where(np.isfinite(self.x_local))[0] # find where the indices are finite in the clover/local reference frame (this is being updated in the lidar function)
            ang = self.lidar_angles[finite_indices] # select the angles that are finite readings

            # Step 3: Compute the sums for left and right sides using array operations
            left_sum = np.sum(ang[ang > 0])
            right_sum = np.sum(ang[ang < 0])

            # Determine which side has the obstacle closer
            if left_sum > abs(right_sum):
                # print("Obstacle is more to the left.")
                #---------Object is more to the left of the clover when detecting-------------
                # this is intuitively backwards, you would think you would have to extend the kutta off the right side of the object
                # if the object was more to the left of the clover so we could go around the irght side of the object.

                directionVect = [self.xa[-1] - self.xa[-2], self.ya[-1] - self.ya[-2]] # off the end of the clockwise ending panel
                directionVect = directionVect / np.linalg.norm(directionVect)
                # normalize the direction vector
                
                # Calculate the coordinates of the extended point in the general frame
                extendedX = self.xa[-1] + ext_dist*directionVect[0]
                extendedY = self.ya[-1] + ext_dist*directionVect[1]

                #---------Log intuitive trail point for the plots in paper-------------
                q  = [self.xa[0] - self.xa[1], self.ya[0] - self.ya[1]] 
                q  = q / np.linalg.norm(q)

                int_Y = self.ya[0] + ext_dist*q[1]
                int_X = self.xa[0] + ext_dist*q[0]

                trail_intuitive = [int_X, int_Y]

            elif left_sum < abs(right_sum):
                # print("Obstacle is more to the right.")
                #-----------Object is more to the right of the clover when detecting------------
            # This is intuitively backwards, as you would think this would be the case for when the object is detected more in the left half 
            # of the clover...(not sure why this works this way). This kutta condition will have the flowlines go off the left side of the object
                directionVect = [self.xa[0] - self.xa[1], self.ya[0] - self.ya[1]] # off the end of the CCW/right ending panel
                directionVect = directionVect / np.linalg.norm(directionVect)
                extendedY = self.ya[0] + ext_dist*directionVect[1]
                extendedX = self.xa[0] + ext_dist*directionVect[0]

                #---------Log intuitive trail point for the plots in paper-------------
                q  = [self.xa[-1] - self.xa[-2], self.ya[-1] - self.ya[-2]]
                q  = q / np.linalg.norm(q)

                int_Y = self.ya[-1] + ext_dist*q[1]
                int_X = self.xa[-1] + ext_dist*q[0]

                trail_intuitive = [int_X, int_Y]
            else:
                # print("Obstacle is centered.")
                 
                directionVect = [self.xa[0] - self.xa[1], self.ya[0] - self.ya[1]] # off the end of the CCW/right ending panel
                directionVect = directionVect / np.linalg.norm(directionVect)
                extendedY = self.ya[0] + ext_dist*directionVect[1]
                extendedX = self.xa[0] + ext_dist*directionVect[0]

                #---------Log other one-------------
                q  = [self.xa[-1] - self.xa[-2], self.ya[-1] - self.ya[-2]]
                q  = q / np.linalg.norm(q)

                int_Y = self.ya[-1] + ext_dist*q[1]
                int_X = self.xa[-1] + ext_dist*q[0]

                trail_intuitive = [int_X, int_Y]

            
           
            # coordinates of the translated point off the trailing edge
            trail_point = [extendedX, extendedY]
#---------------------------------------------------------------------------------------------------------------------------------------------
            # Form the last line of the system of equations with the kutta condition 
            [I, rhs] = CLOVER_KUTTA(I, trail_point, self.xa, self.ya, phi, Sj, self.n, self.flagKutta, rhs, self.U_inf, self.V_inf, self.xs, self.ys, self.xsi, self.ysi, self.g_source, self.g_sink, self.g_clover, x_clover, y_clover)

            # calculating the vortex density (and stream function from kutta condition)
            # by solving linear equations given by
            g = np.linalg.solve(I, rhs.T)  # = gamma = Vxy/V_infinity for when we are using U_inf as the free flow magnitude
            # broken down into components (using cos(alpha), and sin(alpha)), and free flow is the only other RHS
            # contribution (no source or sink). It is equal to Vxy 
            # when we define the x and y (U_inf and V_inf) components seperately.

            ## Compute Streamlines with stream function velocity equation
            # Too many gridpoints is not good, it will cause the control loop to run too slow

            #-----Update meshgrid on and around obstacle for improved plotting-------------------------------------------
            
            # # Concatenate XX with xa and xa_orig
            # XX_new = np.concatenate((self.XX[0, :], self.xa_orig[::5]))#self.xa, self.xa_orig))

            # # Concatenate YY with ya and ya_orig
            # YY_new = np.concatenate((self.YY[:, 0], self.ya_orig[::5]))#self.ya, self.ya_orig))

            # # Sort XX_new and YY_new individually
            # XX_sorted = np.sort(XX_new)
            # YY_sorted = np.sort(YY_new)

            # # Create mesh grids from the sorted XX_sorted and YY_sorted
            # self.XX, self.YY = np.meshgrid(XX_sorted, YY_sorted)

            # # Re-generated the velocity field grid size
            # self.Vxe = np.zeros((max(XX_sorted.shape), max(YY_sorted.shape)))
            # self.Vye = np.zeros((max(XX_sorted.shape), max(YY_sorted.shape)))
           

            #---------------------------------------------------------------

            # Path to figure out if grid point is inside polygon or not
            AF = np.vstack((self.xa,self.ya)).T
            AF_orig = np.vstack((self.xa_orig,self.ya_orig)).T
		    #print(AF)
            afPath = path.Path(AF)
            afPath_orig = path.Path(AF_orig)

            for m in range(self.nGridX):
                for n in range(self.nGridY):
                    XP, YP = self.XX[m, n], self.YY[m, n]
                    # XP, YP = self.X_mesh[m, n], self.Y_mesh[m, n]
                    # Check if the current grid point corresponds to (xa, ya) or (xa_orig, ya_orig)
                    if  afPath_orig.contains_points([[XP,YP]]):#afPath.contains_points([[XP,YP]]) or afPath_orig.contains_points([[XP,YP]]):
                        self.Vxe[m, n] = 0
                        self.Vye[m, n] = 0

                    else:
                        u, v = CLOVER_STREAMLINE(XP, YP, self.xa, self.ya, phi, g, Sj, self.U_inf, self.V_inf, self.xs, self.ys, self.xsi, self.ysi, self.g_source, self.g_sink, self.g_clover, x_clover, y_clover)
                    # print(u)

                        
                        self.Vxe[m, n] = u

                        self.Vye[m, n] = v

            # Log the fist velocity field update reading
            if self.count: 
                x_field[:,:] = self.XX # Assign XX to x_field, assuming XX and x_field have the same shape
                y_field[:,:] = self.YY
                u_field[:,:] = self.Vxe
                v_field[:,:] = self.Vye
                lidar_x.append(self.xa)
                lidar_y.append(self.ya)

                trail_x.append(extendedX)
                trail_y.append(extendedY)

                # Create a group to store velocity field for thisiteration
                iteration_group = hf.create_group(f'iteration_{1}')
                iteration_group.create_dataset('Vxe', data=self.Vxe)
                iteration_group.create_dataset('Vye', data=self.Vye)


                # update the flag variable (turn off so we only log the first update/obstacle reading)
                self.count = False


            print('toc')
            # Flatten the grid point matices and velocity matrices into vectory arrays for the griddata function
            # Update the velocity field:
            self.XX_f = self.XX.flatten()
            self.YY_f = self.YY.flatten()
            self.Vxe_f = self.Vxe.flatten()
            self.Vye_f = self.Vye.flatten()

            #-------------External LOG------------------
            # Create a group to store velocity field for this iteration/time
            iteration_group = hf.create_group(f'Field_update_iteration_{self.lidar_timestamp}')
            iteration_group.create_dataset('XX', data=self.XX)
            iteration_group.create_dataset('YY', data=self.YY)
            iteration_group.create_dataset('Vxe', data=self.Vxe)
            iteration_group.create_dataset('Vye', data=self.Vye)
            iteration_group.create_dataset('xa_extend', data=self.xa)
            iteration_group.create_dataset('ya_extend', data=self.ya)
            iteration_group.create_dataset('xa_orig', data=self.xa_orig)
            iteration_group.create_dataset('ya_orig', data=self.ya_orig)
            # Log the trail edge kutta condition point (extended off of the extended readings, in the global frame)
            iteration_group.create_dataset('x_trail', data=extendedX)
            iteration_group.create_dataset('y_trail', data=extendedY)
            # log the intuitive position of the trail point (where you think it should be for the results that come from it)
            iteration_group.create_dataset('x_trail_intuitive', data=int_X)
            iteration_group.create_dataset('y_trail_intuitive', data=int_Y)
            # log the current clover position as well for plotting marker location on map plot
            iteration_group.create_dataset('x_clover_cur', data=self.clover_pose.position.x)
            iteration_group.create_dataset('y_clover_cur', data=self.clover_pose.position.y)

            #------------------------------------------

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
                u = griddata((self.XX_f, self.YY_f),self.Vxe_f,(x_clover,y_clover),method='linear') #+ self.u_inf #+ u_source #+ u_inf
                v = griddata((self.XX_f, self.YY_f),self.Vye_f,(x_clover,y_clover),method='linear') #+self.v_inf#+ v_source #+self.v_inf #+ v_source #+ v_inf



                # Complete contributions from pre-computed grid distribution
                self.u = u
                self.v = v

                # normalize velocities
                vec = np.array([[self.u],[self.v]],dtype=float) 
                magnitude = math.sqrt(self.u**2 + self.v**2)

                if magnitude > 0:
                    norm_vel = (vec/magnitude)*self.vel_max
                else:
                    norm_vel = np.zeros_like(vec)

                self.u = norm_vel[0,0]
                self.v = norm_vel[1,0]
                #print(self.u)
                # determine the yaw
                self.omega = math.atan2(self.v,self.u)


            # Update last_timestamp for the next callback
            self.last_timestamp = current_timestamp


        def main(self):

            # Wait for 3 seconds
            rospy.sleep(3)
            # Takeoff to a desired altitude # x=0.2, y=2
            navigate(x=0.75,y=-1.4,z = self.FLIGHT_ALTITUDE, frame_id='map',auto_arm=True) # drone seems to be unstable when I set frame to map... I think there is something wrong with lidar physical values

            # Give the Clover time to reach the takeoff altitude
            rospy.sleep(15)

            # Define object that will be published
            target = PositionTarget()
            rr = rospy.Rate(self.RATE)

            release() # stop navigate service from publishing before beginning the figure-8 publishing

            while not rospy.is_shutdown():
                # Trajectory publishing-----------------------------------------------
                target.header.frame_id = self.FRAME  # Define the frame that will be used

                target.coordinate_frame = 1 #MAV_FRAME_LOCAL_NED  # =1

                target.type_mask = 1+2+64+128+256+2048  # Use everything!
                # PositionTarget::IGNORE_VX +
                # PositionTarget::IGNORE_VY +
                # PositionTarget::IGNORE_VZ +
                # PositionTarget::IGNORE_AFX +
                # PositionTarget::IGNORE_AFY +
                # PositionTarget::IGNORE_AFZ +
                # PositionTarget::IGNORE_YAW;

                # Gather position for publishing
                #target.position.x = posx[k]
                #target.position.y = posy[k]
                target.position.z = self.FLIGHT_ALTITUDE

                # Gather velocity for publishing
                target.velocity.x = self.u
                target.velocity.y = self.v
                target.velocity.z = 0

                # Gather acceleration for publishing
                target.acceleration_or_force.x = 0
                target.acceleration_or_force.y = 0
                target.acceleration_or_force.z = 0

                # Gather yaw for publishing
                target.yaw = self.omega

                # Gather yaw rate for publishing
                #target.yaw_rate = yaw_ratec[k]

                # Publish to the setpoint topic


                self.publisher.publish(target)



                # Get current state of this follower 
                telem = get_telemetry(frame_id='map') # should be fairly accurate now that I am publishing ground truth data to EKF, so the velocity estimation must be decent

                x_clover = self.clover_pose.position.x
                y_clover = self.clover_pose.position.y
                z_clover = self.clover_pose.position.z
                quaternion = [self.clover_pose.orientation.w,self.clover_pose.orientation.x, self.clover_pose.orientation.y, self.clover_pose.orientation.z ]
                euler_angles = euler_from_quaternion(quaternion)
                roll = euler_angles[2] #+ math.pi
                yaw = euler_angles[0]
                pitch = euler_angles[1]

                # logging/debugging
                xf.append(x_clover)
                yf.append(y_clover)
                # evx.append(self.u-x_clover)
                # evy.append(self.v-telem.vy)
                eyaw.append(self.omega-yaw)
                YawC.append(self.omega*(180/math.pi))
                YawF.append(yaw*(180/math.pi))
                #xdispf.append(self.d_xi)
                #ydispf.append(self.d_yi)
                velfx.append(telem.vx)
                velfy.append(telem.vy)
                velcx.append(self.u)
                velcy.append(self.v)
                # U_infx.append(self.U_inf)
                # V_infy.append(self.V_inf)


                if math.sqrt((x_clover-self.xsi) ** 2 + (y_clover-self.ysi) ** 2) < 0.6: # 0.4
                    # release()
                    navigate(x=self.xsi,y=self.ysi,z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.2, frame_id = self.FRAME)
                    # navigate(x=0,y=0,z=self.FLIGHT_ALTITUDE, yaw=float('nan'), speed=0.2, frame_id = self.FRAME)
                    # navigate(x=0,y=0,z=0, yaw=float('nan'), frame_id = 'body')
                    # set_position(frame_id='body') # something weight going on with release and navigate functon, no matter what I put the navigate function
                    # to here, it has a moment of time where it repeated what was set for lift off so starts going back to the origin. So the release function 
                    # seems to pause it instead of shut it off completely. Therefore use the set_position function to hover on the spot (something different then 
                    # navigate because if I call navigate, it resumes what was paused which jolts it back to the origin, before actually reading the command in 
                    # this code block).
                    
                    break
                rr.sleep()
                # rospy.spin()


            # Wait for 3 seconds
            rospy.sleep(6)
            # Perform landing

            land()

    if __name__ == '__main__':
        try:
            # Define the performance parameters here which starts the script
            q=clover(FLIGHT_ALTITUDE = 0.3, RATE = 50, RADIUS = 2.0, CYCLE_S = 13, REF_FRAME = 'map')
            
            q.main()

            #-------------External LOG------------------
            # Create a group to store velocity field for this iteration/time
            iteration_group = hf.create_group('Control_log')
            iteration_group.create_dataset('x_clover', data=xf)
            iteration_group.create_dataset('y_clover', data=yf)
            iteration_group.create_dataset('yaw_clover', data=YawF)
            iteration_group.create_dataset('yaw_com', data=YawC)
            iteration_group.create_dataset('yaw_error', data=eyaw)
            iteration_group.create_dataset('velx_clover', data=velfx)
            iteration_group.create_dataset('vely_clover', data=velfy)
            iteration_group.create_dataset('velx_com', data=velcx)
            iteration_group.create_dataset('vely_com', data=velcy)
                #------------------------------------------
            #print(xa)
            #print(xf)

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
            
            # Plot logged data for analyses and debugging
            plt.figure(1)
            plt.subplot(211)
            plt.plot(xf,yf,'r',label='x-fol')
            #plt.plot(xa,'b--',label='x-obs')
            plt.fill(xa[0],ya[0],'k') # plot first reading
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
            plt.subplot(311)
            plt.plot(velfx,'r',label='vx-vel')
            plt.plot(velcx,'b',label='vx-com')
            plt.ylabel('vel[m/s]')
            plt.xlabel('Time [s]')
            plt.legend()
            plt.grid(True)
            plt.subplot(312)
            plt.plot(velfy,'r',label='vy-vel')
            plt.plot(velcy,'b--',label='vy-com')
            plt.legend()
            plt.grid(True)
            plt.ylabel('Position [m]')
            plt.subplot(313)
            plt.plot(evx,'r',label='evx')
            plt.plot(evy,'b',label='evy')
            plt.plot(eyaw,'g',label='eyaw')
            plt.ylabel('Error[m]')
            plt.xlabel('Time [s]')
            plt.legend()
            plt.grid(True)

            plt.figure(3)
            for x_row, y_row in zip(xa, ya):
                plt.plot(x_row,y_row, '-o',label=f'Reading {len(plt.gca().lines)}')
                #plt.fill(xa,ya,'k')
            plt.grid(True)
            plt.legend()
            

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
            plt.plot(trail_x, trail_y, 'o')
            plt.xlabel('X Units')
            plt.ylabel('Y Units')
            plt.title('Streamlines with Stream Function Velocity Equations')

            plt.figure(5)
            plt.scatter(x_field,y_field,color = 'blue', label='data-points')
            plt.xlabel('x-data')
            plt.ylabel('y-data')
            plt.grid(True)
            plt.legend()

            # Plot the airfoil
            plt.figure(6)
            plt.plot(lidar_x, lidar_y, 'ko-')
            plt.plot(trail_x, trail_y, 'o')
            plt.xlim() #auto
            plt.ylim() # auto
            plt.xlabel('X Units')
            plt.ylabel('Y Units')
            plt.title('Airfoil')
            plt.axis('equal')
            plt.grid(True)

            plt.figure(7)
            plt.scatter(u_field,v_field,color = 'blue', label='data-points')
            plt.xlabel('vx-data')
            plt.ylabel('vy-data')
            plt.grid(True)
            plt.legend()


            
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
