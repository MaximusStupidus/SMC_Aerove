from re import U
from turtle import Vec2D
import rospy
import time
from math import sin, cos, tan, sqrt,pi
import numpy as np
from numpy import arcsin, array, dtype, float256, roll, sign, sin as S
from numpy import tan as T
from numpy import cos as C

from math import factorial as f
from scipy.linalg import block_diag
from qpsolvers import solve_qp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
from scipy.optimize import Bounds,minimize
from constrained_time_opt_new import min_snap
from std_msgs.msg import String, Float64, Int16
from sensor_msgs.msg import NavSatFix, Image,Imu
from mavros_msgs.srv import CommandTOL, SetMode, CommandBool
from mavros_msgs.msg import AttitudeTarget
from geometry_msgs.msg import PoseStamped, Pose, Point, Twist, TwistStamped
import math
from time import sleep
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# p=[x,y,z]T position
# η = [φ, θ, ψ]T euler angles
# ξ = [p, q, r]T angular velocites
# υ = [υx,υy,υz]T linear velocities


class Drone:
	
	def __init__(self):
		self.X=np.array([
			# x0, y1, z2, phi3, theta4, psi5, 
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			# x_dot6, y_dot7, z_dot8
			0.0, 0.0, 0.0])       
		self.g = 9.81

		self.gps_lat=0
		self.gps_long=0

		rospy.init_node('iris_drone', anonymous = True)

		#SUBSCRIBERS
		
		rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.global_pose)
		rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.loc_pose)
		self.get_linear_vel=rospy.Subscriber('/mavros/local_position/velocity_local', TwistStamped, self.get_vel,)
		rospy.Subscriber('/mavros/imu/data',Imu,self.get_euler_angles)
		#self.loc=Point()
		self.glob=Point()

		#PUBLISHERS
		self.pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped,queue_size=1)
		self.publish_attitude_thrust=rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget,queue_size=1)

		#POSITION QAUD
		rospy.loginfo('INIT')
		self.setarm(1)
		rospy.sleep(2)
		self.offboard()
		self.takeoff(1.0)
		#self.offboard()
		rospy.sleep(5)
		self.gotopose(0,0,4)
		rospy.sleep(5)

	def offboard(self):
		rate = rospy.Rate(10)
		sp = PoseStamped()
		sp.pose.position.x = 0.0
		sp.pose.position.y = 0.0
		sp.pose.position.z = 7.0
		for i in range(10):
			self.pub.publish(sp)
			rate.sleep()
		print('We are good to go!!')
		self.setmode("GUIDED")

	def loc_pose(self, data):

		self.X[0] = data.pose.position.x
		self.X[1] = data.pose.position.y
		self.X[2] = data.pose.position.z

	def global_pose(self, data):
		self.glob.x = data.latitude 
		self.glob.y = data.longitude  
		self.glob.z = data.altitude 

	def setmode(self,md):
		rospy.wait_for_service('/mavros/set_mode')
		try:
			mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
			response = mode(0,md)
			response.mode_sent
		except rospy.ServiceException as e:
			print ("Service call failed: %s"%e)

	def takeoff(self, alt):
		rospy.wait_for_service('/mavros/cmd/takeoff')
		try:
			mode = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
			response = mode(0,0, self.glob.x, self.glob.y, alt)
			response.success
		except rospy.ServiceException as e:
			print ("Service call failed: %s"%e)

	def setarm(self,av): # input: 1=arm, 0=disarm
		rospy.wait_for_service('/mavros/cmd/arming')
		try:
			arming = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
			response = arming(av)
			response.success
		except rospy.ServiceException as e:
			print ("Service call failed: %s" %e)

	def gotopose(self, x, y ,z):
		rate = rospy.Rate(20)
		self.sp = PoseStamped()
		self.sp.pose.position.x = x
		self.sp.pose.position.y = y
		self.sp.pose.position.z = z
		dist = np.sqrt(((self.X[0]-x)**2) + ((self.X[1]-y)**2) + ((self.X[2]-z)**2))
		while(dist > 0.18):
			self.pub.publish(self.sp)
			dist = np.sqrt(((self.X[0]-x)**2) + ((self.X[1]-y)**2) + ((self.X[2]-z)**2))
			rate.sleep()

	def get_vel(self,vel_data):
		self.X[6]=	vel_data.twist.linear.x
		self.X[7]=	vel_data.twist.linear.y
		self.X[8]=	vel_data.twist.linear.z
	
	global p_error = np.array(self.X[0]-x,self.X[1]-y, self.X[2]-z])
	global v_error = np.array([self.X[6], self.X[7], self.X[9]]
	global p_error_dot = v_error
	global v_error_dot = p_dot_dot - v_error_dot
	global v = np.array([self.X[6], self.X[7], self.X[9]]

class SMCController:
	mass = 1
	a1 = 1
	a2 = 1
	b = 1
	dp = np.diag(1,1,1) #drag coefficients
	f2 = -(1/mass)*dp*v

    
	def s0_calc(self):
		dt = 0.1
		s0 = np.array
		s0 = a2*p_error + p_error_dot
		s0 += a1*p_error*dt
		return s0


	def E_calc(self):
		E = np.array
		E = a1*p_error + a2*v_error + p_dot_dot + g*np.array(0,0,1) - f2 
		return E

	def E_calc_cap():
		global E_cap = E_calc() + b*sign(s0_calc())
		return E_cap 

	def mod_E_calc_cap():
		mod_E_cap = numpy.mod(E_calc_cap)

	global phi_d = arcsin((E_cap[0]*sin(roll)-E_cap[1]*cos(roll))/mod_E_calc_cap())
	global theta_d = arctan((E_cap[0]*cos(roll)+E_cap[1]*sin(roll))/E_cap[2])


	def thrust_input():
		u = np.array(cos(phi_d)*cos(roll)*sin(theta_d)+sin(roll)*sin(phi_d)
			, sin(theta_d)*sin(roll)*sin(phi_d) - cos(roll)*sin(phi_d)
			, cos(phi_d)*cos(roll))*E_calc_cap
		return u


#just publish u now to ROS using setpoint_accel