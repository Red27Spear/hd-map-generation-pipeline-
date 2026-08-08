import pandas as pd
import numpy as np
import scipy as sp
from geodetic_tools import *

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

def getEulerAnglesFromRotationMatrix(R):
    """
    Extracts Euler angles (roll, pitch, yaw) from a rotation matrix R.
    The angles are returned in radians.
    """
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    
    return roll, pitch, yaw


def rotMat(roll,pitch,yaw):
    """ Computes the rotation matrix from roll, pitch, and yaw angles.
    The angles are in radians.
    Returns a 3x3 rotation matrix.
    """
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]], dtype=np.float64)
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]], dtype=np.float64)
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw), np.cos(yaw),0],[0,0,1]], dtype=np.float64)
    R = Rz @ Ry @ Rx
    return R

def transformMatrix(roll,pitch,yaw,tx,ty,tz):
    R = rotMat(roll,pitch,yaw)
    t = np.array([tx,ty,tz], dtype=np.float64)

    T = np.zeros((4,4), dtype=np.float64)
    T[0:3,0:3] = R
    T[0:3,3] = t
    T[3,3] = 1

    return T


def projection_matrix(T_c_w, K):
    """
    Computes the projection matrix from a transformation matrix T and a camera intrinsic matrix K.
    T is a 4x4 transformation matrix, K is a 3x3 camera intrinsic matrix.
    Returns a 3x4 projection matrix.
    """
    T_w_c = np.linalg.inv(T_c_w)
    aux = np.zeros((3,4), dtype=np.float64)
    aux[0:3,0:3] = np.eye(3,3, dtype=np.float64)
    T_w_im = K @ aux @ T_w_c


    return T_w_im


def projectPoints_to_image(points_world, T_world_image, filter=False):
    """
    Projects 3D points in the world coordinate system to 2D image coordinates.
    points_world: Nx3 array of points in cartesian coordinates (x, y, z).
    T_world_image: 3x4 projection matrix.
    Returns a Nx3 array of projected points in pixel coordinates (x_c, y_c).
    """
    points_image = T_world_image @ np.hstack((points_world, np.ones((points_world.shape[0], 1))), dtype=np.float64).T
    if filter:
        # Filter out points that are behind the camera (z < 0)
        valid_indices = points_image[2, :] > 0
        idx_filter = np.where(valid_indices)[0]
    else:
        idx_filter = np.arange(points_image.shape[1])

    points_image /= points_image[2, :]  # Normalize by the third coordinate
    
    points_image = points_image[:2, :]  # Keep only x and y coordinates
    points_image = points_image.T  # Transpose to get Nx2 shape
    
    return points_image, idx_filter

def getTrafofromTrajectory(Traj, time_camera):
    f_interp_pos = sp.interpolate.interp1d(Traj["Time"].values,Traj[["E","N","height"]].values,kind='cubic',fill_value='extrapolate',axis=0,assume_sorted=True)
    Pos_IMU_camTime = f_interp_pos(time_camera)

    # Interpolate Attitude (RPY)
    slerp = Slerp(Traj["Time"].values,R.from_euler("xyz",Traj[["roll","pitch","yaw"]].values))
    Att_IMU_camTime = slerp(time_camera).as_euler("xyz")
    
    

    T_imu_world = transformMatrix(Att_IMU_camTime[0],Att_IMU_camTime[1],Att_IMU_camTime[2],Pos_IMU_camTime[0],Pos_IMU_camTime[1],Pos_IMU_camTime[2])
    return T_imu_world


def print_diff_Trafo(T_camera_world,T_camera_world_soll,name):

            diff = T_camera_world_soll[:3, 3] - T_camera_world[:3, 3]
            roll_soll, pitch_soll, yaw_soll = getEulerAnglesFromRotationMatrix(T_camera_world_soll[:3, :3])
            roll_ist, pitch_ist, yaw_ist = getEulerAnglesFromRotationMatrix(T_camera_world[:3, :3])
            d_roll = roll_soll - roll_ist
            d_pitch = pitch_soll - pitch_ist
            d_yaw = yaw_soll - yaw_ist
            
            print("-----------------------------------------------------------------------------------------------")
            print(f"Difference of Pose of Camera {name}:")
            print(f"Difference in position: {diff * 1000} mm")
            print(f"Difference in orientation (roll, pitch, yaw): {d_roll*180/np.pi:.6f}, {d_pitch*180/np.pi:.6f}, {d_yaw*180/np.pi:.6f} degrees")
            print("-----------------------------------------------------------------------------------------------")