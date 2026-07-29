"""
Stage 1: Charuco Board Fisheye Calibration (Z+F 9020)
=======================================================
Calibrates 4 fisheye cameras using Charuco board for robust corner detection
at extreme angles. Outputs: K (3×3), D (1×4 k1-k4), and T_camera→scanner.

Usage:
    python stage1_charuco_fisheye_calib.py --images_dir ./calib_images --board_w 11 --board_h 8 --square_size 0.06 --marker_size 0.045 --output ./calib_results
"""

import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
import xml.etree.ElementTree as ET


@dataclass
class FisheyeCamera:
    """Z+F 9020 fisheye camera model."""
    serial: str
    camera_id: int  # 0=left, 1=down, 2=right, 3=up
    K: np.ndarray           # 3×3 intrinsics
    D: np.ndarray           # 1×4 distortion [k1, k2, k3, k4]
    image_size: Tuple[int, int]  # (width, height)
    rms_error: float

    # Camera-to-scanner extrinsic (from Charuco calibration)
    T_cam_to_scanner: Optional[np.ndarray] = None  # 4×4

    def to_dict(self) -> dict:
        return {
            "serial": self.serial,
            "camera_id": self.camera_id,
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "image_size": self.image_size,
            "rms_error": self.rms_error,
            "T_cam_to_scanner": self.T_cam_to_scanner.tolist() if self.T_cam_to_scanner is not None else None
        }


class CharucoFisheyeCalibrator:
    """
    Charuco-based fisheye calibration for Z+F PROFILER 9020.

    Charuco boards are superior to plain checkerboards for fisheye because:
    - ArUco markers remain detectable at extreme angles (>120° FOV)
    - Marker IDs resolve the permutation problem (which corner is which)
    - Subpixel refinement on chessboard corners still works when visible
    """

    def __init__(self, board_w: int, board_h: int, 
                 square_size: float, marker_size: float,
                 aruco_dict=cv2.aruco.DICT_6X6_250):
        """
        Args:
            board_w: Number of squares horizontally
            board_h: Number of squares vertically  
            square_size: Chessboard square size in meters
            marker_size: ArUco marker size in meters
            aruco_dict: ArUco dictionary type
        """
        self.board_w = board_w
        self.board_h = board_h
        self.square_size = square_size
        self.marker_size = marker_size

        # Create Charuco board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (board_w, board_h), 
            square_size, 
            marker_size, 
            self.aruco_dict
        )
        self.detector = cv2.aruco.CharucoDetector(self.charuco_board)

        # Object points for all Charuco corners (board-relative)
        self.obj_points_template = self.charuco_board.getChessboardCorners()

    def detect_charuco(self, image: np.ndarray, 
                       refine: bool = True) -> Tuple[Optional[np.ndarray], 
                                                     Optional[np.ndarray],
                                                     Optional[np.ndarray],
                                                     Optional[np.ndarray]]:
        """
        Detect Charuco corners and markers in a single image.

        Returns:
            charuco_corners: N×1×2 detected corner pixel coordinates
            charuco_ids: N×1 corner IDs
            marker_corners: List of marker corner arrays
            marker_ids: M×1 marker IDs
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 4:
            return None, None, None, None

        # Need minimum corners for reliable calibration
        if len(charuco_ids) < 6:
            return None, None, None, None

        return charuco_corners, charuco_ids, marker_corners, marker_ids

    def collect_detections(self, image_paths: List[str], 
                           min_corners: int = 6,
                           visualize: bool = False) -> Tuple[List[np.ndarray], 
                                                             List[np.ndarray],
                                                             List[str]]:
        """
        Collect Charuco detections across all calibration images.

        Returns:
            all_obj_points: List of N×3 object point arrays
            all_img_points: List of N×1×2 image corner arrays  
            valid_paths: List of image paths that succeeded
        """
        all_obj_points = []
        all_img_points = []
        valid_paths = []

        for img_path in image_paths:
            image = cv2.imread(img_path)
            if image is None:
                continue

            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detect_charuco(image)

            if charuco_corners is None or len(charuco_ids) < min_corners:
                continue

            # Get object points for detected corners
            obj_pts = []
            for cid in charuco_ids.flatten():
                obj_pts.append(self.obj_points_template[cid])
            obj_pts = np.array(obj_pts, dtype=np.float32)

            all_obj_points.append(obj_pts)
            all_img_points.append(charuco_corners)
            valid_paths.append(img_path)

            if visualize:
                vis = image.copy()
                cv2.aruco.drawDetectedCornersCharuco(vis, charuco_corners, charuco_ids)
                cv2.imshow("Charuco Detection", vis)
                cv2.waitKey(100)

        if visualize:
            cv2.destroyAllWindows()

        return all_obj_points, all_img_points, valid_paths

    def calibrate_fisheye(self, 
                          all_obj_points: List[np.ndarray],
                          all_img_points: List[np.ndarray],
                          image_size: Tuple[int, int],
                          flags: int = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND,
                          criteria: Tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
                          ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]:
        """
        Calibrate fisheye camera using OpenCV fisheye model.

        The OpenCV fisheye model (Kannala-Brandt):
            θ = atan2(sqrt(X²+Y²), Z)
            r = θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹
            u = fx·(r·X/√(X²+Y²)) + cx
            v = fy·(r·Y/√(X²+Y²)) + cy

        Returns:
            K: 3×3 camera matrix
            D: 1×4 distortion coefficients [k1, k2, k3, k4]
            rvecs: List of rotation vectors (world→camera for each image)
            tvecs: List of translation vectors
            rms: Reprojection RMS error
        """
        # Prepare data in OpenCV format
        obj_points_cv = [pts.reshape(-1, 1, 3) for pts in all_obj_points]
        img_points_cv = [pts.reshape(-1, 1, 2) for pts in all_img_points]

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))

        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            obj_points_cv,
            img_points_cv,
            image_size,
            K,
            D,
            rvecs=None,
            tvecs=None,
            flags=flags,
            criteria=criteria
        )

        return K, D, rvecs, tvecs, rms

    def calibrate_camera_to_scanner(self,
                                     rvecs: List[np.ndarray],
                                     tvecs: List[np.ndarray],
                                     scanner_poses: List[np.ndarray]) -> np.ndarray:
        """
        Compute camera-to-scanner extrinsic by hand-eye calibration.

        Given:
            T_world→camera_i from Charuco calibration (rvec_i, tvec_i)
            T_world→scanner_i from external tracking/GNSS (scanner_poses)

        Solve for T_scanner→camera such that:
            T_world→camera = T_world→scanner × T_scanner→camera

        This is the AX=XB hand-eye calibration problem.

        Args:
            rvecs: Rotation vectors (world→camera) for each image
            tvecs: Translation vectors (world→camera) for each image
            scanner_poses: List of 4×4 T_world→scanner matrices

        Returns:
            T_scanner_to_camera: 4×4 homogeneous transform
        """
        from scipy.spatial.transform import Rotation as R

        # Collect relative motions
        A_list = []
        B_list = []

        for i in range(len(scanner_poses) - 1):
            # Relative camera motion: T_cam_i → T_cam_{i+1}
            T_cam_i = np.eye(4)
            T_cam_i[:3, :3] = cv2.Rodrigues(rvecs[i])[0]
            T_cam_i[:3, 3] = tvecs[i].flatten()

            T_cam_j = np.eye(4)
            T_cam_j[:3, :3] = cv2.Rodrigues(rvecs[i+1])[0]
            T_cam_j[:3, 3] = tvecs[i+1].flatten()

            T_cam_rel = np.linalg.inv(T_cam_i) @ T_cam_j

            # Relative scanner motion
            T_scan_rel = np.linalg.inv(scanner_poses[i]) @ scanner_poses[i+1]

            A_list.append(T_cam_rel)
            B_list.append(T_scan_rel)

        # Solve AX = XB using Tsai-Lenz method
        # X = T_scanner→camera
        return self._solve_hand_eye_tsai(A_list, B_list)

    def _solve_hand_eye_tsai(self, A_list: List[np.ndarray], 
                              B_list: List[np.ndarray]) -> np.ndarray:
        """Tsai-Lenz hand-eye calibration."""
        from scipy.spatial.transform import Rotation as R

        # Rotation part
        S = []
        v = []

        for A, B in zip(A_list, B_list):
            Ra = A[:3, :3]
            Rb = B[:3, :3]

            # Axis-angle representations
            theta_a = np.arccos((np.trace(Ra) - 1) / 2)
            theta_b = np.arccos((np.trace(Rb) - 1) / 2)

            if abs(theta_a) < 1e-6 or abs(theta_b) < 1e-6:
                continue

            skew_a = (Ra - Ra.T) / (2 * np.sin(theta_a))
            skew_b = (Rb - Rb.T) / (2 * np.sin(theta_b))

            a_vec = np.array([skew_a[2,1], skew_a[0,2], skew_a[1,0]]) * theta_a
            b_vec = np.array([skew_b[2,1], skew_b[0,2], skew_b[1,0]]) * theta_b

            S.append(self._skew_symmetric(a_vec + b_vec))
            v.append(b_vec - a_vec)

        if len(S) == 0:
            raise ValueError("No valid motion pairs for hand-eye calibration")

        S = np.vstack(S)
        v = np.hstack(v)

        # Solve for rotation axis
        P = np.linalg.lstsq(S, v, rcond=None)[0]
        P_norm = np.linalg.norm(P)
        if P_norm > 1e-10:
            P = 2 * P / np.sqrt(1 + P_norm**2)

        Rx = self._rodrigues(P)

        # Translation part
        C = []
        d = []
        for A, B in zip(A_list, B_list):
            C.append(A[:3, :3] - np.eye(3))
            d.append(Rx @ B[:3, 3] - A[:3, 3])

        C = np.vstack(C)
        d = np.hstack(d)
        tx = np.linalg.lstsq(C, d, rcond=None)[0]

        T = np.eye(4)
        T[:3, :3] = Rx
        T[:3, 3] = tx

        return T

    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    @staticmethod
    def _rodrigues(v: np.ndarray) -> np.ndarray:
        """Rodrigues formula for rotation matrix from axis-angle."""
        theta = np.linalg.norm(v)
        if theta < 1e-10:
            return np.eye(3)
        k = v / theta
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def export_to_xml(cameras: List[FisheyeCamera], output_path: str):
    """Export calibration to Z+F-style XML format."""
    root = ET.Element("CameraCalibration")
    root.set("version", "1.0")

    for cam in cameras:
        cam_elem = ET.SubElement(root, "Camera")
        cam_elem.set("serial", cam.serial)
        cam_elem.set("id", str(cam.camera_id))
        cam_elem.set("type", "fisheye")
        cam_elem.set("model", "15")  # OpenCV fisheye model ID

        # Image size
        size_elem = ET.SubElement(cam_elem, "ImageSize")
        size_elem.set("width", str(cam.image_size[0]))
        size_elem.set("height", str(cam.image_size[1]))

        # Intrinsics K
        k_elem = ET.SubElement(cam_elem, "Intrinsics")
        k_elem.set("fx", str(cam.K[0, 0]))
        k_elem.set("fy", str(cam.K[1, 1]))
        k_elem.set("cx", str(cam.K[0, 2]))
        k_elem.set("cy", str(cam.K[1, 2]))

        # Distortion D
        d_elem = ET.SubElement(cam_elem, "Distortion")
        d_elem.set("k1", str(cam.D[0, 0]))
        d_elem.set("k2", str(cam.D[1, 0]))
        d_elem.set("k3", str(cam.D[2, 0]))
        d_elem.set("k4", str(cam.D[3, 0]))

        # Extrinsics (camera-to-scanner)
        if cam.T_cam_to_scanner is not None:
            ext_elem = ET.SubElement(cam_elem, "Extrinsic")
            ext_elem.set("R0_tx", str(cam.T_cam_to_scanner[0, 3]))
            ext_elem.set("R0_ty", str(cam.T_cam_to_scanner[1, 3]))
            ext_elem.set("R0_tz", str(cam.T_cam_to_scanner[2, 3]))

            # Euler angles from rotation matrix
            R = cam.T_cam_to_scanner[:3, :3]
            # ZYX Euler angles (roll, pitch, yaw)
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            if sy > 1e-6:
                roll = np.arctan2(R[2,1], R[2,2])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = np.arctan2(R[1,0], R[0,0])
            else:
                roll = np.arctan2(-R[1,2], R[1,1])
                pitch = np.arctan2(-R[2,0], sy)
                yaw = 0
            ext_elem.set("R0_roll", str(roll))
            ext_elem.set("R0_pitch", str(pitch))
            ext_elem.set("R0_yaw", str(yaw))

        cam_elem.set("rms_error", str(cam.rms_error))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Calibration exported to: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Charuco Fisheye Calibration for Z+F 9020")
    parser.add_argument("--images_dir", required=True, help="Directory with calibration images")
    parser.add_argument("--board_w", type=int, default=11, help="Charuco squares width")
    parser.add_argument("--board_h", type=int, default=8, help="Charuco squares height")
    parser.add_argument("--square_size", type=float, default=0.06, help="Square size in meters")
    parser.add_argument("--marker_size", type=float, default=0.045, help="ArUco marker size in meters")
    parser.add_argument("--output", default="/data/Project/hd_map_p06/new_config", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show detection visualization")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize calibrator
    calibrator = CharucoFisheyeCalibrator(
        board_w=args.board_w,
        board_h=args.board_h,
        square_size=args.square_size,
        marker_size=args.marker_size
    )

    # Find images per camera (expects subdirs: cam0, cam1, cam2, cam3 or similar)
    cameras = []
    for cam_id in range(4):
        cam_dir = os.path.join(args.images_dir, f"cam{cam_id}")
        if not os.path.exists(cam_dir):
            # Try serial-based naming
            cam_dir = args.images_dir

        image_paths = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")) + 
                             glob.glob(os.path.join(cam_dir, "*.png")))

        if len(image_paths) == 0:
            print(f"Warning: No images found for camera {cam_id}")
            continue

        print(f"\n=== Calibrating Camera {cam_id} ({len(image_paths)} images) ===")

        # Collect detections
        obj_pts, img_pts, valid_paths = calibrator.collect_detections(
            image_paths, min_corners=6, visualize=args.visualize
        )

        print(f"  Valid detections: {len(obj_pts)}/{len(image_paths)}")

        if len(obj_pts) < 3:
            print(f"  ERROR: Need at least 3 valid detections, got {len(obj_pts)}")
            continue

        # Calibrate fisheye
        sample_img = cv2.imread(valid_paths[0])
        image_size = (sample_img.shape[1], sample_img.shape[0])

        K, D, rvecs, tvecs, rms = calibrator.calibrate_fisheye(
            obj_pts, img_pts, image_size
        )

        print(f"  RMS error: {rms:.4f} pixels")
        print(f"  K = [[{K[0,0]:.2f}, 0, {K[0,2]:.2f}],")
        print(f"       [0, {K[1,1]:.2f}, {K[1,2]:.2f}],")
        print(f"       [0, 0, 1]]")
        print(f"  D = [{D[0,0]:.6f}, {D[1,0]:.6f}, {D[2,0]:.6f}, {D[3,0]:.6f}]")

        camera = FisheyeCamera(
            serial=f"cam{cam_id}",
            camera_id=cam_id,
            K=K,
            D=D,
            image_size=image_size,
            rms_error=rms
        )
        cameras.append(camera)

        # Save individual camera results
        with open(os.path.join(args.output, f"camera_{cam_id}_calib.json"), "w") as f:
            json.dump(camera.to_dict(), f, indent=2)

    # Export combined XML
    export_to_xml(cameras, os.path.join(args.output, "9020C_0140_toScanner_final.xml"))

    print(f"\n=== Calibration Complete ===")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
