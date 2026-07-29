"""
Stage 4: Point Cloud Registration
================================
Transforms LiDAR points from scanner frame to UTM/ENU world coordinates
using interpolated scanner poses from Stage 3.

For each LiDAR point P_scanner:
    P_world = T_scanner_to_world @ P_scanner

Where T_scanner_to_world comes from trajectory interpolation at scan time.

Usage:
    python stage4_pointcloud_registration.py --las input.las --poses poses_at_images.json --output registered.las
"""

import numpy as np
import json
import struct
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterator
from pathlib import Path
import warnings


@dataclass
class LASPoint:
    """Single LiDAR point with attributes."""
    x: float          # Scanner-frame X (m)
    y: float          # Scanner-frame Y (m)
    z: float          # Scanner-frame Z (m)
    intensity: int    # 0-65535
    return_number: int
    num_returns: int
    classification: int
    scan_angle: float
    gps_time: float   # GPS seconds-of-week (for time-based lookup)

    # World coordinates (filled after registration)
    x_world: Optional[float] = None
    y_world: Optional[float] = None
    z_world: Optional[float] = None


class LASReader:
    """
    Minimal LAS/LAZ file reader (LAS 1.2/1.4 format).

    Supports point formats 0, 1, 2, 3 (common for terrestrial/mobile LiDAR).
    For full LAZ support, use laspy. This is a lightweight fallback.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header = self._read_header()
        self.point_format = self.header["point_data_format"]
        self.point_count = self.header["legacy_number_of_point_records"]
        if self.point_count == 0:
            self.point_count = self.header.get("number_of_point_records", 0)

    def _read_header(self) -> dict:
        """Read LAS file header (227 bytes for LAS 1.2, 375 for LAS 1.4)."""
        with open(self.filepath, "rb") as f:
            header = {}

            # File signature
            header["file_signature"] = f.read(4).decode("ascii")
            assert header["file_signature"] == "LASF", "Not a valid LAS file"

            # File source ID
            header["file_source_id"] = struct.unpack("<H", f.read(2))[0]

            # Global encoding
            header["global_encoding"] = struct.unpack("<H", f.read(2))[0]

            # GUID
            header["guid"] = f.read(16)

            # Version
            header["version_major"] = struct.unpack("<B", f.read(1))[0]
            header["version_minor"] = struct.unpack("<B", f.read(1))[0]

            # System identifier
            header["system_identifier"] = f.read(32).decode("ascii", errors="ignore").strip()

            # Generating software
            header["generating_software"] = f.read(32).decode("ascii", errors="ignore").strip()

            # Creation date
            header["creation_day"] = struct.unpack("<H", f.read(2))[0]
            header["creation_year"] = struct.unpack("<H", f.read(2))[0]

            # Header size
            header["header_size"] = struct.unpack("<H", f.read(2))[0]

            # Offset to point data
            header["offset_to_point_data"] = struct.unpack("<I", f.read(4))[0]

            # Number of variable length records
            header["number_of_vlrs"] = struct.unpack("<I", f.read(4))[0]

            # Point data format
            header["point_data_format"] = struct.unpack("<B", f.read(1))[0]

            # Point data record length
            header["point_data_record_length"] = struct.unpack("<H", f.read(2))[0]

            # Legacy number of point records
            header["legacy_number_of_point_records"] = struct.unpack("<I", f.read(4))[0]

            # Legacy number of points by return
            header["legacy_number_of_points_by_return"] = struct.unpack("<5I", f.read(20))

            # Scale factors
            header["x_scale"] = struct.unpack("<d", f.read(8))[0]
            header["y_scale"] = struct.unpack("<d", f.read(8))[0]
            header["z_scale"] = struct.unpack("<d", f.read(8))[0]

            # Offsets
            header["x_offset"] = struct.unpack("<d", f.read(8))[0]
            header["y_offset"] = struct.unpack("<d", f.read(8))[0]
            header["z_offset"] = struct.unpack("<d", f.read(8))[0]

            # Max/Min
            header["x_max"] = struct.unpack("<d", f.read(8))[0]
            header["x_min"] = struct.unpack("<d", f.read(8))[0]
            header["y_max"] = struct.unpack("<d", f.read(8))[0]
            header["y_min"] = struct.unpack("<d", f.read(8))[0]
            header["z_max"] = struct.unpack("<d", f.read(8))[0]
            header["z_min"] = struct.unpack("<d", f.read(8))[0]

            # LAS 1.4 extended fields
            if header["version_major"] == 1 and header["version_minor"] >= 4:
                header["start_of_waveform_data_packet_record"] = struct.unpack("<Q", f.read(8))[0]
                header["start_of_first_extended_vlr"] = struct.unpack("<Q", f.read(8))[0]
                header["number_of_extended_vlrs"] = struct.unpack("<I", f.read(4))[0]
                header["number_of_point_records"] = struct.unpack("<Q", f.read(8))[0]
                header["number_of_points_by_return"] = struct.unpack("<15Q", f.read(120))

            return header

    def _point_record_size(self) -> int:
        """Get point record size based on format."""
        format_sizes = {
            0: 20, 1: 28, 2: 26, 3: 34,
            4: 57, 5: 63, 6: 30, 7: 36,
            8: 38, 9: 59, 10: 67
        }
        return format_sizes.get(self.point_format, self.header["point_data_record_length"])

    def read_points(self, max_points: Optional[int] = None) -> Iterator[LASPoint]:
        """
        Read points from LAS file.

        Yields LASPoint objects with scanner-frame coordinates.
        """
        record_size = self._point_record_size()
        scale_x = self.header["x_scale"]
        scale_y = self.header["y_scale"]
        scale_z = self.header["z_scale"]
        offset_x = self.header["x_offset"]
        offset_y = self.header["y_offset"]
        offset_z = self.header["z_offset"]

        fmt = self.point_format

        with open(self.filepath, "rb") as f:
            f.seek(self.header["offset_to_point_data"])

            count = min(max_points, self.point_count) if max_points else self.point_count

            for i in range(count):
                record = f.read(record_size)
                if len(record) < record_size:
                    break

                # Parse based on point format
                if fmt in [0, 1, 2, 3]:
                    # Common fields for formats 0-3
                    x_raw = struct.unpack("<i", record[0:4])[0]
                    y_raw = struct.unpack("<i", record[4:8])[0]
                    z_raw = struct.unpack("<i", record[8:12])[0]
                    intensity = struct.unpack("<H", record[12:14])[0]

                    # Return info (bits 0-2 = return number, bits 3-5 = number of returns)
                    return_byte = struct.unpack("<B", record[14:15])[0]
                    return_number = (return_byte & 0b00000111)
                    num_returns = (return_byte & 0b00111000) >> 3

                    classification = struct.unpack("<B", record[15:16])[0]
                    scan_angle = struct.unpack("<b", record[16:17])[0]  # signed byte

                    # GPS time (format 1, 3)
                    gps_time = 0.0
                    if fmt in [1, 3]:
                        gps_time = struct.unpack("<d", record[20:28])[0]

                    yield LASPoint(
                        x=x_raw * scale_x + offset_x,
                        y=y_raw * scale_y + offset_y,
                        z=z_raw * scale_z + offset_z,
                        intensity=intensity,
                        return_number=return_number,
                        num_returns=num_returns,
                        classification=classification,
                        scan_angle=scan_angle,
                        gps_time=gps_time
                    )
                else:
                    # Skip unsupported formats
                    continue


class PointCloudRegistrar:
    """
    Registers LiDAR point clouds to world coordinates.

    For each point with GPS time:
        1. Find scanner pose at that GPS time (from Stage 3 interpolation)
        2. Transform: P_world = T_scanner_to_world @ P_scanner
        3. Output registered point cloud
    """

    def __init__(self, poses_json_path: str):
        """
        Args:
            poses_json_path: Stage 3 output with interpolated scanner poses
        """
        self.poses = self._load_poses(poses_json_path)
        self._build_time_index()

    def _load_poses(self, path: str) -> List[dict]:
        with open(path, "r") as f:
            data = json.load(f)
        return data.get("poses", [])

    def _build_time_index(self):
        """Build sorted time array for efficient lookup."""
        self.pose_times = np.array([p["time"] for p in self.poses])
        self.pose_transforms = []

        for p in self.poses:
            T = np.array(p["T_scanner_to_enu"])
            self.pose_transforms.append(T)

    def find_pose_at_time(self, gps_time: float) -> Optional[np.ndarray]:
        """
        Find scanner pose at given GPS time.

        Uses nearest-neighbor if exact match not found.
        For scan data with per-point GPS time, linear interpolation between
        trajectory poses would be more accurate.
        """
        if len(self.pose_times) == 0:
            return None

        # Find nearest pose
        idx = np.argmin(np.abs(self.pose_times - gps_time))

        # Check if within reasonable time window (e.g., 0.5s)
        dt = abs(self.pose_times[idx] - gps_time)
        if dt > 0.5:
            warnings.warn(f"Large time gap: {dt:.3f}s for GPS time {gps_time}")

        return self.pose_transforms[idx]

    def register_point(self, point: LASPoint) -> LASPoint:
        """
        Transform single LiDAR point from scanner frame to world frame.

        P_world = T_scanner_to_world @ [x, y, z, 1]ᵀ
        """
        T = self.find_pose_at_time(point.gps_time)

        if T is None:
            point.x_world = point.x
            point.y_world = point.y
            point.z_world = point.z
            return point

        p_scanner = np.array([point.x, point.y, point.z, 1.0])
        p_world = T @ p_scanner

        point.x_world = p_world[0]
        point.y_world = p_world[1]
        point.z_world = p_world[2]

        return point

    def register_points(self, points: List[LASPoint], 
                        batch_size: int = 10000) -> List[LASPoint]:
        """Register a batch of points efficiently."""
        registered = []

        for i, point in enumerate(points):
            registered.append(self.register_point(point))

            if (i + 1) % batch_size == 0:
                print(f"  Registered {i+1}/{len(points)} points...")

        return registered

    def register_stream(self, las_reader: LASReader,
                        max_points: Optional[int] = None) -> Iterator[LASPoint]:
        """Stream-process points from LAS file (memory-efficient)."""
        count = 0
        for point in las_reader.read_points(max_points=max_points):
            yield self.register_point(point)
            count += 1
            if max_points and count >= max_points:
                break


def write_las_header(f, point_count: int,
                     x_min: float, x_max: float,
                     y_min: float, y_max: float,
                     z_min: float, z_max: float,
                     point_format: int = 1,
                     x_off: float = 0.0, y_off: float = 0.0, z_off: float = 0.0) -> int:
    """
    Write minimal LAS 1.2 header for registered point cloud.
    Returns offset to point data.
    """
    # File signature
    f.write(b"LASF")

    # File source ID
    f.write(struct.pack("<H", 0))

    # Global encoding (GPS time = week seconds)
    f.write(struct.pack("<H", 1))

    # GUID (zeros)
    f.write(b"\x00" * 16)

    # Version 1.2
    f.write(struct.pack("<B", 1))
    f.write(struct.pack("<B", 2))

    # System identifier
    f.write(b"Z+F 9020 Registration".ljust(32))

    # Generating software
    f.write(b"stage4_registration".ljust(32))

    # Creation date
    from datetime import datetime
    now = datetime.now()
    f.write(struct.pack("<H", now.timetuple().tm_yday))
    f.write(struct.pack("<H", now.year))

    # Header size
    header_size = 227
    f.write(struct.pack("<H", header_size))

    # Offset to point data (after header)
    offset_to_data = header_size
    f.write(struct.pack("<I", offset_to_data))

    # Number of VLRs
    f.write(struct.pack("<I", 0))

    # Point data format
    f.write(struct.pack("<B", point_format))

    # Point data record length
    record_lengths = {0: 20, 1: 28, 2: 26, 3: 34}
    f.write(struct.pack("<H", record_lengths.get(point_format, 28)))

    # Number of point records
    f.write(struct.pack("<I", point_count))

    # Number of points by return (5 returns)
    f.write(struct.pack("<5I", point_count, 0, 0, 0, 0))

    # Scale factors (use 0.001 mm precision)
    scale = 0.001
    f.write(struct.pack("<d", scale))
    f.write(struct.pack("<d", scale))
    f.write(struct.pack("<d", scale))

    # Offsets — must accommodate large UTM coordinates to avoid int32 overflow
    f.write(struct.pack("<d", x_off))
    f.write(struct.pack("<d", y_off))
    f.write(struct.pack("<d", z_off))

    # Max/Min
    f.write(struct.pack("<d", x_max))
    f.write(struct.pack("<d", x_min))
    f.write(struct.pack("<d", y_max))
    f.write(struct.pack("<d", y_min))
    f.write(struct.pack("<d", z_max))
    f.write(struct.pack("<d", z_min))

    return offset_to_data


def write_registered_las(output_path: str, 
                           points: List[LASPoint],
                           point_format: int = 1):
    """Write registered points to LAS file."""

    # Compute bounds
    xs = [p.x_world for p in points if p.x_world is not None]
    ys = [p.y_world for p in points if p.y_world is not None]
    zs = [p.z_world for p in points if p.z_world is not None]

    if not xs:
        print("No valid points to write")
        return

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    scale = 0.001
    # Offsets must be set so that (coord - offset)/scale fits in int32
    x_off = float(int(x_min))
    y_off = float(int(y_min))
    z_off = float(int(z_min))

    with open(output_path, "wb") as f:
        offset = write_las_header(f, len(points),
                                  x_min, x_max, y_min, y_max, z_min, z_max,
                                  point_format, x_off, y_off, z_off)

        for point in points:
            if point.x_world is None:
                continue

            # Write point record — subtract offset before scaling to avoid int32 overflow
            x_int = int(round((point.x_world - x_off) / scale))
            y_int = int(round((point.y_world - y_off) / scale))
            z_int = int(round((point.z_world - z_off) / scale))

            f.write(struct.pack("<i", x_int))
            f.write(struct.pack("<i", y_int))
            f.write(struct.pack("<i", z_int))
            f.write(struct.pack("<H", point.intensity))

            # Return info byte
            return_byte = (point.return_number & 0x07) | ((point.num_returns & 0x07) << 3)
            f.write(struct.pack("<B", return_byte))

            f.write(struct.pack("<B", point.classification))
            f.write(struct.pack("<b", int(point.scan_angle)))
            f.write(struct.pack("<B", 0))  # User data
            f.write(struct.pack("<H", 0))  # Point source ID

            # GPS time (format 1)
            if point_format in [1, 3]:
                f.write(struct.pack("<d", point.gps_time))

    print(f"Written {len(points)} points to: {output_path}")
    print(f"Bounds: X[{x_min:.3f}, {x_max:.3f}] Y[{y_min:.3f}, {y_max:.3f}] Z[{z_min:.3f}, {z_max:.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Point Cloud Registration for Z+F 9020")
    parser.add_argument("--las", required=True, help="Input LAS file (scanner frame)")
    parser.add_argument("--poses", required=True, help="Stage 3 poses JSON")
    parser.add_argument("--output", default="./registered.las", help="Output LAS file")
    parser.add_argument("--max_points", type=int, help="Max points to process (for testing)")
    parser.add_argument("--format", type=int, default=1, choices=[0, 1, 2, 3],
                        help="Output LAS point format")
    args = parser.parse_args()

    # Initialize registrar
    print("Loading poses...")
    registrar = PointCloudRegistrar(args.poses)
    print(f"Loaded {len(registrar.poses)} poses")

    # Read LAS / LAZ — prefer laspy (handles compressed LAZ), fall back to built-in reader
    print(f"\nReading file: {args.las}")
    registered = []

    try:
        import laspy
        las_data = laspy.read(args.las)
        total = len(las_data.x)
        max_pts = args.max_points if args.max_points else total
        print(f"Point format: {las_data.header.point_format.id}  Total: {total:,}")
        print("\nRegistering points...")
        has_gps = hasattr(las_data, "gps_time")
        for i in range(min(max_pts, total)):
            gps_t = float(las_data.gps_time[i]) if has_gps else 0.0
            pt = LASPoint(
                x=float(las_data.x[i]), y=float(las_data.y[i]), z=float(las_data.z[i]),
                intensity=int(las_data.intensity[i]),
                return_number=int(las_data.return_number[i]),
                num_returns=int(las_data.number_of_returns[i]),
                classification=int(las_data.classification[i]),
                scan_angle=float(las_data.scan_angle_rank[i]) if hasattr(las_data, 'scan_angle_rank') else 0.0,
                gps_time=gps_t,
            )
            registered.append(registrar.register_point(pt))
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i+1:,} points...")
    except ImportError:
        reader = LASReader(args.las)
        print(f"Point format: {reader.point_format}  Total: {reader.point_count:,}")
        print("\nRegistering points...")
        for i, point in enumerate(registrar.register_stream(reader, args.max_points)):
            registered.append(point)
            if args.max_points and i >= args.max_points - 1:
                break
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i+1:,} points...")

    print(f"\nTotal registered: {len(registered):,}")

    # Write output
    print(f"\nWriting output...")
    write_registered_las(args.output, registered, args.format)

    print("\nDone!")


if __name__ == "__main__":
    main()
