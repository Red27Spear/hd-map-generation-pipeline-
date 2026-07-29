"""
Stage 2: Time Synchronization (UTC → GPS Seconds-of-Week)
=========================================================
Converts image EXIF timestamps to GPS time (seconds-of-week) for alignment
with GNSS/INS trajectory. Handles leap seconds (+18s as of 2026).

Usage:
    python stage2_time_sync.py --images_dir ./images --output ./synced_times.json
    python stage2_time_sync.py --image_paths ./img1.jpg ./img2.jpg --output ./times.json
"""

import os
import json
import struct
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass, asdict
import argparse


# GPS-UTC leap seconds (current as of 2026-07-08)
# Leap seconds are added irregularly. As of Jan 2017: +18s.
# Check https://hpiers.obspm.fr/eoppc/bul/bulc/ for updates.
GPS_UTC_LEAP_SECONDS = 18  # seconds

# GPS epoch: 1980-01-06 00:00:00 UTC
GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)


@dataclass
class SyncedTimestamp:
    """Synchronized timestamp record for one image."""
    image_path: str
    image_name: str

    # Original EXIF timestamp
    exif_utc: Optional[datetime] = None
    exif_utc_str: Optional[str] = None

    # GPS time
    gps_week: Optional[int] = None
    gps_seconds_of_week: Optional[float] = None  # This is what moro.track uses
    gps_full_seconds: Optional[float] = None     # Total seconds since GPS epoch

    # Unix timestamp (for cross-reference)
    unix_timestamp: Optional[float] = None

    # Camera info from filename
    serial_number: Optional[str] = None
    camera_id: Optional[int] = None
    exposure_time: Optional[float] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.exif_utc:
            result["exif_utc"] = self.exif_utc.isoformat()
        return result


class TimeSynchronizer:
    """
    Synchronizes image timestamps with GNSS/INS trajectory.

    Z+F PROFILER 9020 workflow:
    1. Images have EXIF DateTimeOriginal in UTC
    2. Trajectory (moro.track.txt) uses GPS seconds-of-week
    3. Need to convert EXIF UTC → GPS time → seconds-of-week
    4. Align by matching closest trajectory epoch to image time

    Leap seconds: GPS time = UTC time + leap_seconds
    So: GPS_seconds = UTC_seconds_since_GPS_epoch + leap_seconds
    """

    def __init__(self, leap_seconds: int = GPS_UTC_LEAP_SECONDS):
        self.leap_seconds = leap_seconds

    def utc_to_gps(self, utc_dt: datetime) -> Tuple[int, float, float]:
        """
        Convert UTC datetime to GPS time.

        Args:
            utc_dt: UTC datetime (must be timezone-aware or assumed UTC)

        Returns:
            gps_week: GPS week number (rollover every 1024 weeks ≈ 19.6 years)
            gps_seconds_of_week: Seconds within the GPS week [0, 604800)
            gps_full_seconds: Total seconds since GPS epoch (includes leap seconds)
        """
        if utc_dt.tzinfo is None:
            utc_dt = utc_dt.replace(tzinfo=timezone.utc)

        # GPS time runs ahead of UTC by leap seconds
        gps_dt = utc_dt + timedelta(seconds=self.leap_seconds)

        # Seconds since GPS epoch
        delta = gps_dt - GPS_EPOCH
        gps_full_seconds = delta.total_seconds()

        # GPS week and seconds-of-week
        SECONDS_PER_WEEK = 7 * 24 * 3600  # 604800
        gps_week = int(gps_full_seconds // SECONDS_PER_WEEK)
        gps_seconds_of_week = gps_full_seconds % SECONDS_PER_WEEK

        return gps_week, gps_seconds_of_week, gps_full_seconds

    def gps_sow_to_utc(self, gps_week: int, gps_seconds_of_week: float) -> datetime:
        """Convert GPS week + seconds-of-week back to UTC datetime."""
        SECONDS_PER_WEEK = 7 * 24 * 3600
        gps_full_seconds = gps_week * SECONDS_PER_WEEK + gps_seconds_of_week
        gps_dt = GPS_EPOCH + timedelta(seconds=gps_full_seconds)
        utc_dt = gps_dt - timedelta(seconds=self.leap_seconds)
        return utc_dt

    def parse_exif_timestamp(self, exif_bytes: bytes) -> Optional[datetime]:
        """
        Parse EXIF DateTimeOriginal from raw bytes.
        Format: "2026:05:08 08:10:23\0" (20 chars + null)
        """
        try:
            timestamp_str = exif_bytes.decode("ascii").strip("\x00").strip()
            # EXIF format: "YYYY:MM:DD HH:MM:SS"
            dt = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    def read_exif_timestamp(self, image_path: str) -> Optional[datetime]:
        """
        Read DateTimeOriginal from JPEG EXIF without external dependencies.

        Parses APP1 marker (0xFFE1) and IFD0 for DateTimeOriginal tag (0x9003).
        """
        try:
            with open(image_path, "rb") as f:
                # Check JPEG magic
                if f.read(2) != b"\xFF\xD8":
                    return None

                while True:
                    marker = f.read(2)
                    if len(marker) < 2:
                        break

                    if marker[0] != 0xFF:
                        continue

                    marker_type = marker[1]

                    if marker_type == 0xD9:  # EOI
                        break
                    if marker_type == 0xD8:  # SOI
                        continue
                    if marker_type == 0x01:  # TEM
                        continue
                    if 0xD0 <= marker_type <= 0xD9:
                        continue

                    # Read segment length
                    length_bytes = f.read(2)
                    if len(length_bytes) < 2:
                        break
                    length = struct.unpack(">H", length_bytes)[0]

                    if marker_type == 0xE1:  # APP1 (EXIF)
                        segment = f.read(length - 2)

                        # Check for EXIF header
                        if segment[:6] == b"Exif\x00\x00":
                            tiff_header = segment[6:8]

                            # Determine byte order
                            if tiff_header == b"II":
                                endian = "<"
                            elif tiff_header == b"MM":
                                endian = ">"
                            else:
                                continue

                            # IFD0 offset
                            ifd_offset = struct.unpack(endian + "I", segment[10:14])[0]

                            # Parse IFD0
                            num_entries = struct.unpack(endian + "H", 
                                                        segment[6+ifd_offset:6+ifd_offset+2])[0]

                            offset = 6 + ifd_offset + 2
                            for _ in range(num_entries):
                                tag = struct.unpack(endian + "H", segment[offset:offset+2])[0]
                                type_ = struct.unpack(endian + "H", segment[offset+2:offset+4])[0]
                                count = struct.unpack(endian + "I", segment[offset+4:offset+8])[0]
                                value_offset = struct.unpack(endian + "I", segment[offset+8:offset+12])[0]

                                if tag == 0x9003:  # DateTimeOriginal
                                    # Value is ASCII string, either inline or offset
                                    if count <= 4:
                                        ts_bytes = segment[offset+8:offset+8+count]
                                    else:
                                        ts_bytes = segment[6+value_offset:6+value_offset+count]
                                    return self.parse_exif_timestamp(ts_bytes)

                                offset += 12
                    else:
                        # Skip other segments
                        f.seek(length - 2, 1)

        except Exception as e:
            pass

        return None

    def parse_zf_filename(self, filename: str) -> Dict[str, Optional[Union[str, int, float]]]:
        """
        Parse Z+F filename: MoRo_Bonn{utc-20260508_08-10-23-655;sn-4108690914;exp-0.205892;c-0;pls-7}.jpg

        Returns dict with: utc_str, serial, exposure, camera_id, pls
        """
        result = {"utc_str": None, "serial": None, "exposure": None, 
                  "camera_id": None, "pls": None}

        try:
            # Extract content between braces
            start = filename.find("{")
            end = filename.find("}")
            if start == -1 or end == -1:
                return result

            content = filename[start+1:end]
            parts = content.split(";")

            for part in parts:
                # Z+F filename format uses "-" as key-value separator: utc-20260508_..., sn-..., etc.
                if "-" in part:
                    key, value = part.split("-", 1)
                    key = key.strip()
                    value = value.strip()

                    if key == "utc":
                        result["utc_str"] = value
                    elif key == "sn":
                        result["serial"] = value
                    elif key == "exp":
                        result["exposure"] = float(value)
                    elif key == "c":
                        result["camera_id"] = int(value)
                    elif key == "pls":
                        result["pls"] = int(value)

        except Exception:
            pass

        return result

    def filename_to_utc(self, filename: str) -> Optional[datetime]:
        """
        Extract UTC timestamp from Z+F filename.
        Format: utc-20260508_08-10-23-655 → 2026-05-08 08:10:23.655 UTC
        """
        parsed = self.parse_zf_filename(filename)
        utc_str = parsed.get("utc_str")

        if utc_str is None:
            return None

        try:
            # Format: 20260508_08-10-23-655
            dt = datetime.strptime(utc_str, "%Y%m%d_%H-%M-%S-%f")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def sync_image(self, image_path: str, use_filename: bool = True) -> SyncedTimestamp:
        """
        Synchronize a single image: extract timestamp and convert to GPS time.

        Priority:
        1. Z+F filename UTC (most reliable for this system)
        2. EXIF DateTimeOriginal (fallback)
        """
        path = Path(image_path)
        filename = path.name

        record = SyncedTimestamp(
            image_path=str(path.absolute()),
            image_name=filename
        )

        # Parse filename metadata
        parsed = self.parse_zf_filename(filename)
        record.serial_number = parsed.get("serial")
        record.camera_id = parsed.get("camera_id")
        record.exposure_time = parsed.get("exposure")

        # Get UTC timestamp
        utc_dt = None

        if use_filename:
            utc_dt = self.filename_to_utc(filename)

        if utc_dt is None:
            utc_dt = self.read_exif_timestamp(str(path))

        if utc_dt is None:
            return record  # Failed to get timestamp

        record.exif_utc = utc_dt
        record.exif_utc_str = utc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        record.unix_timestamp = utc_dt.timestamp()

        # Convert to GPS time
        gps_week, gps_sow, gps_full = self.utc_to_gps(utc_dt)
        record.gps_week = gps_week
        record.gps_seconds_of_week = gps_sow
        record.gps_full_seconds = gps_full

        return record

    def sync_directory(self, images_dir: str, 
                       pattern: str = "*.jpg",
                       use_filename: bool = True) -> List[SyncedTimestamp]:
        """Synchronize all images in a directory."""
        import glob

        image_paths = sorted(glob.glob(os.path.join(images_dir, "**", pattern), recursive=True))

        records = []
        for img_path in image_paths:
            record = self.sync_image(img_path, use_filename=use_filename)
            if record.gps_seconds_of_week is not None:
                records.append(record)

        # Sort by GPS time
        records.sort(key=lambda r: r.gps_seconds_of_week or 0)

        return records

    def find_nearest_trajectory_epoch(self, 
                                       gps_sow: float, 
                                       trajectory_times: List[float]) -> Tuple[int, float]:
        """
        Find nearest trajectory epoch to given GPS seconds-of-week.

        Args:
            gps_sow: Target GPS seconds-of-week
            trajectory_times: Sorted list of trajectory GPS SOW values

        Returns:
            idx: Index of nearest trajectory epoch
            dt: Time difference (trajectory_time - image_time)
        """
        # Binary search for nearest
        import bisect

        idx = bisect.bisect_left(trajectory_times, gps_sow)

        if idx == 0:
            return 0, trajectory_times[0] - gps_sow
        if idx == len(trajectory_times):
            return len(trajectory_times) - 1, trajectory_times[-1] - gps_sow

        # Check which is closer
        before_diff = abs(trajectory_times[idx - 1] - gps_sow)
        after_diff = abs(trajectory_times[idx] - gps_sow)

        if before_diff <= after_diff:
            return idx - 1, trajectory_times[idx - 1] - gps_sow
        else:
            return idx, trajectory_times[idx] - gps_sow


def load_trajectory_times(trajectory_path: str, time_col: int = 0) -> List[float]:
    """
    Load GPS seconds-of-week from trajectory file.

    Expected format (moro.track.txt):
        Time(sow)  Easting  Northing  Height  Roll  Pitch  Heading
    """
    times = []
    with open(trajectory_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) > time_col:
                try:
                    times.append(float(parts[time_col]))
                except ValueError:
                    continue
    return times


def main():
    parser = argparse.ArgumentParser(description="Time Synchronization for Z+F 9020")
    parser.add_argument("--images_dir", help="Directory with images")
    parser.add_argument("--image_paths", nargs="+", help="Individual image paths")
    parser.add_argument("--trajectory", help="Trajectory file (moro.track.txt)")
    parser.add_argument("--output", default="./synced_times.json", help="Output JSON path")
    parser.add_argument("--leap_seconds", type=int, default=GPS_UTC_LEAP_SECONDS, 
                        help=f"GPS-UTC leap seconds (default: {GPS_UTC_LEAP_SECONDS})")
    parser.add_argument("--use_filename", action="store_true", default=True,
                        help="Use filename UTC (default) or EXIF")
    args = parser.parse_args()

    sync = TimeSynchronizer(leap_seconds=args.leap_seconds)

    # Collect images
    if args.images_dir:
        print(f"Synchronizing images from: {args.images_dir}")
        records = sync.sync_directory(args.images_dir, use_filename=args.use_filename)
    elif args.image_paths:
        print(f"Synchronizing {len(args.image_paths)} images...")
        records = [sync.sync_image(p, use_filename=args.use_filename) for p in args.image_paths]
        records = [r for r in records if r.gps_seconds_of_week is not None]
    else:
        print("Error: Provide --images_dir or --image_paths")
        return

    print(f"Successfully synchronized: {len(records)} images")

    # Load trajectory and find nearest epochs
    if args.trajectory:
        print(f"Loading trajectory: {args.trajectory}")
        traj_times = load_trajectory_times(args.trajectory)
        print(f"Trajectory epochs: {len(traj_times)}")

        for record in records:
            idx, dt = sync.find_nearest_trajectory_epoch(
                record.gps_seconds_of_week, traj_times
            )
            record.nearest_traj_idx = idx
            record.time_offset_ms = dt * 1000

    # Save results
    output_data = {
        "leap_seconds": args.leap_seconds,
        "total_images": len(records),
        "images": [r.to_dict() for r in records]
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {args.output}")

    # Summary
    if records:
        sow_values = [r.gps_seconds_of_week for r in records]
        print(f"\nGPS SOW range: {min(sow_values):.3f} - {max(sow_values):.3f} s")
        print(f"Duration: {max(sow_values) - min(sow_values):.3f} s")


if __name__ == "__main__":
    main()
