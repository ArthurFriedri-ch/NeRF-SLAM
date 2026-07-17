import os
import numpy as np
import cv2
import pandas as pd
from datasets.dataset import *


class TumDataset(Dataset):
    def __init__(self, args, device):
        super().__init__("Tum", args, device)
        self.dataset_dir = args.dataset_dir
        self.undistort = True          # TUM fr1 has real radial distortion; see note below
        self.t0 = None
        self.parse_dataset()
        self._build_dataset_index()

    def _get_cam_calib(self):
        # freiburg1 defaults — fr1/fr2/fr3 DIFFER; set per sequence.
        width, height  = 640, 480
        fx, fy, cx, cy = 517.306408, 516.469215, 318.643040, 255.313989
        k1, k2, p1, p2, k3 = 0.262383, -0.953104, -0.005358, 0.002628, 1.163314
        body_T_cam0    = np.eye(4)
        rate_hz        = 30.0
        resolution     = Resolution(width, height)
        pinhole0       = PinholeCameraModel(fx, fy, cx, cy)
        # if we undistort in the loader, downstream sees a distortion-free image:
        dist_coeffs    = (0.0, 0.0, 0.0, 0.0) if self.undistort else (k1, k2, p1, p2)
        distortion0    = RadTanDistortionModel(*dist_coeffs)
        aabb           = np.array([[-2, -2, -2], [2, 2, 2]])
        depth_scale    = 1.0          # unused in monocular, kept for calib shape
        self._K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self._dist = np.array([k1, k2, p1, p2, k3], dtype=np.float64)  # 5 coeffs, k3 included
        return CameraCalibration(body_T_cam0, pinhole0, distortion0,
                                 rate_hz, resolution, aabb, depth_scale)

    def parse_dataset(self):
        self.cam0_calib = self._get_cam_calib()
        self.cam_calibs = [self.cam0_calib]

        # Monocular: read rgb.txt (timestamp, filename). No associations needed.
        rgb_txt = os.path.join(self.dataset_dir, "rgb.txt")
        with open(rgb_txt, "r") as f:
            rows = [ln.strip().split() for ln in f
                    if ln.strip() and not ln.startswith("#")]
        assert rows, f"No RGB frames found in {rgb_txt}"
        self.rgb_timestamps = np.array([float(r[0]) for r in rows])
        self.rgb_files      = [r[1] for r in rows]

        # Cap to buffer, like ReplicaDataset does.
        N = len(self.rgb_files)
        self.rgb_timestamps = self.rgb_timestamps[:N]
        self.rgb_files      = self.rgb_files[:N]

        # GT trajectory — for EVALUATION only, never fed to SLAM.
        gt_path = os.path.join(self.dataset_dir, "groundtruth.txt")
        if os.path.exists(gt_path):
            self.gt_df = pd.read_csv(gt_path, sep=r"\s+", comment="#",
                                     names=["timestamp","tx","ty","tz",
                                            "qx","qy","qz","qw"])
            self.gt_df.set_index("timestamp", drop=False, inplace=True)
        else:
            self.gt_df = None

    def _load_image(self, frame_id):
        img = cv2.imread(os.path.join(self.dataset_dir, self.rgb_files[frame_id]))
        assert img is not None, f"missing rgb {self.rgb_files[frame_id]}"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.undistort:
            img = cv2.undistort(img, self._K, self._dist)   # rectify with same K
        return img

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, k):
        return self.data_packets[k] if self.data_packets is not None \
               else self._get_data_packet(k)

    def _get_data_packet(self, k):
        t_cam0 = self.rgb_timestamps[k]
        image  = self._load_image(k)

        # nearest GT pose slice since the previous frame (evaluation aid only)
        gt_t0_t1 = None
        if self.gt_df is not None:
            t1_near = self.gt_df.index.get_indexer([t_cam0], method="nearest")[0]
            gt_t0_t1 = self.gt_df.iloc[self.t0:t1_near + 1] if self.t0 is not None \
                       else self.gt_df.iloc[[t1_near]]
            self.t0 = t1_near

        return {"k": np.array([k]),
                "t_cams": np.array([t_cam0]),
                "images": np.array([image]),
                "calibs": self.cam_calibs,  # was "cam_calibs"
                "gt_t0_t1": gt_t0_t1,
                "is_last_frame": (k >= self.__len__() - 1)}

    def _build_dataset_index(self):
        self.data_packets = [dp for dp in self.stream()]

    def stream(self):
        for k in range(self.__len__()):        # fixed: range(), was iterating an int
            yield self._get_data_packet(k)