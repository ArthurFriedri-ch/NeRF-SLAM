# fusion/dump_fusion.py
import os
import numpy as np
import torch
from lietorch import SE3
import slam_interface


class DumpFusion:
    def __init__(self, name, args, device):
        self.name = name
        self.args = args
        self.device = device
        self.dsf = 8.0  # frontend downscale factor; matches TsdfFusion
        self.out_dir = getattr(args, "dump_dir", "./dump_records")
        os.makedirs(self.out_dir, exist_ok=True)  # fork-safe: created before any spawn use
        self.count = 0

    def fuse(self, data_packets):
        if data_packets:
            for name, packet in data_packets.items():
                if name == "slam":
                    self.handle_slam_packet(packet)
                else:
                    raise NotImplementedError(f"DumpFusion got unexpected input: {name}")
        return None  # like TsdfFusion: no GUI output, keep spinning

    def handle_slam_packet(self, packet):
        if not packet:
            return True
        packet = packet[1]
        if packet is None:
            return True
        if packet["is_last_frame"]:
            print(f"DumpFusion: last frame reached, {self.count} keyframes written.")
            return True

        viz_idx          = packet["viz_idx"]
        kf_idx_to_f_idx  = packet["kf_idx_to_f_idx"]
        poses            = packet["cam0_poses"]                 # w2c (see SE3(...).matrix())
        idepths_up       = packet["cam0_idepths_up"]           # disparity, full-res
        depths_cov_up    = packet["cam0_depths_cov_up"]        # depth-space variance, full-res
        images           = packet["cam0_images"]               # (N,3,H,W), 0..255
        intrinsics       = packet["cam0_intrinsics"]           # [fx,fy,cx,cy] at BA (downsampled) res
        gt_depths        = packet.get("gt_depths", None)

        # world<-cam (c2w) matrices, matching render_volume's convention
        c2w = SE3(poses).inv().matrix().cpu().numpy()          # (N,4,4)

        depths = idepths_up.pow(-1)                            # disparity -> metric depth
        depths = torch.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)

        for i in range(len(viz_idx)):
            f_idx = kf_idx_to_f_idx[viz_idx[i].item()]
            fx, fy, cx, cy = (self.dsf * np.asarray(intrinsics[i].cpu().numpy()))
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            record = slam_interface.Record(
                index=int(f_idx),
                pose_c2w=c2w[i].astype(np.float64),
                depth=depths[i].cpu().numpy().astype(np.float32),          # metres, full-res
                depth_cov=depths_cov_up[i].cpu().numpy().astype(np.float32),  # depth-space variance
                intrinsics=K,
                rgb=images[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8),  # HWC, 0..255
                gt_depth=(gt_depths[i].cpu().numpy() if gt_depths is not None else None),
            )
            slam_interface.write(record, self.out_dir)
            self.count += 1
        return True

    def stop_condition(self):
        return False