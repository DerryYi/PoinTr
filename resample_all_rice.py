import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

ROOT = "data/Rice"
SPLITS = ["train", "val", "test"]
TARGET_N = 16384   # 必须和 config 里一致

def resample_pcd(path, target_n):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)

    if pts.shape[0] == 0:
        print("EMPTY:", path)
        return

    if pts.shape[0] >= target_n:
        idx = np.random.choice(pts.shape[0], target_n, replace=False)
    else:
        idx = np.random.choice(pts.shape[0], target_n, replace=True)

    pts_new = pts[idx]
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(pts_new)
    o3d.io.write_point_cloud(path, pcd_new)

def main():
    all_files = []

    for split in SPLITS:
        for sub in ["complete", "partial"]:
            base = os.path.join(ROOT, split, sub)

            for root, dirs, files in os.walk(base):
                for f in files:
                    if f.endswith(".pcd"):
                        all_files.append(os.path.join(root, f))

    print("Total pcd files:", len(all_files))
    print("Resampling to:", TARGET_N)

    for f in tqdm(all_files):
        resample_pcd(f, TARGET_N)

    print("\nAll done. Every point cloud now has", TARGET_N, "points.")

if __name__ == "__main__":
    main()
