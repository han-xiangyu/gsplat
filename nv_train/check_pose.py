import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def read_images_txt(path):
    centers = []
    names = []

    with open(path) as f:
        lines = f.readlines()

    for i in range(4, len(lines), 2):
        parts = lines[i].strip().split()
        if len(parts) < 10:
            continue

        img_id = parts[0]
        q = np.array(list(map(float, parts[1:5])))
        t = np.array(list(map(float, parts[5:8])))
        name = parts[9]

        # qvec → R
        qw, qx, qy, qz = q
        R = np.array([
            [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw,   2*qx*qz+2*qy*qw],
            [2*qx*qy+2*qz*qw,   1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
            [2*qx*qz-2*qy*qw,   2*qy*qz+2*qx*qw,   1-2*qx*qx-2*qy*qy]
        ])

        C = -R.T @ t
        centers.append(C)
        names.append(name)

    return np.array(centers), names


centers, names = read_images_txt(
    "YOUR_NEW_DATASET/sparse/0/images.txt"
)

# 分三类
orig = [c for c,n in zip(centers,names) if "difix3d" not in n and "up_tilt" not in n]
left = [c for c,n in zip(centers,names) if "_left" in n]
right = [c for c,n in zip(centers,names) if "_right" in n]
up = [c for c,n in zip(centers,names) if "up_tilt" in n]

fig = go.Figure()

def add(pts, name, color):
    if not pts:
        return
    pts = np.array(pts)
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode='markers',
        name=name,
        marker=dict(size=3, color=color)
    ))

add(orig, "original", "black")
add(left, "left", "blue")
add(right, "right", "red")
add(up, "up_tilt", "green")

fig.update_layout(
    title="Camera Centers",
    scene=dict(aspectmode='data')
)

fig.write_html("poses.html")
print("Saved poses.html")
