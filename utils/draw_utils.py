#!/usr/bin/python3
"""
@Project: SiamPillar 
@File: draw_utils.py
@Author: Zhuang Yi
@Date: 2020/8/28
"""
from mayavi import mlab
import numpy as np
from config import cfg



def draw_lidar(lidar, is_grid=True, is_axis = False, is_top_region=False, fig=None):

    pxs = lidar[:, 0]
    pys = lidar[:, 1]
    pzs = lidar[:, 2]
    #prs=lidar[:,3]

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=(1, 1, 1), engine=None, size=(1000, 500))

    mlab.points3d(
        pxs, pys, pzs,
        mode='point',  # 'point'  'sphere'
        colormap='gnuplot',  #'bone',  #'spectral',  #'copper',
        scale_factor=1,
        figure=fig)

    #draw grid
    if is_grid:
        mlab.points3d(0, 0, 0, color=(1, 0, 0), mode='sphere', scale_factor=0.2)

        for y in np.arange(-50, 50, 1):
            x1,y1,z1 = -50, y, 0
            x2,y2,z2 = 50, y, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(1, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)

        for x in np.arange(-50, 50, 1):
            x1, y1, z1 = x, -50, 0
            x2, y2, z2 = x, 50, 0
            mlab.plot3d([x1, x2], [y1, y2], [z1, z2], color=(1, 0.5, 0.5), tube_radius=None, line_width=1, figure=fig)

    #draw axis
    if is_axis:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

        axes=np.array([
            [2.,0.,0.,0.],
            [0.,2.,0.,0.],
            [0.,0.,2.,0.],
        ],dtype=np.float64)
        fov=np.array([  ##<todo> : now is 45 deg. use actual setting later ...
            [20., 20., 0.,0.],
            [20.,-20., 0.,0.],
        ],dtype=np.float64)


        mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
        mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
        mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    #draw top_image feature area
    if is_top_region:
        x1 = cfg.SCENE_X_MIN
        x2 = cfg.SCENE_X_MAX
        y1 = cfg.SCENE_Y_MIN
        y2 = cfg.SCENE_Y_MAX
        mlab.plot3d([x1, x1], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x2, x2], [y1, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y1, y1], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)
        mlab.plot3d([x1, x2], [y2, y2], [0,0], color=(0.5,0.5,0.5), tube_radius=None, line_width=1, figure=fig)

    mlab.orientation_axes()
    mlab.view(azimuth=180, elevation=None, distance=50, focalpoint=[2.0909996, -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991

    return fig

def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,0,0), line_width=2) -> object:


    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]

        for k in range(0,4):

            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+3)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

    mlab.view(azimuth=180,elevation=None,distance=50,focalpoint=[2.0909996 , -1.04700089, -2.03249991])#2.0909996 , -1.04700089, -2.03249991
