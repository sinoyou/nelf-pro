from typing import Any, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly import express as ex
import matplotlib as mlp

line_width = 2
plotly_camera_scale = 0.05

def plot_spheres(coordinates, name='default', scatter_size=7, color=None):
    cam_centers_go = []
    if color is None:
        color = 'rgba(0, 255, 0, 1)'
    colors = []
    for idx in range(len(coordinates['center'])):
        colors.append(color)

    for idx in range(len(coordinates['center'])):
        one = go.Scatter3d(
            x = coordinates['center'][[idx], 0], 
            y = coordinates['center'][[idx], 1], 
            z = coordinates['center'][[idx], 2], 
            mode="markers",
            name=name,
            legendgroup=name, 
            marker=dict(color=colors[idx], size=scatter_size),
            hovertext=f'{idx}', 
            showlegend=True if idx == 0 else False,
            opacity=0.5
        )
        cam_centers_go.append(one)
    
    return cam_centers_go

def plot_point3d(xyz, color):
    point_cloud_size = 0.8
    cam_centers_go = go.Scatter3d(
        x = xyz[:, 0], 
        y = xyz[:, 1], 
        z = xyz[:, 2], 
        mode="markers",
        name="sparse point clouds",
        marker=dict(size=point_cloud_size, color=color),
    )
    return [cam_centers_go]

def plot_a_segment(coordinates, camera_group, idx, direction, color, hovertext): 
    _data_line = go.Scatter3d(  # type: ignore
        x=[coordinates['center'][idx, 0], coordinates[direction][idx, 0]],
        y=[coordinates['center'][idx, 1], coordinates[direction][idx, 1]],
        z=[coordinates['center'][idx, 2], coordinates[direction][idx, 2]],
        showlegend=True if idx == 0 else False,  # seperate lines are grouped to manipulate, but only one line appearts in legend. 
        marker=dict(
            size=0.5,
            color=color,
        ),
        legendgroup='{} camera axis {}'.format(camera_group, direction),
        name='{} camera axis {}'.format(camera_group, direction), 
        hovertext=hovertext, 
        line=dict(color=color, width=line_width),
        # visible='legendonly'
    )
    return _data_line

def plot_camera_axis(camera_group, coordinates, image_list):
    # forward - blue, right - red, down = green
    data = []

    forward_axis_color = 'rgba(0, 0, 255, 1)'
    right_axis_color = 'rgba(255, 0, 0, 1)'
    up_axis_color = 'rgba(0, 255, 0, 1)'
    
    def add_a_segment(idx, direction, color, hovertext): 
        _data_line = go.Scatter3d(  # type: ignore
            x=[coordinates['center'][idx, 0], coordinates[direction][idx, 0]],
            y=[coordinates['center'][idx, 1], coordinates[direction][idx, 1]],
            z=[coordinates['center'][idx, 2], coordinates[direction][idx, 2]],
            showlegend=True if idx == 0 else False,  # seperate lines are grouped to manipulate, but only one line appearts in legend. 
            marker=dict(
                size=0.5,
                color=color,
            ),
            legendgroup='{} camera axis {}'.format(camera_group, direction),
            name='{} camera axis {}'.format(camera_group, direction), 
            hovertext=hovertext, 
            line=dict(color=color, width=line_width),
            visible='legendonly'
        )
        return _data_line
    
    for idx, label in enumerate(image_list):
        data.append(add_a_segment(idx, 'forward', forward_axis_color, label+'_forward'))
        data.append(add_a_segment(idx, 'right', right_axis_color, label+'_right'))
        data.append(add_a_segment(idx, 'up', up_axis_color, label+'_up'))
    
    return data

def plot_camera_pyramid(camera_group, coordinates, image_list, special_camera):
    data = []

    def get_color(camera_group, special=False):
        alpha = 0.5 if not special else 1.0
        if camera_group == 'train':
            return f'rgba(0, 0, 255, {alpha})'
        elif camera_group == 'eval':
            return f'rgba(255, 0, 0, {alpha})'
        else:
            return f'rgba(0, 0, 0, {alpha})'

    pyramid_color = get_color(camera_group)
    pyramid_special_color = get_color(camera_group, special=True)
    
    def add_a_pyramid(idx, color, hovertext, surface_opacity=0.05):
        _data = []
        
        ## model mesh (surface only)
        # vertex_names = ['center', 'pyramid_ur', 'pyramid_dr', 'pyramid_ul', 'pyramid_dl']
        # _data_mesh = go.Mesh3d(
        #     x = [coordinates[name][idx, 0] for name in vertex_names], 
        #     y = [coordinates[name][idx, 1] for name in vertex_names], 
        #     z = [coordinates[name][idx, 2] for name in vertex_names], 
        #     showlegend=True if idx == 0 else False,
        #     hovertext=hovertext, 
        #     opacity=surface_opacity,
        #     alphahull=1,
        #     legendgroup='{} camera pyramid surface'.format(camera_group), 
        #     name='{} camera pyramid surface'.format(camera_group), 
        #     color=color
        # )
        # _data.append(_data_mesh)
        
        ## model mesh edge manually
        bvs = ['pyramid_ur', 'pyramid_dr', 'pyramid_dl', 'pyramid_ul']
        
        # bottom square edges
        bottom_square = go.Scatter3d(
            x = [coordinates[name][idx, 0] for name in bvs+[bvs[0]]], 
            y = [coordinates[name][idx, 1] for name in bvs+[bvs[0]]], 
            z = [coordinates[name][idx, 2] for name in bvs+[bvs[0]]], 
            mode='lines', 
            hovertext=hovertext,
            name='{} camera pyramid edge'.format(camera_group), 
            legendgroup='{} camera pyramid edge'.format(camera_group), 
            showlegend=True if idx == 0 else False,
            line=dict(color=color, width=line_width), 
            visible='legendonly'
        )
        _data.append(bottom_square)
        
        # side edges
        for bv in bvs:
            _line = go.Scatter3d(
                x = [coordinates['center'][idx, 0], coordinates[bv][idx, 0]], 
                y = [coordinates['center'][idx, 1], coordinates[bv][idx, 1]], 
                z = [coordinates['center'][idx, 2], coordinates[bv][idx, 2]], 
                mode='lines', 
                hovertext=hovertext,
                name='{} camera pyramid edge'.format(camera_group), 
                legendgroup='{} camera pyramid edge'.format(camera_group), 
                showlegend=False, 
                line=dict(color=color, width=line_width),
                visible='legendonly'
            )
            _data.append(_line)
            
        return _data
    
    for idx, label in enumerate(image_list):
        # draw different color for pyramids of 'special camera'
        if label in special_camera:
            data += add_a_pyramid(idx, pyramid_special_color, label, surface_opacity=0.5)
        else:
            data += add_a_pyramid(idx, pyramid_color, label)
    
    return data

# yzn
def plot_camera_components(coordinates, image_list, white_list=None, special_camera=None, camera_group=None):
    """
    The function generates all components related to cameras (e.g. axis, pyramid)
    coordinates: dict, containing vertices for drawing camera axis and pyramid. (world coordinate)
    image_list: list, string name of full camera, same index order as in coordinates. 
    white_list: list, string name of cameras to be draw. (default None, which means full drawing). 
    special_camera: list, string name of cameras which needs special color. (default None, which means empty). 
    """
    white_coordinates = dict()
    white_image_list = list()
    
    if white_list is None:
        white_coordinates = coordinates
        white_image_list = image_list
    else:
        index = [image_list.index(name) for name in white_list]
        white_image_list = [image_list[i] for i in index]
        for k,v in coordinates.items():
            white_coordinates[k] = coordinates[k][index]
        
    if special_camera is None:
        special_camera = []
        
    _data = []
    # camera axis
    fig_axis = plot_camera_axis(camera_group, white_coordinates, white_image_list)
    _data += fig_axis

    # camera viewpoint pyramid
    fig_pyramid = plot_camera_pyramid(camera_group, white_coordinates, white_image_list, special_camera)
    _data += fig_pyramid
    
    return _data