#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE ####################################
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################




import torch
import math
import time 
import torch.nn.functional as F
import time 




# from scene.oursfull import GaussianModel
from scene.gaussian_model_from_4dgaussian_splatting import GaussianModel
from utils.sh_utils import eval_sh, eval_shfs_4d
from utils.graphics_utils import getProjectionMatrixCV, focal2fov, fov2focal

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer



def train_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""

Tests with the implementation available in 4d-gaussian-splatting/gaussian_renderer/__init__.py

"""
# def train_mine_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
def train_mine_render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction=None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sh_degree_t=pc.active_sh_degree_t,
        campos=viewpoint_camera.camera_center,
        timestamp=viewpoint_camera.timestamp,
        time_duration=pc.time_duration[1]-pc.time_duration[0],
        rot_4d=pc.rot_4d,
        gaussian_dim=pc.gaussian_dim,
        force_sh_3d=pc.force_sh_3d,
        prefiltered=False,
        debug=pipe.debug
    )
"""
    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color if not pipe.env_map_res else torch.zeros(3, device="cuda"),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        sh_degree_t=pc.active_sh_degree_t,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
    )
"""

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    scales_t = None
    rotations = None
    rotations_r = None
    ts = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if pc.rot_4d:
            cov3D_precomp, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
            means3D = means3D + delta_mean
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        if pc.gaussian_dim == 4:
            marginal_t = pc.get_marginal_t(viewpoint_camera.timestamp)
            # marginal_t = torch.clamp_max(marginal_t, 1.0) # NOTE: 这里乘完会大于1，绝对不行——marginal_t应该用个概率而非概率密度 暂时可以clamp一下，后期用积分 —— 2d 也用的clamp
            opacity = opacity * marginal_t
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if pc.gaussian_dim == 4:
            scales_t = pc.get_scaling_t
            ts = pc.get_t
            if pc.rot_4d:
                rotations_r = pc.get_rotation_r

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, pc.get_max_sh_channels)
            if pipe.compute_cov3D_python:
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            else:
                _, delta_mean = pc.get_current_covariance_and_mean_offset(scaling_modifier, viewpoint_camera.timestamp)
                dir_pp = ((means3D + delta_mean) - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)).detach()
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            if pc.gaussian_dim == 3 or pc.force_sh_3d:
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            elif pc.gaussian_dim == 4:
                dir_t = (pc.get_t - viewpoint_camera.timestamp).detach()
                sh2rgb = eval_shfs_4d(pc.active_sh_degree, pc.active_sh_degree_t, shs_view, dir_pp_normalized, dir_t, pc.time_duration[1] - pc.time_duration[0])
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
            if pc.gaussian_dim == 4 and ts is None:
                ts = pc.get_t
    else:
        colors_precomp = override_color
    
    flow_2d = torch.zeros_like(pc.get_xyz[:,:2])
    
    # Prefilter
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        mask = marginal_t[:,0] > 0.05
        if means2D is not None:
            means2D = means2D[mask]
        if means3D is not None:
            means3D = means3D[mask]
        if ts is not None:
            ts = ts[mask]
        if shs is not None:
            shs = shs[mask]
        if colors_precomp is not None:
            colors_precomp = colors_precomp[mask]
        if opacity is not None:
            opacity = opacity[mask]
        if scales is not None:
            scales = scales[mask]
        if scales_t is not None:
            scales_t = scales_t[mask]
        if rotations is not None:
            rotations = rotations[mask]
        if rotations_r is not None:
            rotations_r = rotations_r[mask]
        if cov3D_precomp is not None:
            cov3D_precomp = cov3D_precomp[mask]
        if flow_2d is not None:
            flow_2d = flow_2d[mask]
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha, flow, covs_com = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        flow_2d = flow_2d,
        opacities = opacity,
        ts = ts,
        scales = scales,
        scales_t = scales_t,
        rotations = rotations,
        rotations_r = rotations_r,
        cov3D_precomp = cov3D_precomp)
    
    if pipe.env_map_res:
        assert pc.env_map is not None
        R = 60
        rays_o, rays_d = viewpoint_camera.get_rays()
        delta = ((rays_o*rays_d).sum(-1))**2 - (rays_d**2).sum(-1)*((rays_o**2).sum(-1)-R**2)
        assert (delta > 0).all()
        t_inter = -(rays_o*rays_d).sum(-1)+torch.sqrt(delta)/(rays_d**2).sum(-1)
        xyz_inter = rays_o + rays_d * t_inter.unsqueeze(-1)
        tu = torch.atan2(xyz_inter[...,1:2], xyz_inter[...,0:1]) / (2 * torch.pi) + 0.5 # theta
        tv = torch.acos(xyz_inter[...,2:3] / R) / torch.pi
        texcoord = torch.cat([tu, tv], dim=-1) * 2 - 1
        bg_color_from_envmap = F.grid_sample(pc.env_map[None], texcoord[None])[0] # 3,H,W
        # mask2 = (0 < xyz_inter[...,0]) & (xyz_inter[...,1] > 0) # & (xyz_inter[...,2] > -19)
        rendered_image = rendered_image + (1 - alpha) * bg_color_from_envmap # * mask2[None]
    
    if pipe.compute_cov3D_python and pc.gaussian_dim == 4:
        radii_all = radii.new_zeros(mask.shape)
        radii_all[mask] = radii
    else:
        radii_all = radii

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii_all > 0,
            "radii": radii_all,
            "depth": depth,
            "alpha": alpha,
            "flow": flow}


def train_mine_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_ours_full(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    torch.cuda.synchronize()
    startime = time.time()

    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling
    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays, viewpoint_camera.timestamp) # 1 , 3
    rendered_image = rendered_image.squeeze(0)
    torch.cuda.synchronize()
    duration = time.time() - startime 

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth,
            "duration":duration}
def test_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None,GRsetting=None, GRzer=None):

    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    torch.cuda.synchronize()
    startime = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)
    
    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points
   

    cov3D_precomp = None


    shs = None
 
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    
    torch.cuda.synchronize()
    duration = time.time() - startime 
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "duration":duration}




def train_ours_lite(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "depth": depth}


def train_ours_fullss(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    opacity = pointopacity * trbfoutput  # - 0.5

    pc.trbfoutput = trbfoutput
    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    
    tforpoly = trbfdistanceoffset.detach()

    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly)


    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rawrendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays) # 1 , 3      
    newiamge = F.grid_sample(rawrendered_image, viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(newiamge, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    #rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)# 
  
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "rawrendered_image": rawrendered_image,
            "depth": depth}




def test_ours_fullss_fused(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!

    pc.rgbdecoder.mlp2.weight.shape (3,6,1,1)

    pc.rgbdecoder.mlp2.weight.flatten()[0:6] = pc.rgbdecoder.mlp2.weight[0]

    # https://stackoverflow.com/questions/76949152/why-does-pytorchs-linear-layer-store-the-weight-in-shape-out-in-and-transpos
    # store in (outfeature, infeature) order..
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # mapperss = viewpoint_camera.fisheyemapper.permute(0,3,1,2) 
    # mapperoriginsize =  F.interpolate(mapperss, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    # mapperoriginsize = mapperoriginsize.permute(0,2,3,1)


    torch.cuda.synchronize()
    startime = time.time()#perf_counter

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)
    
    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points
   

    cov3D_precomp = None

    #scales = pc.get_scaling

    shs = None
    # sandwichold 2,3
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        mlp1 = pc.rgbdecoder.mlp2.weight.squeeze(3).squeeze(2).flatten(),
        mlp2 = pc.rgbdecoder.mlp3.weight.squeeze(3).squeeze(2).flatten(),
        rayimage = viewpoint_camera.rays)

    torch.cuda.synchronize()
    duration = time.time() - startime #

    rendered_image = F.grid_sample(rendered_image.unsqueeze(0), viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(rendered_image, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    rendered_image = rendered_image.squeeze(0)


    # rendered_image = F.grid_sample(rendered_image.unsqueeze(0), mapperoriginsize, mode='bicubic', padding_mode='zeros', align_corners=True)
    #  rawrendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays) # 1 , 3      
    ## rendered_image = rendered_image #[0:3,:,:]
    # rendered_image = torch.sigmoid(rendered_image) # sigmoid not fused.
    #rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rawrendered_image": rendered_image,
             "duration": duration}



def test_ours_fullss(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    torch.cuda.synchronize()
    startime = time.time()#perf_counter

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    opacity = pointopacity * trbfoutput  # - 0.5

    pc.trbfoutput = trbfoutput
    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    
    tforpoly = trbfdistanceoffset.detach()

    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly)


    colors_precomp = pc.get_features(tforpoly)
    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rawrendered_image = pc.rgbdecoder(rendered_image.unsqueeze(0), viewpoint_camera.rays) # 1 , 3      
    newiamge = F.grid_sample(rawrendered_image, viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(newiamge, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)

    torch.cuda.synchronize()
    duration = time.time() - startime
    
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "rawrendered_image": rawrendered_image,
            "depth": depth,
            "duration":duration}







def train_ours_litess(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None, GRsetting=None, GRzer=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    pointtimes = torch.ones((pc.get_xyz.shape[0],1), dtype=pc.get_xyz.dtype, requires_grad=False, device="cuda") + 0 # 
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False)

    rasterizer = GRzer(raster_settings=raster_settings)

    
    means3D = pc.get_xyz
    means2D = screenspace_points
    pointopacity = pc.get_opacity

    trbfcenter = pc.get_trbfcenter
    trbfscale = pc.get_trbfscale
   

    trbfdistanceoffset = viewpoint_camera.timestamp * pointtimes - trbfcenter
    trbfdistance =  trbfdistanceoffset / torch.exp(trbfscale) 
    trbfoutput = basicfunction(trbfdistance)
    
    opacity = pointopacity * trbfoutput  # - 0.5
    pc.trbfoutput = trbfoutput

    cov3D_precomp = None

    scales = pc.get_scaling

    shs = None
    tforpoly = trbfdistanceoffset.detach()
    means3D = means3D +  pc._motion[:, 0:3] * tforpoly + pc._motion[:, 3:6] * tforpoly * tforpoly + pc._motion[:, 6:9] * tforpoly *tforpoly * tforpoly

    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    rendered_image, radii, depth = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rawrendered_image = rendered_image.unsqueeze(0)  
    newiamge = F.grid_sample(rawrendered_image, viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(newiamge, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    #rendered_image = torch.clamp(rendered_image, min=0.0, max=1.0)# 
  
    rendered_image = rendered_image.squeeze(0)
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "opacity": opacity,
            "rawrendered_image": rawrendered_image,
            "depth": depth}




def test_ours_litess(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, basicfunction = None,GRsetting=None, GRzer=None):
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0


    torch.cuda.synchronize()


    startime = time.time()

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GRsetting(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False)

    rasterizer = GRzer(raster_settings=raster_settings)    


    tforpoly = viewpoint_camera.timestamp - pc.get_trbfcenter
    rotations = pc.get_rotation(tforpoly) # to try use 
    colors_precomp = pc.get_features(tforpoly)

    
    means2D = screenspace_points
   

    cov3D_precomp = None


    shs = None
 
    rendered_image, radii = rasterizer(
        timestamp = viewpoint_camera.timestamp, 
        trbfcenter = pc.get_trbfcenter,
        trbfscale = pc.computedtrbfscale ,
        motion = pc._motion,
        means3D = pc.get_xyz,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = pc.computedopacity,
        scales = pc.computedscales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    torch.cuda.synchronize()
    duration = time.time() - startime #

    rendered_image = F.grid_sample(rendered_image.unsqueeze(0), viewpoint_camera.fisheyemapper, mode='bicubic', padding_mode='zeros', align_corners=True)
    # INTER_CUBIC
    rendered_image = F.interpolate(rendered_image, size=(int(0.5*viewpoint_camera.image_height) , int(0.5*viewpoint_camera.image_width)), mode='bicubic', align_corners=True)
    rendered_image = rendered_image.squeeze(0)


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "duration":duration}
