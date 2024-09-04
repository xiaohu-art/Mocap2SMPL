import os.path as osp
from moshpp.mosh_head import MoSh
import numpy as np

soma_work_base_dir = '/home/ubuntu/Desktop/test/workspace'
motion_name = 'sweep'
mosh_stageii_pkl_fname = osp.join(soma_work_base_dir, f'training_experiments/V48_02_SOMA/OC_05_G_03_real_000_synt_100/evaluations/mosh_results_tracklet/KIT/soma_subject1/{motion_name}_stageii.pkl')
mosh_result = MoSh.load_as_amass_npz(mosh_stageii_pkl_fname, include_markers=True)
print({k:v if isinstance(v, str) or isinstance(v,float) or isinstance(v,int) else v.shape for k,v in mosh_result.items() if not isinstance(v, list) and not isinstance(v,dict)})

time_length = len(mosh_result['trans'])
mosh_result['betas'] = np.repeat(mosh_result['betas'][None], repeats=time_length, axis=0)

subject_gender = mosh_result['gender']
surface_model_type = mosh_result['surface_model_type']
print(f'subject_gender: {subject_gender}, surface_model_type: {surface_model_type}, time_length: {time_length}')

import trimesh
from body_visualizer.tools.vis_tools import colors
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image

imw, imh=1600, 1600
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

from human_body_prior.body_model.body_model import BodyModel
import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c 

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bm_fname = osp.join(soma_work_base_dir, f'support_files/{surface_model_type}/{subject_gender}/model.npz')

num_betas = mosh_result['num_betas'] # number of body parameters

bm = BodyModel(bm_fname=bm_fname, num_betas=num_betas).to(comp_device)
faces = c2c(bm.f)

body_parms = {k:torch.Tensor(v).to(comp_device) for k,v in mosh_result.items() if k in ['pose_body', 'betas', 'pose_hand']}
print({k:v.shape for k,v in body_parms.items()})

body_pose_hand = bm(**body_parms)

def vis_body_pose_hand(fId = 0):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False)
    show_image(body_image)

# Save the video
import cv2
import os
from tqdm import tqdm

video_name = f"{motion_name}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 30, (imw, imh))

for fId in tqdm(range(time_length)):
    body_mesh = trimesh.Trimesh(vertices=c2c(body_pose_hand.v[fId]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
    mv.set_static_meshes([body_mesh])
    body_image = mv.render(render_wireframe=False) # numpy.ndarray of shape (imh, imw, 3)
    video.write(cv2.cvtColor(body_image, cv2.COLOR_RGB2BGR))

video.release()
print(f'Video saved at {video_name}')