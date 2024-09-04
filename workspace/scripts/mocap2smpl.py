import os.path as osp
import numpy as np
from glob import glob

from soma.train.train_soma_multiple import train_multiple_soma

soma_expr_id = 'V48_02_SOMA'

soma_data_settings = [(5, 3, 0.0, 1.0), ] # upto 5 occlusions, upto 3 ghost points, 0.0% real data, 100. % synthetic data
soma_work_base_dir = '/home/ubuntu/Desktop/Mocap2SMPL/workspace'
support_base_dir = osp.join(soma_work_base_dir, 'support_files')
soma_marker_layout_fname = osp.join(support_base_dir, 'marker_layouts/SOMA/soma_subject1/clap_001.c3d')

num_gpus = 1 # number of gpus for training
num_cpus = 4

import os.path as osp
from soma.train.soma_trainer import create_soma_data_id
from soma.run_soma.paper_plots.mosh_soma_dataset import gen_stagei_mocap_fnames_customized
from soma.tools.run_soma_multiple import run_soma_on_multiple_settings
soma_data_ids = [create_soma_data_id(*soma_data_setting) for soma_data_setting in soma_data_settings]
print(soma_data_ids)
mocap_base_dir = osp.join(support_base_dir, 'evaluation_mocaps/original')
soma_mocap_target_ds_name = 'KIT'

run_soma_on_multiple_settings(
        soma_expr_ids=[soma_expr_id],
        soma_mocap_target_ds_names=[
            'KIT'
        ],
        soma_data_ids=soma_data_ids,
        soma_cfg={
            'soma.batch_size': 512,
            'dirs.support_base_dir': support_base_dir,
            'mocap.unit': 'mm',
            'save_c3d': True,
            'keep_nan_points': True,  # required for labeling evaluation
            'remove_zero_trajectories': False  # required for labeling evaluation
        },
        mocap_base_dir=mocap_base_dir,
        run_tasks=['soma'],

        mocap_ext='.c3d',
        soma_work_base_dir = soma_work_base_dir,
        
        parallel_cfg = {
            # 'max_num_jobs': 1, # comment to run on all mocaps
            'randomly_run_jobs': True,
        },
    )

motion_name = 'sweep'
for subject_name in [
    'soma_subject1',
    # 'soma_subject2' # uncomment to process this subject as well
]:
    mocap_dir = osp.join(soma_work_base_dir,
                         'training_experiments',
                         soma_expr_id, soma_data_ids[0],
                         'evaluations',
                         'soma_labeled_mocap_tracklet',
                         soma_mocap_target_ds_name)
    print(f"mocap_dir: {mocap_dir}")
    stagei_mocap_fnames = gen_stagei_mocap_fnames_customized(
                                    motion_name,
                                    mocap_dir, 
                                    subject_name, 
                                    ext='.pkl'
                                )

    run_soma_on_multiple_settings(
        soma_expr_ids=[
            soma_expr_id,
        ],
        soma_mocap_target_ds_names=[
            'KIT'
        ],
        soma_data_ids=
        soma_data_ids,
        mosh_cfg={
            'moshpp.verbosity': 1,  # set to two to visualize the process in psbody.mesh.mesh_viewer
            'moshpp.stagei_frame_picker.stagei_mocap_fnames': stagei_mocap_fnames,
            'moshpp.stagei_frame_picker.type': 'manual',

            'dirs.support_base_dir': support_base_dir,

            'mocap.end_fidx': 10  # comment in real runs
        },
        mocap_base_dir=mocap_base_dir,
        run_tasks=['mosh'],
        fname_filter=[subject_name],
        #         fast_dev_run=True,
        mocap_ext='.c3d',
        soma_work_base_dir=soma_work_base_dir,
        parallel_cfg={
            'max_num_jobs': 1,  # comment to run on all mocaps
            'randomly_run_jobs': True,
        },

    )