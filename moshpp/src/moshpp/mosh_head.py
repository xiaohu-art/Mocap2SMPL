# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
# If you use this code in a research publication please cite the following:
#
# @conference{AMASS:ICCV:2019,
#   title = {{AMASS}: Archive of Motion Capture as Surface Shapes},
#   author = {Mahmood, Naureen and Ghorbani, Nima and Troje, Nikolaus F. and Pons-Moll, Gerard and Black, Michael J.},
#   booktitle = {International Conference on Computer Vision},
#   pages = {5442--5451},
#   month = oct,
#   year = {2019},
#   month_numeric = {10}
# }
#
# You can find complementary content at the project website: https://amass.is.tue.mpg.de/
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
# Naureen Mahmood <https://ps.is.tuebingen.mpg.de/person/nmahmood>
# Matthew Loper <https://ps.is.mpg.de/~mloper>
# While at Max-Planck Institute for Intelligent Systems, Tübingen, Germany
#
# 2021.06.18

import pickle
import sys
import time
from copy import deepcopy
from datetime import timedelta
from glob import glob
from os import path as osp
from pathlib import Path
from typing import List
from typing import Union

import numpy as np
import torch
from human_body_prior.tools.omni_tools import flatten_list
from human_body_prior.tools.omni_tools import get_support_data_dir
from human_body_prior.tools.omni_tools import makepath
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf

from moshpp import frame_picker
from moshpp.marker_layout.create_marker_layout_for_mocaps import marker_labels_to_marker_layout
from moshpp.marker_layout.edit_tools import marker_layout_as_mesh
from moshpp.marker_layout.edit_tools import marker_layout_load
from moshpp.marker_layout.edit_tools import marker_layout_to_c3d
from moshpp.marker_layout.edit_tools import marker_layout_write
from moshpp.marker_layout.labels_map import general_labels_map
from moshpp.tools.run_tools import turn_fullpose_into_parts, setup_mosh_omegaconf_resolvers


class MoSh:
    """
    The role of the head is to ensure a flexible input/output to various implementations of MoSh
    """

    def __init__(self, dict_cfg=None, **kwargs) -> None:
        super(MoSh, self).__init__()

        self.cfg = MoSh.prepare_cfg(dict_cfg=dict_cfg, **kwargs)

        logger.remove()
        if self.cfg.moshpp.verbosity > 0:
            makepath(self.cfg.dirs.log_fname, isfile=True)

            log_format = f"{self.cfg.mocap.session_name} -- {self.cfg.mocap.basename} -- " + \
                         (f"{self.cfg.mocap.subject_name} -- " if self.cfg.mocap.multi_subject else "") + \
                         f"{{module}}:{{function}}:{{line}} -- {{message}}"
            logger.add(self.cfg.dirs.log_fname, format=log_format, enqueue=True)
            logger.add(sys.stdout, colorize=True, format=f"<level>{log_format}</level>", enqueue=True)

        if self.cfg.mocap.multi_subject:
            logger.info('MoCap is multi subject. Available subjects (id:name): {}'.format(
                {sid: sname for sid, sname in enumerate(self.cfg.mocap.subject_names)}))
            logger.info(f'mocap.subject_id: {self.cfg.mocap.subject_id} is selected: {self.cfg.mocap.subject_name}')

        self.stagei_fname = self.cfg.dirs.stagei_fname
        self.stageii_fname = self.cfg.dirs.stageii_fname

        if self.cfg.moshpp.verbosity < 0: return  # this is just a status call

        logger.info(f'mocap_fname: {self.cfg.mocap.fname}')

        logger.info(f'stagei_fname: {self.stagei_fname}')
        logger.info(f'stageii_fname: {self.stageii_fname}')

        assert osp.exists(self.cfg.surface_model.fname), \
            FileNotFoundError(f'surface_model_fname not found: {self.cfg.surface_model.fname}')

        logger.debug(f'surface_model: type: {self.cfg.surface_model.type}; '
                     f'gender: {self.cfg.surface_model.gender}; '
                     f'fname:{self.cfg.surface_model.fname}')

        logger.debug(f'optimize_fingers: {self.cfg.moshpp.optimize_fingers}, '
                     f'optimize_face: {self.cfg.moshpp.optimize_face}, '
                     f'optimize_toes: {self.cfg.moshpp.optimize_toes}, '
                     f'optimize_betas: {self.cfg.moshpp.optimize_betas}, '
                     f'optimize_dynamics: {self.cfg.moshpp.optimize_dynamics}')

        if self.cfg.surface_model.type in ['smplh', 'smplx', 'mano'] and self.cfg.moshpp.optimize_fingers:
            logger.debug(f'optimizing for fingers. dof_per_hand = {self.cfg.surface_model.dof_per_hand}')

        if self.cfg.surface_model.type in ['smplx', 'flame'] and self.cfg.moshpp.optimize_face:
            logger.debug(
                f'optimizing for facial expressions. num_expressions = {self.cfg.surface_model.num_expressions:d}')

        if self.cfg.dirs.marker_layout.fname is None:
            self.cfg.dirs.marker_layout.fname = osp.join(osp.dirname(osp.dirname(self.cfg.mocap.fname)),
                                                         f'{self.cfg.surface_model.type}_{self.cfg.mocap.ds_name}.json')
        self.stagei_data = None
        self.stageii_data = None
        # OmegaConf.create(OmegaConf.to_yaml(self.cfg, resolve=True))
        # todo: find a better way to bake these values. maybe sing the omegaconf cache?
        # i am using OmegaConf.resolve(self.cfg)
        # enabling cache on these fields would do the job however it would disable reevaluation based on chenged subject id.
        # when the field subject id changes we want the gender and subject name to be re-evaluated!
        # we bake the gender into the config so that it is not dependent on the settings.json file
        # self.cfg.surface_model.gender = f"{self.cfg.surface_model.gender}"
        # self.cfg.mocap.subject_name = f"{self.cfg.mocap.subject_name}"
        # self.cfg.mocap.subject_names = self.cfg.mocap.subject_names

    def prepare_stagei_frames(self, stagei_mocap_fnames: List[str] = None):

        frame_picker_cfg = self.cfg.moshpp.stagei_frame_picker
        frame_picker_cfg.num_frames = min(frame_picker_cfg.num_frames, len(stagei_mocap_fnames))

        if stagei_mocap_fnames is None:
            assert frame_picker_cfg.type != 'manual', ValueError(
                'with frame_picker.type manual you should provide list of [/path/to/mocap.c3d_frameid]')
            mocap_base_dir = osp.dirname(self.cfg.mocap.fname)
            mocap_path_split = osp.basename(self.cfg.mocap.fname).split('.')
            mocap_extension = mocap_path_split[-1]

            # if too much mocaps are available choose stagei_num_frames of them; i.e. one frame per sequence
            mocap_fnames = sorted(glob(osp.join(mocap_base_dir, f'*.{mocap_extension}')))

            assert len(mocap_fnames) > 0
            mc_ids = np.random.choice(len(mocap_fnames), self.cfg.moshpp.stagei_frame_picker.num_frames,
                                      replace=False) if len(
                mocap_fnames) > self.cfg.moshpp.stagei_frame_picker.num_frames else np.arange(len(mocap_fnames))
            stagei_mocap_fnames = [mocap_fnames[i] for i in mc_ids]
            logger.debug(f'{len(stagei_mocap_fnames)} subject specific mocap(s) are selected for mosh stagei.')

        logger.debug(
            f'Selecting {frame_picker_cfg.num_frames:d} frames using method {frame_picker_cfg.type} '
            f'on frames with {int(frame_picker_cfg.least_avail_markers * 100.):d}% least_avail_markers')
        if frame_picker_cfg.type == 'random':
            stagei_frames, stagei_fnames = frame_picker.load_marker_sessions_random(stagei_mocap_fnames,
                                                                                    mocap_unit=self.cfg.mocap.unit,
                                                                                    mocap_rotate=self.cfg.mocap.rotate,
                                                                                    num_frames=frame_picker_cfg.num_frames,
                                                                                    seed=frame_picker_cfg.seed,
                                                                                    least_avail_markers=frame_picker_cfg.least_avail_markers,
                                                                                    only_markers=self.cfg.mocap.only_markers,
                                                                                    only_subjects=[
                                                                                        self.cfg.mocap.subject_name] if self.cfg.mocap.multi_subject else None,
                                                                                    exclude_markers=self.cfg.mocap.exclude_markers,
                                                                                    labels_map=general_labels_map)
        elif frame_picker_cfg.type == 'random_strict':
            stagei_frames, stagei_fnames = frame_picker.load_marker_sessions_random_strict(stagei_mocap_fnames,
                                                                                           mocap_unit=self.cfg.mocap.unit,
                                                                                           mocap_rotate=self.cfg.mocap.rotate,
                                                                                           num_frames=frame_picker_cfg.num_frames,
                                                                                           seed=frame_picker_cfg.seed,
                                                                                           least_avail_markers=frame_picker_cfg.least_avail_markers,
                                                                                           only_markers=self.cfg.mocap.only_markers,
                                                                                           only_subjects=[
                                                                                               self.cfg.mocap.subject_name] if self.cfg.mocap.multi_subject else None,
                                                                                           exclude_markers=self.cfg.mocap.exclude_markers,
                                                                                           labels_map=general_labels_map)

        elif frame_picker_cfg.type == 'manual':
            stagei_frames, stagei_fnames = frame_picker.load_marker_sessions_manual(stagei_mocap_fnames,
                                                                                    mocap_unit=self.cfg.mocap.unit,
                                                                                    mocap_rotate=self.cfg.mocap.rotate,
                                                                                    only_markers=self.cfg.mocap.only_markers,
                                                                                    only_subjects=[
                                                                                        self.cfg.mocap.subject_name] if self.cfg.mocap.multi_subject else None,
                                                                                    exclude_markers=self.cfg.mocap.exclude_markers,
                                                                                    labels_map=general_labels_map)

        else:
            raise ValueError(f'Wrong frame_picker value: {frame_picker_cfg.type}')

        logger.debug(f'Using frames for stage-i: {stagei_fnames}')
        return stagei_frames, stagei_fnames

    def mosh_stagei(self, mosh_stagei_func):

        """
        Fitting shape parameters (betas) of the body model to MoCap data.
        It is assumed that all the MoCap data within a folder are performed by the same mosh_subject, and for each mosh_subject it is meaningful
        to optimize shape only once. Therefore, for all the MoCap data within a folder only one shape parameters will be computed.
        :param mosh_stagei_func:
        :param stagei_frames: list of dictionaries with shape values to the body model
        :return:
        """

        if osp.exists(self.stagei_fname):
            self.stagei_data = pickle.load(open(self.stagei_fname, 'rb'))
            prev_surface_model_fname = self.stagei_data['stagei_debug_details']['cfg']['surface_model']['fname']
            assert prev_surface_model_fname == self.cfg.surface_model.fname, \
                ValueError(
                    f'The surface_model_fname used for previous stagei '
                    f'({prev_surface_model_fname}) '
                    f'is different than the current surface model ({self.cfg.surface_model.type})')

            logger.info(f'loading mosh stagei results from {self.stagei_fname}')
        else:
            log_fname = makepath(self.cfg.dirs.stagei_fname.replace('.pkl', '.log'), isfile=True)
            log_format = "{module}:{function}:{line} -- {level} -- {message}"
            stagei_logger_id = logger.add(log_fname, format=log_format, enqueue=True)

            stagei_frames, stagei_fnames = self.prepare_stagei_frames(
                self.cfg.moshpp.stagei_frame_picker.stagei_mocap_fnames)

            if not osp.exists(self.cfg.dirs.marker_layout.fname):
                logger.debug(f'Marker layout not available. It will be produced: {self.cfg.dirs.marker_layout.fname}')
                marker_labels_to_marker_layout(chosen_markers=flatten_list([list(d.keys()) for d in stagei_frames]),
                                               marker_layout_fname=self.cfg.dirs.marker_layout.fname,
                                               surface_model_type=self.cfg.surface_model.type,
                                               labels_map=general_labels_map,
                                               wrist_markers_on_stick=self.cfg.moshpp.wrist_markers_on_stick,
                                               separate_types=self.cfg.moshpp.separate_types,
                                               )
                # todo: check how many chosen markers could not be assigned to a body vertex?

            logger.info(f'Attempting mosh stagei to create {self.stagei_fname}')
            tm = time.time()
            stagei_data = mosh_stagei_func(stagei_frames=stagei_frames, cfg=self.cfg,
                                           betas_fname=self.cfg.moshpp.betas_fname,
                                           v_template_fname=self.cfg.moshpp.v_template_fname)

            stagei_elapsed_time = time.time() - tm

            stagei_data['stagei_debug_details']['stagei_fnames'] = stagei_fnames
            stagei_data['stagei_debug_details']['stagei_frames'] = stagei_frames
            stagei_data['stagei_debug_details']['cfg'] = OmegaConf.to_container(self.cfg, resolve=True,
                                                                                enum_to_str=True)

            stagei_data['stagei_debug_details']['stagei_elapsed_time'] = stagei_elapsed_time

            pickle.dump(stagei_data, open(makepath(self.stagei_fname, isfile=True), 'wb'))

            logger.debug(f'created stagei_fname: {self.stagei_fname}')

            logger.debug(f'finished mosh stagei in {timedelta(seconds=stagei_elapsed_time)}')
            self.stagei_data = stagei_data

            if self.cfg.dirs.write_optimized_marker_layout:
                MoSh.dump_stagei_marker_layout(self.stagei_fname)
            logger.remove(stagei_logger_id)

        return self.stagei_fname

    def mosh_stageii(self, mosh_stageii_func):
        if self.stagei_data is None:
            raise ValueError(f'stagei_fname results could not be found: {self.stagei_fname}. please run stagei first.')

        if osp.exists(self.stageii_fname):
            self.stageii_data = pickle.load(open(self.stageii_fname, 'rb'))
            logger.info(f'loading mosh stageii results from {self.stageii_fname}')

        else:
            logger.info(f'attempting mosh stageii to create {self.stageii_fname}')
            tm = time.time()

            stageii_data = mosh_stageii_func(mocap_fname=self.cfg.mocap.fname,
                                             cfg=self.cfg,
                                             markers_latent=self.stagei_data['markers_latent'],
                                             latent_labels=self.stagei_data['latent_labels'],
                                             betas=self.stagei_data['betas'],
                                             marker_meta=self.stagei_data['marker_meta'],
                                             v_template_fname=self.stagei_data.get('v_template_fname'))
            stageii_elapsed_time = time.time() - tm

            stageii_data.update(self.stagei_data)

            stageii_data['stageii_debug_details']['stageii_elapsed_time'] = stageii_elapsed_time
            stageii_data['stageii_debug_details']['cfg'] = OmegaConf.to_container(self.cfg, resolve=True,
                                                                                  enum_to_str=True)

            pickle.dump(stageii_data, open(makepath(self.stageii_fname, isfile=True), 'wb'))

            logger.debug(f'created stageii_fname: {self.stageii_fname}')
            logger.debug(f'finished mosh stageii in {timedelta(seconds=stageii_elapsed_time)}')
            self.stageii_data = stageii_data

        return self.stageii_fname

    @staticmethod
    def dump_stagei_marker_layout(mosh_stagei_pkl_fname,
                                  out_marker_layout_fname=None,
                                  template_marker_layout_fname: str= None):
        assert mosh_stagei_pkl_fname.endswith('.pkl'), ValueError(f'mosh_stagei_pkl_fname should be a valid pkl file: {mosh_stagei_pkl_fname}')
        mosh_stagei = pickle.load(open(mosh_stagei_pkl_fname, 'rb'))

        marker_meta = MoSh.extract_marker_layout_from_mosh(mosh_stagei,
                                                           template_marker_layout_fname=template_marker_layout_fname)
        if out_marker_layout_fname is None:
            out_marker_layout_fname = mosh_stagei_pkl_fname.replace('.pkl', '.json')

        output_ply_fname = mosh_stagei_pkl_fname.replace('.pkl', '.ply')
        output_c3d_fname = mosh_stagei_pkl_fname.replace('.pkl', '.c3d')

        surface_model_fname = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['fname']
        if surface_model_fname.endswith('.pkl'):
            surface_model_fname = surface_model_fname.replace('.pkl', '.npz')
        marker_layout_write(marker_meta, out_marker_layout_fname)

        body_parms = {}
        if 'betas' in mosh_stagei:
            num_betas = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['num_betas']
            body_parms['betas'] =  torch.Tensor(mosh_stagei['betas'][:num_betas][None])
            body_parms['num_betas'] =  num_betas
        surface_model_type = mosh_stagei['stagei_debug_details']['cfg']['surface_model']['type']
        marker_layout_as_mesh(surface_model_fname,
                              preserve_vertex_order=True,
                              body_parms=body_parms,
                              surface_model_type= surface_model_type)(out_marker_layout_fname,output_ply_fname)
        marker_layout_to_c3d(out_marker_layout_fname,
                             surface_model_fname=surface_model_fname,
                             out_c3d_fname=output_c3d_fname,
                             surface_model_type= surface_model_type)

        logger.info(f'created {out_marker_layout_fname}')
        logger.info(f'created {output_ply_fname}')
        logger.info(f'created {output_c3d_fname}')

    @staticmethod
    def load_as_amass_npz_legacy(stageii_pkl_data_or_fname: Union[dict, Union[str, Path]],
                          stageii_npz_fname: Union[str, Path] = None,
                          stagei_npz_fname: Union[str, Path] = None,
                          include_markers: bool = False,
                          ) -> dict:
        """

                :param stageii_pkl_data_or_fname:
                :param stageii_npz_fname:
                :param stagei_npz_fname:
                :param include_markers:
                :return:
                """
        setup_mosh_omegaconf_resolvers()  # this method could be called on its own

        if isinstance(stageii_pkl_data_or_fname, dict):
            stageii_pkl_data = stageii_pkl_data_or_fname
        else:
            stageii_pkl_data = pickle.load(open(stageii_pkl_data_or_fname, 'rb'), encoding='latin-1')


        cfg = stageii_pkl_data['ps']
        # print(cfg.keys())
        # print(stageii_pkl_data.keys())
        # if not mo['ps']['optimize_dynamics']: print 'does not have dynamics: %s'%mosh_path
        stageii_npz_data = {
            'gender': cfg['gender'],
            'surface_model_type': cfg['fitting_model'],

            'mocap_frame_rate': stageii_pkl_data['mocap_framerate'],
            'mocap_time_length': stageii_pkl_data['mocap_timelength'],

            'markers_latent': stageii_pkl_data['shape_est_lmrks'],
            'latent_labels': stageii_pkl_data['shape_est_lmlabels'],
            'markers_latent_vids': stageii_pkl_data['shape_debug_details']['shape_est_lmrks_vids'],

            'trans': stageii_pkl_data['pose_est_trans'],
            'poses': stageii_pkl_data['pose_est_fullposes'],
        }
        if 'vtemplate_fname' in stageii_pkl_data:
            from psbody.mesh import Mesh
            stageii_npz_data['v_template'] = Mesh(filename=stageii_pkl_data['vtemplate_fname']).v
            stageii_npz_data['v_template_fname'] =stageii_pkl_data['vtemplate_fname']

        optimize_betas = ('vtemplate_fname' not in stageii_pkl_data) and (cfg['betas'] is None)
        if optimize_betas:
            num_betas = cfg['num_betas']
            stageii_npz_data['betas'] = stageii_pkl_data['shape_est_betas'][:num_betas]
            stageii_npz_data['num_betas'] = num_betas

        if cfg['use_dynamics']:
            num_dmpls = cfg['num_dmpls']
            stageii_npz_data['dmpls'] = stageii_pkl_data['pose_est_dmpls'][:,:num_dmpls]
            stageii_npz_data['num_dmpls'] = num_dmpls

        if cfg['optimize_face']:
            num_expressions = cfg['num_expr']
            stageii_npz_data['expression'] = stageii_pkl_data['pose_est_exprs'][:,:num_expressions]
            stageii_npz_data['num_expressions'] = num_expressions

        part_based_pose = turn_fullpose_into_parts(stageii_pkl_data['pose_est_fullposes'], cfg['fitting_model'])
        stageii_npz_data.update(part_based_pose)

        if include_markers:
            marker_layout_fname = cfg['mrk_settings_fname']
            from moshpp.marker_layout.edit_tools import marker_layout_load
            marker_meta = marker_layout_load(marker_layout_fname, only_markers=stageii_pkl_data['shape_est_lmlabels'])
            stageii_npz_data['markers'] = stageii_pkl_data['pose_est_obmrks']
            stageii_npz_data['labels'] = stageii_pkl_data['pose_est_mrk_labels']

            stageii_npz_data['markers_obs'] = stageii_pkl_data['pose_est_obmrks']
            stageii_npz_data['labels_obs'] = stageii_pkl_data['pose_est_mrk_labels']

            stageii_npz_data['markers_sim'] = stageii_pkl_data['pose_est_simmrks']
            stageii_npz_data['marker_meta'] = marker_meta

            stageii_npz_data['num_markers'] = stageii_npz_data['markers'].shape[1]

        if stageii_npz_fname:
            if not osp.exists(stageii_npz_fname):
                np.savez(makepath(stageii_npz_fname, isfile=True), **stageii_npz_data)
                logger.info(f'created amass_stageii_npz_fname: {stageii_npz_fname}')

            if stagei_npz_fname is None:
                stagei_npz_fname = osp.join(osp.dirname(stageii_npz_fname),
                                            f"{cfg['gender']}_stagei.npz")
            if not osp.exists(stagei_npz_fname):
                np.savez(makepath(stagei_npz_fname, isfile=True), **{k: v for k, v in stageii_npz_data.items()
                                                                     if k in ['gender',
                                                                              'surface_model_type',
                                                                              'markers_latent',
                                                                              'latent_labels',
                                                                              'markers_latent_vids',
                                                                              'betas',
                                                                              'v_template']})

                logger.info(f'created amass_stagei_npz_fname: {stagei_npz_fname}')

        return stageii_npz_data


    @staticmethod
    def load_as_amass_npz(stageii_pkl_data_or_fname: Union[dict, Union[str, Path]],
                          stageii_npz_fname: Union[str, Path] = None,
                          stagei_npz_fname: Union[str, Path] = None,
                          include_markers: bool = False,
                          include_extra_details: bool=False,
                          ) -> dict:
        """

        :param stageii_pkl_data_or_fname:
        :param stageii_npz_fname:
        :param stagei_npz_fname:
        :param include_markers:
        :return:
        """
        setup_mosh_omegaconf_resolvers() # this method could be called on its own

        if isinstance(stageii_pkl_data_or_fname, dict):
            stageii_pkl_data = stageii_pkl_data_or_fname
        else:
            try:
                stageii_pkl_data = pickle.load(open(stageii_pkl_data_or_fname, 'rb'))
            except UnicodeDecodeError:
                return MoSh.load_as_amass_npz_legacy(stageii_pkl_data_or_fname, stageii_npz_fname, stagei_npz_fname,
                                                     include_markers)

        cfg = stageii_pkl_data['stageii_debug_details']['cfg']
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # if not mo['ps']['optimize_dynamics']: print 'does not have dynamics: %s'%mosh_path
        stageii_npz_data = {
            'gender': cfg['surface_model']['gender'],
            'surface_model_type': cfg['surface_model']['type'],

            'mocap_frame_rate': stageii_pkl_data['stageii_debug_details']['mocap_frame_rate'],
            'mocap_time_length': stageii_pkl_data['stageii_debug_details']['mocap_time_length'],

            'markers_latent': stageii_pkl_data['markers_latent'],
            'latent_labels': stageii_pkl_data['latent_labels'],
            'markers_latent_vids': stageii_pkl_data['markers_latent_vids'],

            'trans': stageii_pkl_data['trans'],
            'poses': stageii_pkl_data['fullpose'],
        }
        if include_extra_details:
            stageii_npz_data['surface_model_fname'] = cfg['surface_model']['fname']
        if 'v_template' in stageii_pkl_data['stagei_debug_details']:
            stageii_npz_data['v_template'] = stageii_pkl_data['stagei_debug_details']['v_template']

        if cfg.moshpp.optimize_betas:
            stageii_npz_data['betas'] = stageii_pkl_data['betas'][:cfg['surface_model']['num_betas']]
            stageii_npz_data['num_betas'] = cfg['surface_model']['num_betas']

        if cfg.moshpp.optimize_dynamics:
            stageii_npz_data['dmpls'] = stageii_pkl_data['dmpls'][:cfg['surface_model']['num_dmpls']]
            stageii_npz_data['num_dmpls'] = cfg['surface_model']['num_dmpls']

        if cfg.moshpp.optimize_face:
            stageii_npz_data['expression'] = stageii_pkl_data['expression'][:,:cfg['surface_model']['num_expressions']]
            stageii_npz_data['num_expressions'] = cfg['surface_model']['num_expressions']

        part_based_pose = turn_fullpose_into_parts(stageii_pkl_data['fullpose'], cfg['surface_model']['type'])
        stageii_npz_data.update(part_based_pose)

        if include_markers:
            stageii_npz_data['markers'] = stageii_pkl_data['stageii_debug_details']['markers_orig']
            stageii_npz_data['labels'] = stageii_pkl_data['stageii_debug_details']['labels_orig']

            stageii_npz_data['markers_obs'] = stageii_pkl_data['stageii_debug_details']['markers_obs']
            stageii_npz_data['labels_obs'] = stageii_pkl_data['stageii_debug_details']['labels_obs']

            stageii_npz_data['markers_sim'] = stageii_pkl_data['stageii_debug_details']['markers_sim']
            stageii_npz_data['marker_meta'] = stageii_pkl_data['marker_meta']

            stageii_npz_data['num_markers'] = stageii_npz_data['markers'].shape[1]

        if stageii_npz_fname:
            if not osp.exists(stageii_npz_fname):
                np.savez(makepath(stageii_npz_fname, isfile=True), **stageii_npz_data)
                logger.info(f'created amass_stageii_npz_fname: {stageii_npz_fname}')

            if stagei_npz_fname is None:
                stagei_npz_fname = osp.join(osp.dirname(stageii_npz_fname),
                                            f"{cfg['surface_model']['gender']}_stagei.npz")
            if not osp.exists(stagei_npz_fname):
                np.savez(makepath(stagei_npz_fname, isfile=True), **{k: v for k, v in stageii_npz_data.items()
                                                                     if k in ['gender',
                                                                              'surface_model_type',
                                                                              'markers_latent',
                                                                              'latent_labels',
                                                                              'markers_latent_vids',
                                                                              'betas',
                                                                              'v_template']})

                logger.info(f'created amass_stagei_npz_fname: {stagei_npz_fname}')

        return stageii_npz_data

    @staticmethod
    def prepare_cfg(dict_cfg=None, **kwargs) -> DictConfig:
        setup_mosh_omegaconf_resolvers()

        # todo: make the function accept args and kwarg. for args one can provide list of yaml conf file names
        if dict_cfg is None:
            dict_cfg = {}

        app_support_dir = get_support_data_dir(__file__)
        base_cfg = OmegaConf.load(osp.join(app_support_dir, 'conf/moshpp_conf.yaml'))

        override_cfg_dotlist = [f'{k}={v}' if v != None else f'{k}=null' for k, v in kwargs.items()]
        override_cfg = OmegaConf.from_dotlist(override_cfg_dotlist)

        dict_cfg = OmegaConf.create(dict_cfg)

        return OmegaConf.merge(base_cfg, override_cfg, dict_cfg)

    @staticmethod
    def extract_marker_layout_from_mosh(mosh_stagei_pkl_fname: Union[str, dict],
                                        template_marker_layout_fname: str= None) -> dict:
        if not isinstance(mosh_stagei_pkl_fname, dict):
            mosh_stagei = pickle.load(open(mosh_stagei_pkl_fname, 'rb'))
        else:
            mosh_stagei = mosh_stagei_pkl_fname

        opt_marker_vids = mosh_stagei['markers_latent_vids']
        if template_marker_layout_fname:
            # open the marker layout again and replace the optimized vids
            marker_meta = marker_layout_load(template_marker_layout_fname)
        else:
            marker_meta = deepcopy(mosh_stagei['marker_meta'])

        for l, vid in marker_meta['marker_vids'].items():
            if l in opt_marker_vids:
                marker_meta['marker_vids'][l] = opt_marker_vids[l]
                # logger.info(f'updating {l}: {mosh_stagei["marker_meta"]["marker_vids"][l]} ->  {opt_marker_vids[l]}: {marker_meta["marker_vids"][l]}')

        return marker_meta


def run_moshpp_once(cfg):
    """
    This function should be self-contained; i.e. module imports should all be inside
    :param cfg:
    :return:
    """
    from moshpp.chmosh import mosh_stagei, mosh_stageii
    from moshpp.mosh_head import MoSh
    from loguru import logger
    import numpy as np

    mp = MoSh(**cfg)

    mp.mosh_stagei(mosh_stagei)

    logger.debug('Final mosh stagei loss: {}'.format(' | '.join(
        [f'{k} = {np.sum(v ** 2):2.2e}' for k, v in mp.stagei_data['stagei_debug_details']['stagei_errs'].items()])))

    if not mp.cfg.runtime.stagei_only:
        mp.mosh_stageii(mosh_stageii)
        logger.debug('Final mosh stageii loss: {}'.format(' | '.join(
            [f'{k} = {np.sum(v ** 2):2.2e}' for k, v in
             mp.stageii_data['stageii_debug_details']['stageii_errs'].items()])))
