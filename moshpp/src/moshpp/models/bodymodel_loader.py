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
import warnings

import chumpy as ch
import numpy as np
from loguru import logger

from moshpp.models.smpl_fast_derivatives import SmplModelLBS
from moshpp.models.smpl_fast_derivatives import load_surface_model


def load_dmpl(pkl_path):
    with open(pkl_path) as f:
        dmpl_pcs = pickle.load(f)['eigvec']

    return dmpl_pcs


class AliasedBetas(ch.Ch):
    dterms = 'sv', '_result'

    def __init__(self, surface_model_type='smpl', **kwargs):
        self.surface_model_type = surface_model_type
        super(AliasedBetas, self).__init__()

    def on_changed(self, which):
        # if self.surface_model_type == 'smplx':
        #     self._result = self.sv.body_shape
        # else:
        self._result = self.sv.betas

    def compute_r(self):
        # if self.surface_model_type == 'smplx':
        #     self._result = self.sv.body_shape
        # else:
        return self.sv.betas

    def compute_dr_wrt(self, wrt):
        if wrt is self._result:
            return 1

    def sample(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.random.randn(self.sv.betas.size)


def load_moshpp_models(surface_model_fname,
                       surface_model_type,
                       optimize_face=False,
                       pose_body_prior_fname=None,
                       pose_hand_prior_fname=None,
                       use_hands_mean=False,
                       dof_per_hand=12,
                       v_template_fname=None,
                       num_beta_shared_models=12):
    """
    load model
    """
    logger.info(f'Loading model: {surface_model_fname}')

    if surface_model_type == 'object':
        from moshpp.models.object_model import RigidObjectModel
        can_model = RigidObjectModel(ply_fname=surface_model_fname)
        beta_shared_models = [RigidObjectModel(ply_fname=surface_model_fname) for _ in
                                               range(num_beta_shared_models)]
    else:
        from moshpp.prior.gmm_prior_ch import create_gmm_body_prior

        sm_temp = load_surface_model(surface_model_fname=surface_model_fname,
                                     pose_hand_prior_fname=pose_hand_prior_fname,
                                     use_hands_mean=use_hands_mean,
                                     dof_per_hand=dof_per_hand,
                                     v_template_fname=v_template_fname,
                                     surface_model_type=surface_model_type,
                                     #this is to address the models that have the same number of jonts i.e. dog, rat
                                     )

        betas = ch.array(np.zeros(len(sm_temp.betas)))

        can_model = SmplModelLBS(trans=ch.array(np.zeros(sm_temp.trans.size)),
                                 pose=ch.array(np.zeros(sm_temp.pose.size)),
                                 betas=betas,
                                 temp_model=sm_temp)

        assert can_model.model_type == surface_model_type, ValueError(f'{can_model.model_type} != {surface_model_type}')
        priors = {}
        if surface_model_type == 'animal_horse':
            from moshpp.prior.horse_body_prior import smal_horse_prior, smal_horse_joint_angle_prior
            if pose_body_prior_fname:
                priors['pose'] = smal_horse_prior(pose_body_prior_fname)
            priors['pose_jangles'] = smal_horse_joint_angle_prior()
        elif surface_model_type == 'animal_dog':
            from moshpp.prior.dog_body_prior import MaxMixtureDog
            # from moshpp.prior.smal_dog_prior import smal_dog_prior, smal_dog_joint_angle_prior
            if pose_body_prior_fname:
                priors['pose'] = MaxMixtureDog(prior_pklpath=pose_body_prior_fname).get_gmm_prior()
            # priors['pose_jangles'] = smal_dog_joint_angle_prior()
        else:
            if pose_body_prior_fname:
                priors['pose'] = create_gmm_body_prior(pose_body_prior_fname=pose_body_prior_fname,
                                                   exclude_hands=surface_model_type in ['smplh', 'smplx'])
        if not optimize_face:
            priors['betas'] = AliasedBetas(sv=can_model, surface_model_type=surface_model_type)

        can_model.priors = priors

        # do not share betas when optimizing betas to go around an issue of chumpy
        # with it current implementation of SMPL-X body shape betas and facial expressions share the same parameter; i.e. betas
        # now mosh requires betas to be shared across the canonical model and the 12 frames
        # now those 12 frames cannot share expressions though!
        # and that requires double indexing the same array; i.e. once for body shape and once for expressions
        # that messes up chumpy since it holds a grudge on double indexing the same optimization free variable
        # solution is to precompute betas and to not lock it at all when face needs to be optimized
        beta_shared_models = [SmplModelLBS(pose=ch.array(np.zeros(can_model.pose.size)),
                                           trans=ch.array(np.zeros(can_model.trans.size)),
                                           betas=can_model.betas if not optimize_face else ch.array(np.zeros(can_model.betas.size)),
                                           temp_model=can_model) for _ in range(num_beta_shared_models)]

    return can_model, beta_shared_models
