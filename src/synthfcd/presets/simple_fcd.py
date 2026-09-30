"""
Default parameter ranges for simulation pipelines.
"""

from __future__ import annotations

__all__ = [
    "draw_simple_fcd_params",
    "get_simple_fcd_presets",
]

from copy import deepcopy
from typing import Any, cast

from synthfcd.presets.draw_params import Draw, draw_params
from synthfcd.presets.utils import (
    sample_neg_trunc_lognormal,
    sample_trunc_lognormal,
    seed_params,
)
from synthfcd.utils._aliases import (
    _FCDType,
    _LabelEnumType,
    _RNGType,
    _sMRIType,
)
from synthfcd.utils._base import _LabelEnum
from synthfcd.utils._validators import (
    validate_class_type,
    validate_literal_str,
    validate_obj_type,
)
from synthfcd.utils.misc import (
    get_leaf_paths_dict,
    get_rng,
    replace_paths_dict,
)
from synthfcd.utils.seg_labels import SynthSegLabel


def get_default_simple_fcd_presets(
    label_enum: _LabelEnumType = SynthSegLabel,
) -> dict[str, Any]:
    """
    Return the default simple FCD presets.
    """
    validate_class_type(label_enum, "label_enum", _LabelEnum)
    return {
        "FCD_type": Draw(
            "choice",
            (
                (
                    "FCD_type_Ia",
                    "FCD_type_Ib",
                    "FCD_type_Ic",
                    "FCD_type_IIa",
                    "FCD_type_IIb",
                ),
            ),
            {"p": [0.15, 0.1, 0.05, 0.35, 0.35]},
        ),
        "FCD_type_Ia": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((100, 1000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.9, 1.0),)),
                "growth": "distance",
                "lobe": label_enum.get_temporal_lobe_labels(),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.0),)),
                    "pial_lower_bound": -2.0,
                    "gwb_upper_bound": 2.0,
                    "edge_rolloff": 1.0,
                    "corr_sigma": Draw(sample_trunc_lognormal, ((3.0, 20.0),)),
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(
                        sample_trunc_lognormal, ((3.0, 6.0),)
                    ),  # large, should deform whole cortical ribbon
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": False,
                },
                "sulcal_widening": {
                    "enable": True,
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.4, 0.8),)),
                    "pial_lower_bound": -1.0,
                    "pial_upper_bound": 2.0,
                    "gwb_upper_bound": -1.0,
                    "edge_rolloff": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 4.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (5, 15)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": 1.0,
                    "edge_rolloff": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.01,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.01,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw(
                        "uniform", (1.0, 2.0)
                    ),  # additive to `wmh_seeds_corr`
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "start_prob": Draw("uniform", (0.3, 0.5)),
                    "end_prob": 0.0,
                    "n_iters": Draw("integers", (10, 20)),
                    "scale_factor": Draw(
                        sample_trunc_lognormal, ((0.1, 0.2),)
                    ),  # use for `t1_*` settings
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.05,
                    # Extra for cross-modal translation (T2like->T1like):
                    # only used inside `draw_simple_fcd_params` to adjust the `scale_factor`
                    "T1like_settings": {
                        "scale_factor_shrink": Draw(
                            "uniform", (0.4, 0.8)
                        ),  # shrinkage factor of `scale_factor`
                        "sign": -1,  # hyper/hypo-intensity flag
                    },
                },
            },
        },
        "FCD_type_Ib": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((100, 800),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.9, 1.0),)),
                "growth": "distance",
                "lobe": (
                    label_enum.get_cortical_labels()
                    - label_enum.get_temporal_lobe_labels()
                    - label_enum.get_frontal_lobe_labels()
                ),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.0),)),
                    "pial_lower_bound": -2.0,
                    "gwb_upper_bound": 2.0,
                    "edge_rolloff": 1.0,
                    "corr_sigma": Draw(sample_trunc_lognormal, ((3.0, 20.0),)),
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 6.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": False,
                },
                "sulcal_widening": {
                    "enable": True,
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.4, 0.8),)),
                    "pial_lower_bound": -1.0,
                    "pial_upper_bound": 2.0,
                    "gwb_upper_bound": -1.0,
                    "edge_rolloff": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 4.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (5, 15)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": Draw(
                        sample_neg_trunc_lognormal, ((-1.0, -0.01),)
                    ),
                    "edge_rolloff": 1.0,
                    "app_field_type": "inward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.01,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "inward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.01,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw(
                        "uniform", (1.0, 2.0)
                    ),  # TODO: add to `wmh_seeds_corr`
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "start_prob": Draw("uniform", (0.3, 0.5)),
                    "end_prob": 0.0,
                    "n_iters": Draw("integers", (10, 20)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.1, 0.2),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.4, 0.8)),
                        "sign": -1,
                    },
                },
            },
        },
        "FCD_type_Ic": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((100, 1000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.9, 1.0),)),
                "growth": "distance",
                "lobe": label_enum.get_cortical_labels()
                - label_enum.get_frontal_lobe_labels(),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.0),)),
                    "pial_lower_bound": -2.0,
                    "gwb_upper_bound": 2.0,
                    "edge_rolloff": 1.0,
                    "corr_sigma": Draw(sample_trunc_lognormal, ((3.0, 20.0),)),
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 6.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": False,
                },
                "sulcal_widening": {
                    "enable": True,
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.4, 0.8),)),
                    "pial_lower_bound": -1.0,
                    "pial_upper_bound": 2.0,
                    "gwb_upper_bound": -1.0,
                    "edge_rolloff": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 4.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (5, 15)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": 1.0,
                    "edge_rolloff": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.01,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.01,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw("uniform", (1.0, 2.0)),
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "start_prob": Draw("uniform", (0.3, 0.5)),
                    "end_prob": 0.0,
                    "n_iters": Draw("integers", (10, 20)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.1, 0.2),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.4, 0.8)),
                        "sign": -1,
                    },
                },
            },
        },
        "FCD_type_IIa": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((800, 2000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.8, 0.9),)),
                "growth": "distance",
                "lobe": Draw(
                    "choice",
                    (
                        (
                            label_enum.get_frontal_lobe_labels(),
                            label_enum.get_cortical_labels(),
                        ),
                    ),
                    {"p": [0.7, 0.3]},
                ),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.0),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "corr_sigma": Draw(sample_trunc_lognormal, ((3.0, 20.0),)),
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 6.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": True,
                    "n_iters": Draw("integers", (1, 3)),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.0),)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.5, 4.0),)),
                    "bbox_thres": 0.01,
                },
                "sulcal_widening": {
                    "enable": "growth_params.bottom_of_sulcus",
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 0.8),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-1.5, -0.5),)
                    ),
                    "pial_upper_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (20, 50)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.5),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.05,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.05,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw("uniform", (1.0, 2.0)),
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((3.0, 5.0),)),
                    "start_prob": Draw("uniform", (0.4, 0.8)),
                    "end_prob": Draw("uniform", (0.1, 0.3)),
                    "n_iters": Draw("integers", (20, 50)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.2, 0.4),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.5, 2.5),)),
                    "app_field_type": "inward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.3, 0.6)),
                        "sign": -1,
                    },
                },
            },
        },
        "FCD_type_IIb": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((1000, 10000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.6, 0.8),)),
                "growth": "distance",
                "lobe": Draw(
                    "choice",
                    (
                        (
                            label_enum.get_frontal_lobe_labels(),
                            label_enum.get_cortical_labels(),
                        ),
                    ),
                    {"p": [0.8, 0.2]},
                ),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.5, 0.5]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.0),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "corr_sigma": Draw(sample_trunc_lognormal, ((3.0, 20.0),)),
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 6.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": True,
                    "n_iters": Draw("integers", (1, 3)),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.0),)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.5, 4.0),)),
                    "bbox_thres": 0.01,
                },
                "sulcal_widening": {
                    "enable": "growth_params.bottom_of_sulcus",
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 0.8),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-1.5, -0.5),)
                    ),
                    "pial_upper_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (20, 50)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.5),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.05,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.05,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw("uniform", (1.0, 2.0)),
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((3.0, 7.0),)),
                    "start_prob": Draw("uniform", (0.4, 0.8)),
                    "end_prob": Draw("uniform", (0.1, 0.3)),
                    "n_iters": Draw("integers", (20, 70)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.2, 0.45),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "app_field_type": "inward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.3, 0.6)),
                        "wmh_thres": 0.3,  # lower `scale_factor` to consider T1like hyperintensity
                        "sign": Draw("choice", ((-1, 1),), {"p": [0.9, 0.1]}),
                    },
                },
            },
        },
    }


def get_extreme_simple_fcd_presets(
    label_enum: _LabelEnumType = SynthSegLabel,
) -> dict[str, Any]:
    """
    Return the extreme simple FCD presets.
    """
    validate_class_type(label_enum, "label_enum", _LabelEnum)
    return {
        "FCD_type": Draw(
            "choice",
            (
                (
                    "FCD_type_Ia",
                    "FCD_type_Ib",
                    "FCD_type_Ic",
                    "FCD_type_IIa",
                    "FCD_type_IIb",
                ),
            ),
            {"p": [0.15, 0.1, 0.05, 0.35, 0.35]},
        ),
        "FCD_type_Ia": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((100, 1000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.9, 1.0),)),
                "growth": "distance",
                "lobe": label_enum.get_temporal_lobe_labels(),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.2),)),
                    "pial_lower_bound": -2.0,
                    "gwb_upper_bound": 2.0,
                    "edge_rolloff": 1.0,
                    "corr_sigma": Draw(sample_trunc_lognormal, ((2.0, 25.0),)),
                    "smooth_sigma_field": 1.0,
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(
                        sample_trunc_lognormal, ((3.0, 6.0),)
                    ),  # large, should deform whole cortical ribbon
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": False,
                },
                "sulcal_widening": {
                    "enable": True,
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.4, 0.8),)),
                    "pial_lower_bound": -1.0,
                    "pial_upper_bound": 2.0,
                    "gwb_upper_bound": -1.0,
                    "edge_rolloff": 1.0,
                    "smooth_sigma_grad": None,
                    "smooth_sigma_field": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 4.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (5, 50)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": 1.0,
                    "edge_rolloff": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.01,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.01,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw(
                        "uniform", (1.0, 2.0)
                    ),  # additive to `wmh_seeds_corr`
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((1.0, 3.0),)),
                    "start_prob": Draw("uniform", (0.3, 0.7)),
                    "end_prob": 0.0,
                    "n_iters": Draw("integers", (10, 50)),
                    "scale_factor": Draw(
                        sample_trunc_lognormal, ((0.1, 0.3),)
                    ),  # use for `t1_*` settings
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.05,
                    # Extra for cross-modal translation (T2like->T1like):
                    # only used inside `draw_simple_fcd_params` to adjust the `scale_factor`
                    "T1like_settings": {
                        "scale_factor_shrink": Draw(
                            "uniform", (0.4, 0.8)
                        ),  # shrinkage factor of `scale_factor`
                        "sign": -1,  # hyper/hypo-intensity flag
                    },
                },
            },
        },
        "FCD_type_Ib": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((100, 800),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.9, 1.0),)),
                "growth": "distance",
                "lobe": (
                    label_enum.get_cortical_labels()
                    - label_enum.get_temporal_lobe_labels()
                    - label_enum.get_frontal_lobe_labels()
                ),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.2),)),
                    "pial_lower_bound": -2.0,
                    "gwb_upper_bound": 2.0,
                    "edge_rolloff": 1.0,
                    "corr_sigma": Draw(sample_trunc_lognormal, ((2.0, 25.0),)),
                    "smooth_sigma_field": 1.0,
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 6.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": False,
                },
                "sulcal_widening": {
                    "enable": True,
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.4, 0.8),)),
                    "pial_lower_bound": -1.0,
                    "pial_upper_bound": 2.0,
                    "gwb_upper_bound": -1.0,
                    "edge_rolloff": 1.0,
                    "smooth_sigma_grad": None,
                    "smooth_sigma_field": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 4.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (5, 50)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": Draw(
                        sample_neg_trunc_lognormal, ((-1.0, -0.01),)
                    ),
                    "edge_rolloff": 1.0,
                    "app_field_type": "inward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.01,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "inward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.01,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw(
                        "uniform", (1.0, 2.0)
                    ),  # TODO: add to `wmh_seeds_corr`
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "start_prob": Draw("uniform", (0.3, 0.7)),
                    "end_prob": 0.0,
                    "n_iters": Draw("integers", (10, 50)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.1, 0.3),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.4, 0.8)),
                        "sign": -1,
                    },
                },
            },
        },
        "FCD_type_Ic": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((100, 1000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.9, 1.0),)),
                "growth": "distance",
                "lobe": label_enum.get_cortical_labels()
                - label_enum.get_frontal_lobe_labels(),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.2),)),
                    "pial_lower_bound": -2.0,
                    "gwb_upper_bound": 2.0,
                    "edge_rolloff": 1.0,
                    "corr_sigma": Draw(sample_trunc_lognormal, ((2.0, 25.0),)),
                    "smooth_sigma_field": 1.0,
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 6.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": False,
                },
                "sulcal_widening": {
                    "enable": True,
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.4, 0.8),)),
                    "pial_lower_bound": -1.0,
                    "pial_upper_bound": 2.0,
                    "gwb_upper_bound": -1.0,
                    "edge_rolloff": 1.0,
                    "smooth_sigma_grad": None,
                    "smooth_sigma_field": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 4.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (5, 50)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": 1.0,
                    "edge_rolloff": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.01,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.01,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw("uniform", (1.0, 2.0)),
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "start_prob": Draw("uniform", (0.3, 0.7)),
                    "end_prob": 0.0,
                    "n_iters": Draw("integers", (10, 50)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.1, 0.3),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.4, 0.8)),
                        "sign": -1,
                    },
                },
            },
        },
        "FCD_type_IIa": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((800, 2000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.8, 0.9),)),
                "growth": "distance",
                "lobe": Draw(
                    "choice",
                    (
                        (
                            label_enum.get_frontal_lobe_labels(),
                            label_enum.get_cortical_labels(),
                        ),
                    ),
                    {"p": [0.7, 0.3]},
                ),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.7, 0.3]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.2),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "corr_sigma": Draw(sample_trunc_lognormal, ((2.0, 25.0),)),
                    "smooth_sigma_field": 1.0,
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 6.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": True,
                    "n_iters": Draw("integers", (1, 3)),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.2),)),
                    "pial_lower_bound": Draw(
                        sample_trunc_lognormal, ((1.3, 2.0),)
                    ),  # must exceed mm_per_iter
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "smooth_sigma_grad": None,
                    "smooth_sigma_field": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.5, 4.0),)),
                    "bbox_thres": 0.01,
                },
                "sulcal_widening": {
                    "enable": "growth_params.bottom_of_sulcus",
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 0.8),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-1.5, -0.5),)
                    ),
                    "pial_upper_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "smooth_sigma_grad": None,
                    "smooth_sigma_field": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (20, 80)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.5),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 4.0),)),
                    "bbox_thres": 0.05,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.05,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw("uniform", (1.0, 2.0)),
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((3.0, 5.0),)),
                    "start_prob": Draw("uniform", (0.4, 0.8)),
                    "end_prob": Draw("uniform", (0.1, 0.3)),
                    "n_iters": Draw("integers", (20, 60)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.2, 0.45),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.5, 2.5),)),
                    "app_field_type": "inward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 4.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.3, 0.6)),
                        "sign": -1,
                    },
                },
            },
        },
        "FCD_type_IIb": {
            "growth_params": {
                "volume": Draw(sample_trunc_lognormal, ((1000, 15000),)),
                "gm_prob": Draw(sample_trunc_lognormal, ((0.5, 0.8),)),
                "growth": "distance",
                "lobe": Draw(
                    "choice",
                    (
                        (
                            label_enum.get_frontal_lobe_labels(),
                            label_enum.get_cortical_labels(),
                        ),
                    ),
                    {"p": [0.8, 0.2]},
                ),
                "bottom_of_sulcus": Draw("choice", ((True, False),), {"p": [0.3, 0.7]}),
            },
            "deformation_params": {
                "abnormal_gyration": {
                    "enable": True,
                    "n_iters": Draw("choice", ((1, 2),), {"p": [0.5, 0.5]}),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.2),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "corr_sigma": Draw(sample_trunc_lognormal, ((2.0, 40.0),)),
                    "smooth_sigma_field": 1.0,
                    "scaling_and_squaring_steps": 5,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((3.0, 7.0),)),
                    "bbox_thres": 0.01,
                },
                "cortical_thickening": {
                    "enable": True,
                    "n_iters": Draw("integers", (1, 3)),
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 1.2),)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.3, 2.0),)),
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "smooth_sigma_grad": None,
                    "smooth_sigma_field": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.5, 4.0),)),
                    "bbox_thres": 0.01,
                },
                "sulcal_widening": {
                    "enable": "growth_params.bottom_of_sulcus",
                    "n_iters": 1,
                    "mm_per_iter": Draw(sample_trunc_lognormal, ((0.5, 0.8),)),
                    "pial_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-1.5, -0.5),)
                    ),
                    "pial_upper_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_upper_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "smooth_sigma_grad": None,
                    "smooth_sigma_field": 1.0,
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "bbox_thres": 0.01,
                },
            },
            "intensity_params": {
                "boundary_blurring": {
                    "enable": True,
                    "n_iters": Draw("integers", (20, 80)),
                    "pial_lower_bound": Draw(sample_trunc_lognormal, ((1.0, 2.0),)),
                    "gwb_lower_bound": Draw(
                        sample_neg_trunc_lognormal, ((-3.0, -2.0),)
                    ),
                    "gwb_upper_bound": Draw(sample_trunc_lognormal, ((2.0, 4.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((1.0, 2.5),)),
                    "app_field_type": "outward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 5.0),)),
                    "bbox_thres": 0.05,
                },
                "texture_restoration": {
                    "enable": True,
                    "gm_wm_only": True,
                    "hpf_img_sigma": Draw("uniform", (0.8, 1.2)),
                    "corr_noise_sigma": Draw("uniform", (0.7, 1.0)),
                    "hpf_noise_sigma": Draw("uniform", (2.0, 2.5)),
                    "noise_clip": Draw("uniform", (2.5, 3.5)),
                    "app_field_type": "outward",
                    "app_rolloff": "intensity_params.boundary_blurring.app_rolloff",
                    "bbox_thres": 0.05,
                },
                "hyperintensity": {
                    "enable": True,
                    "wmh_seeds_corr": Draw("uniform", (0.8, 2.0)),
                    "wmh_seeds_hpf": Draw("uniform", (1.0, 2.0)),
                    "gwb_prob_rolloff": Draw(sample_trunc_lognormal, ((3.0, 8.0),)),
                    "start_prob": Draw("uniform", (0.4, 0.8)),
                    "end_prob": Draw("uniform", (0.1, 0.5)),
                    "n_iters": Draw("integers", (20, 80)),
                    "scale_factor": Draw(sample_trunc_lognormal, ((0.2, 0.5),)),
                    "gm_rolloff": Draw(sample_trunc_lognormal, ((2.0, 4.0),)),
                    "edge_rolloff": Draw(sample_trunc_lognormal, ((2.0, 3.0),)),
                    "app_field_type": "inward",
                    "app_rolloff": Draw(sample_trunc_lognormal, ((2.0, 5.0),)),
                    "bbox_thres": 0.05,
                    "T1like_settings": {
                        "scale_factor_shrink": Draw("uniform", (0.3, 0.8)),
                        "wmh_thres": 0.3,  # lower `scale_factor` to consider T1like hyperintensity
                        "sign": Draw("choice", ((-1, 1),), {"p": [0.9, 0.1]}),
                    },
                },
            },
        },
    }


def get_minimal_simple_fcd_presets(
    label_enum: _LabelEnumType = SynthSegLabel,
) -> dict[str, Any]:
    """
    Disable abnormal gyration for all FCD types.
    """
    presets = get_default_simple_fcd_presets(label_enum)
    # The effect blocks live under each FCD-type subdict, not at the top level.
    # The top level also holds the "FCD_type" chooser, which must be skipped.
    fcd_types = [k for k in presets if k != "FCD_type"]
    for t in fcd_types:
        presets[t]["deformation_params"]["abnormal_gyration"]["enable"] = False
    return presets


def get_wmh_only_simple_fcd_presets(
    label_enum: _LabelEnumType = SynthSegLabel,
) -> dict[str, Any]:
    """
    Disable all effects except hyperintensity.
    """
    presets = get_default_simple_fcd_presets(label_enum)
    fcd_types = [k for k in presets if k != "FCD_type"]
    for t in fcd_types:
        deformation = presets[t]["deformation_params"]
        intensity = presets[t]["intensity_params"]
        deformation["abnormal_gyration"]["enable"] = False
        deformation["cortical_thickening"]["enable"] = False
        deformation["sulcal_widening"]["enable"] = False
        intensity["hyperintensity"]["enable"] = True
        intensity["boundary_blurring"]["enable"] = False
        intensity["texture_restoration"]["enable"] = False
    return presets


def get_type_II_only_simple_fcd_presets(
    label_enum: _LabelEnumType = SynthSegLabel,
) -> dict[str, Any]:
    """
    Disable all *_Type_I* subtypes.
    """
    presets = get_default_simple_fcd_presets(label_enum)
    presets["FCD_type"] = Draw(
        "choice",
        (
            (
                "FCD_type_Ia",
                "FCD_type_Ib",
                "FCD_type_Ic",
                "FCD_type_IIa",
                "FCD_type_IIb",
            ),
        ),
        {"p": [0.0, 0.0, 0.0, 0.5, 0.5]},
    )
    return presets


def get_simple_fcd_presets(
    key: str = "default",
    label_enum: _LabelEnumType = SynthSegLabel,
) -> dict[str, Any]:
    """
    Return the simple FCD presets for a given key.

    Args:
        key (str, optional):
            The key to get the presets for. Defaults to ``"default"``.
            Available options are:
            - ``"default"``: default presets (see ``synthfcd.get_default_simple_fcd_presets``)
            - ``"extreme"``: presets with more extreme ranges
                (see ``synthfcd.get_extreme_simple_fcd_presets``)
            - ``"minimal"``: default presets with abnormal gyration disabled
            - ``"wmh_only"``: default presets with only hyperintensity enabled
            - ``"type_II_only"``: default presets with only type IIa and type IIb FCD enabled
            Defaults to ``"default"``.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading the segmentation mask.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            The simple FCD presets for the given key.

    Raises:
        ValueError: If the key is not valid.
    """
    if key == "default":
        return get_default_simple_fcd_presets(label_enum)

    if key == "extreme":
        return get_extreme_simple_fcd_presets(label_enum)

    if key == "minimal":
        return get_minimal_simple_fcd_presets(label_enum)

    if key == "wmh_only":
        return get_wmh_only_simple_fcd_presets(label_enum)

    if key == "type_II_only":
        return get_type_II_only_simple_fcd_presets(label_enum)

    raise ValueError(f"Unknown simple FCD presets key: {key}")


def draw_simple_fcd_params(
    sequences: list[_sMRIType],
    *,
    presets: str | None = "default",
    custom_presets: dict[str, Any] | None = None,
    fcd_type: _FCDType | None = None,
    random_seed: int | None = None,
    random_state: _RNGType | None = None,
    label_enum: _LabelEnumType = SynthSegLabel,
) -> dict[str, Any]:
    """
    Draw concrete parameter values from the SimpleFCD presets.

    Args:
        sequences (list[_sMRIType]):
            The list of sequences to draw parameters for.
        presets (str, optional):
            The presets to draw parameters from. Available options are:
            - ``"default"``: default presets (see ``synthfcd.get_simple_fcd_presets``)
            - ``"extreme"``: presets with more extreme ranges
                (see ``synthfcd.get_extreme_simple_fcd_presets``)
            - ``"minimal"``: default presets with abnormal gyration disabled
            - ``"wmh_only"``: default presets with only hyperintensity enabled
            - ``"type_II_only"``: default presets with only type IIa and type IIb FCD enabled
            Defaults to ``"default"``. Can be overridden with a custom presets dictionary.
        custom_presets (dict[str, Any] | None, optional):
            Custom presets to use instead of any of the available presets. If provided,
            the value  of ``presets`` is ignored. Must match the structure of available
            presets. Defaults to ``None``.
        fcd_type (str | None, optional):
            The FCD type to draw parameters for. If ``None``, a random FCD type is drawn.
        random_seed (int | None, optional):
            The random seed to use for the random number generator.
        random_state (np.random.Generator | np.random.RandomState | None, optional):
            The random number generator to use for the random number generator.
        label_enum (type[LabelEnum], optional):
            The label enumeration to use for reading the segmentation mask.
            Defaults to ``synthfcd.utils.SynthSegLabel``.

    Returns:
        dict[str, Any]:
            A dictionary containing the following keys:
            - ``"FCD_type"`` : the FCD type (e.g., "FCD_type_Ia", "FCD_type_Ib", etc.)
            - ``"growth_params"``: the growth parameters dictionary
            - ``"deformation_params"``: the deformation parameters dictionary
            - ``"intensity_params"``: the intensity parameters list (one dictionary per sequence)

    Raises:
        TypeError: If `sequences` is not a list.
        ValueError: If `sequences` is empty or contains invalid values or if any
            custom presets are malformed.
    """
    validate_obj_type(sequences, name="sequences", target_type=list)
    if len(sequences) == 0:
        raise ValueError("`sequences` must be a non-empty list")
    for i, m in enumerate(sequences):
        validate_literal_str(
            m, name=f"sequences[{i}]", target_literals=("T1like", "T2like")
        )

    if presets is not None:
        _presets = get_simple_fcd_presets(presets, label_enum)
    else:
        if custom_presets is None:
            raise ValueError("Either `presets` or `custom_presets` must be provided.")

    if custom_presets is not None:
        validate_obj_type(custom_presets, name="custom_presets", target_type=dict)
        if get_leaf_paths_dict(custom_presets) != get_leaf_paths_dict(
            get_simple_fcd_presets(key="default", label_enum=label_enum)
        ):
            raise ValueError(
                "`custom_presets` must contain the same keys and structure as "
                "the default presets (see `synthfcd.get_simple_fcd_presets`)."
            )
        _presets = custom_presets

    if fcd_type is not None:
        validate_literal_str(
            fcd_type,
            name="fcd_type",
            target_literals=(
                "FCD_type_Ia",
                "FCD_type_Ib",
                "FCD_type_Ic",
                "FCD_type_IIa",
                "FCD_type_IIb",
            ),
        )

    rng = get_rng(seed=random_seed, random_state=random_state)

    # Draw FCD type and corresponding parameters
    fcd_type = (
        draw_params(_presets["FCD_type"], random_state=rng)
        if fcd_type is None
        else fcd_type
    )
    params = draw_params(_presets[cast(str, fcd_type)], random_state=rng)

    # Replace any dotted paths with the values from the path;
    # e.g., the bbox settings between blurring and texture restoration.
    params = replace_paths_dict(params)

    # Post-hoc modifications
    # 1) build `wmh_seeds_hpf` from `wmh_seeds_corr`
    try:
        params["intensity_params"]["hyperintensity"]["wmh_seeds_hpf"] += params[
            "intensity_params"
        ]["hyperintensity"]["wmh_seeds_corr"]
    except KeyError:
        raise RuntimeError(
            "Internal contract violated: Could not find `wmh_seeds_hpf` and "
            "`wmh_seeds_corr` in `intensity_params.hyperintensity`. Check "
            "`synthfcd.get_simple_fcd_presets` for any errors."
        )

    # 2) generate a unique random seed for the growth and effect functions
    # except for texture restoration that is sequence-specific
    seed_params(
        rng,
        params,
        paths_to_seed=[
            "growth_params",
            "deformation_params.abnormal_gyration",
            "intensity_params.hyperintensity",
        ],
    )

    # 3) Extract and remove T1like settings for cross-modal translation
    if "T1like_settings" not in params["intensity_params"]["hyperintensity"]:
        raise RuntimeError(
            "Internal contract violated: Could not find `T1like_settings` in "
            "`intensity_params.hyperintensity`. Check `synthfcd.get_simple_fcd_presets` "
            "for any errors."
        )
    t1like_settings = params["intensity_params"]["hyperintensity"].pop(
        "T1like_settings"
    )

    # sequence-specific modifications (intensity-only)
    intensity_params_list = []
    for m in sequences:
        intensity_params = deepcopy(params["intensity_params"])
        # 4) Adjust the `scale_factor` and `sign` for T1-like hyper/hypo-intensity
        if m == "T1like":
            try:
                intensity_params["hyperintensity"]["scale_factor"] *= t1like_settings[
                    "scale_factor_shrink"
                ]
                intensity_params["hyperintensity"]["scale_factor"] *= t1like_settings[
                    "sign"
                ]
            except KeyError:
                raise RuntimeError(
                    "Internal contract violated: Could not find `scale_factor_shrink` "
                    "or `sign` in `T1like_settings` of `intensity_params.hyperintensity`. "
                    "Check `synthfcd.get_simple_fcd_presets` for any errors."
                )

        # 5) Set unique random seed for the texture restoration
        seed_params(
            rng,
            intensity_params,
            paths_to_seed=["texture_restoration"],
        )

        intensity_params_list.append(intensity_params)

    return {
        "FCD_type": fcd_type,
        "growth_params": params["growth_params"],
        "deformation_params": params["deformation_params"],
        "intensity_params": intensity_params_list,
    }
