from __future__ import annotations

__all__ = [
    "LabelEnum",
    "SynthSegLabel",
]

from synthfcd.utils._base import _LabelEnum


class LabelEnum(_LabelEnum):
    """
    Base class for label enumerations.
    """


class SynthSegLabel(LabelEnum):
    """
    Enumeration mapping SynthSeg segmentation labels to anatomical structure names.

    This includes both FreeSurfer subcortical labels and Desikan-Killiany-Tourville (DKT)
    cortical parcellation labels as output by SynthSeg with the `--parc` flag.

    Subcortical labels (0-60) follow the FreeSurfer labeling convention.
    Cortical labels (1001-1035 for left, 2001-2035 for right) follow the DKT atlas.

    Examples:
        Lookup by index (int -> enum/name)::

            >>> SynthSegLabel(17)
            <SynthSegLabel.LEFT_HIPPOCAMPUS: 17>
            >>> SynthSegLabel(17).name
            'LEFT_HIPPOCAMPUS'

        Lookup by name (str -> enum/int)::

            >>> SynthSegLabel['LEFT_HIPPOCAMPUS']
            <SynthSegLabel.LEFT_HIPPOCAMPUS: 17>
            >>> SynthSegLabel['LEFT_HIPPOCAMPUS'].value
            17

        Lookup by readable name (str -> int)::

            >>> SynthSegLabel.from_readable_name("left hippocampus").value
            17

        Get readable name from index::

            >>> SynthSegLabel(17).readable_name
            'Left Hippocampus'

    References:
        - FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
        - SynthSeg: https://github.com/BBillot/SynthSeg
        - DKT atlas: Klein & Tourville (2012), Frontiers in Neuroscience
    """

    # ==========================================================================
    # FreeSurfer Subcortical Labels
    # ==========================================================================
    BACKGROUND = 0

    # Left hemisphere subcortical
    LEFT_CEREBRAL_WHITE_MATTER = 2
    LEFT_LATERAL_VENTRICLE = 4
    LEFT_INFERIOR_LATERAL_VENTRICLE = 5
    LEFT_CEREBELLUM_WHITE_MATTER = 7
    LEFT_CEREBELLUM_CORTEX = 8
    LEFT_THALAMUS = 10
    LEFT_CAUDATE = 11
    LEFT_PUTAMEN = 12
    LEFT_PALLIDUM = 13
    LEFT_HIPPOCAMPUS = 17
    LEFT_AMYGDALA = 18
    LEFT_ACCUMBENS_AREA = 26
    LEFT_VENTRAL_DC = 28

    # Midline structures
    THIRD_VENTRICLE = 14
    FOURTH_VENTRICLE = 15
    BRAIN_STEM = 16
    CSF = 24

    # Right hemisphere subcortical
    RIGHT_CEREBRAL_WHITE_MATTER = 41
    RIGHT_LATERAL_VENTRICLE = 43
    RIGHT_INFERIOR_LATERAL_VENTRICLE = 44
    RIGHT_CEREBELLUM_WHITE_MATTER = 46
    RIGHT_CEREBELLUM_CORTEX = 47
    RIGHT_THALAMUS = 49
    RIGHT_CAUDATE = 50
    RIGHT_PUTAMEN = 51
    RIGHT_PALLIDUM = 52
    RIGHT_HIPPOCAMPUS = 53
    RIGHT_AMYGDALA = 54
    RIGHT_ACCUMBENS_AREA = 58
    RIGHT_VENTRAL_DC = 60

    # ==========================================================================
    # DKT Cortical Parcellation Labels - Left Hemisphere (1001-1035)
    # ==========================================================================
    LEFT_UNKNOWN = 1001
    LEFT_CAUDAL_ANTERIOR_CINGULATE = 1002
    LEFT_CAUDAL_MIDDLE_FRONTAL = 1003
    LEFT_CUNEUS = 1005
    LEFT_ENTORHINAL = 1006
    LEFT_FUSIFORM = 1007
    LEFT_INFERIOR_PARIETAL = 1008
    LEFT_INFERIOR_TEMPORAL = 1009
    LEFT_ISTHMUS_CINGULATE = 1010
    LEFT_LATERAL_OCCIPITAL = 1011
    LEFT_LATERAL_ORBITOFRONTAL = 1012
    LEFT_LINGUAL = 1013
    LEFT_MEDIAL_ORBITOFRONTAL = 1014
    LEFT_MIDDLE_TEMPORAL = 1015
    LEFT_PARAHIPPOCAMPAL = 1016
    LEFT_PARACENTRAL = 1017
    LEFT_PARS_OPERCULARIS = 1018
    LEFT_PARS_ORBITALIS = 1019
    LEFT_PARS_TRIANGULARIS = 1020
    LEFT_PERICALCARINE = 1021
    LEFT_POSTCENTRAL = 1022
    LEFT_POSTERIOR_CINGULATE = 1023
    LEFT_PRECENTRAL = 1024
    LEFT_PRECUNEUS = 1025
    LEFT_ROSTRAL_ANTERIOR_CINGULATE = 1026
    LEFT_ROSTRAL_MIDDLE_FRONTAL = 1027
    LEFT_SUPERIOR_FRONTAL = 1028
    LEFT_SUPERIOR_PARIETAL = 1029
    LEFT_SUPERIOR_TEMPORAL = 1030
    LEFT_SUPRAMARGINAL = 1031
    LEFT_TRANSVERSE_TEMPORAL = 1034
    LEFT_INSULA = 1035

    # ==========================================================================
    # DKT Cortical Parcellation Labels - Right Hemisphere (2001-2035)
    # ==========================================================================
    RIGHT_UNKNOWN = 2001
    RIGHT_CAUDAL_ANTERIOR_CINGULATE = 2002
    RIGHT_CAUDAL_MIDDLE_FRONTAL = 2003
    RIGHT_CUNEUS = 2005
    RIGHT_ENTORHINAL = 2006
    RIGHT_FUSIFORM = 2007
    RIGHT_INFERIOR_PARIETAL = 2008
    RIGHT_INFERIOR_TEMPORAL = 2009
    RIGHT_ISTHMUS_CINGULATE = 2010
    RIGHT_LATERAL_OCCIPITAL = 2011
    RIGHT_LATERAL_ORBITOFRONTAL = 2012
    RIGHT_LINGUAL = 2013
    RIGHT_MEDIAL_ORBITOFRONTAL = 2014
    RIGHT_MIDDLE_TEMPORAL = 2015
    RIGHT_PARAHIPPOCAMPAL = 2016
    RIGHT_PARACENTRAL = 2017
    RIGHT_PARS_OPERCULARIS = 2018
    RIGHT_PARS_ORBITALIS = 2019
    RIGHT_PARS_TRIANGULARIS = 2020
    RIGHT_PERICALCARINE = 2021
    RIGHT_POSTCENTRAL = 2022
    RIGHT_POSTERIOR_CINGULATE = 2023
    RIGHT_PRECENTRAL = 2024
    RIGHT_PRECUNEUS = 2025
    RIGHT_ROSTRAL_ANTERIOR_CINGULATE = 2026
    RIGHT_ROSTRAL_MIDDLE_FRONTAL = 2027
    RIGHT_SUPERIOR_FRONTAL = 2028
    RIGHT_SUPERIOR_PARIETAL = 2029
    RIGHT_SUPERIOR_TEMPORAL = 2030
    RIGHT_SUPRAMARGINAL = 2031
    RIGHT_TRANSVERSE_TEMPORAL = 2034
    RIGHT_INSULA = 2035

    @classmethod
    def is_cortical(cls, label: int) -> bool:
        """
        Check if a label corresponds to a cortical region (DKT parcellation).
        """
        return 1001 <= label <= 1035 or 2001 <= label <= 2035

    @classmethod
    def is_left_hemisphere(cls, label: int) -> bool:
        """
        Check if a label belongs to the left hemisphere.
        """
        left_subcortical = {2, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28}
        return label in left_subcortical or 1001 <= label <= 1035

    @classmethod
    def is_right_hemisphere(cls, label: int) -> bool:
        """
        Check if a label belongs to the right hemisphere.
        """
        right_subcortical = {41, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60}
        return label in right_subcortical or 2001 <= label <= 2035

    @classmethod
    def get_all_labels(
        cls,
        hemisphere: str | None = None,
        include_neutral: bool = False,
    ) -> set[int]:
        """
        Return all label values, optionally filtered by hemisphere.

        Can optionally include neutral labels (e.g., CSF, brainstem, ventricles) when
        choosing a hemisphere via ``include_neutral``. Background is excluded; use
        :meth:`get_background_labels` instead.
        """
        all_labels: set[int] = {int(label) for label in cls}
        left: set[int] = {
            label for label in all_labels if cls.is_left_hemisphere(label)
        }
        right: set[int] = {
            label for label in all_labels if cls.is_right_hemisphere(label)
        }
        neutral: set[int] = {
            cls.THIRD_VENTRICLE,
            cls.FOURTH_VENTRICLE,
            cls.BRAIN_STEM,
            cls.CSF,
        }

        if hemisphere == "left":
            if include_neutral:
                return left | neutral
            return left
        elif hemisphere == "right":
            if include_neutral:
                return right | neutral
            return right

        return all_labels

    @classmethod
    def get_background_labels(cls) -> set[int]:
        """
        Return background label values.
        """
        return {cls.BACKGROUND}

    @classmethod
    def get_cortical_labels(cls, hemisphere: str | None = None) -> set[int]:
        """
        Return all cortical (DKT) label values, optionally filtered by hemisphere.
        """
        left: set[int] = {
            label
            for label in cls.get_all_labels(hemisphere="left")
            if cls.is_cortical(label)
        }
        right: set[int] = {
            label
            for label in cls.get_all_labels(hemisphere="right")
            if cls.is_cortical(label)
        }

        if hemisphere == "left":
            return left
        elif hemisphere == "right":
            return right
        return left | right

    @classmethod
    def get_subcortical_labels(cls, hemisphere: str | None = None) -> set[int]:
        """
        Return all subcortical label values, optionally filtered by hemisphere.
        """
        left: set[int] = {
            10,
            11,
            12,
            13,
            17,
            18,
            26,
            28,
        }
        right: set[int] = {
            49,
            50,
            51,
            52,
            53,
            54,
            58,
            60,
        }
        if hemisphere == "left":
            return left
        elif hemisphere == "right":
            return right
        else:
            return left | right

    @classmethod
    def get_white_matter_labels(cls, hemisphere: str | None = None) -> set[int]:
        """
        Return all white matter label values, optionally filtered by hemisphere.
        """
        left: set[int] = {
            cls.LEFT_CEREBRAL_WHITE_MATTER,
            cls.LEFT_CEREBELLUM_WHITE_MATTER,
        }
        right: set[int] = {
            cls.RIGHT_CEREBRAL_WHITE_MATTER,
            cls.RIGHT_CEREBELLUM_WHITE_MATTER,
        }
        if hemisphere == "left":
            return left
        elif hemisphere == "right":
            return right
        else:
            return left | right

    @classmethod
    def get_background_csf_labels(cls) -> set[int]:
        """
        Return the union of background and CSF label values.
        """
        return cls.get_background_labels() | {cls.CSF}

    @classmethod
    def get_ventricle_labels(
        cls,
        hemisphere: str | None = None,
        lateral_only: bool = False,
    ) -> set[int]:
        """
        Return all ventricle label values, optionally filtered by hemisphere.

        Can optionally exclude 3rd and 4th ventricle labels by setting `lateral_only` to
        True.
        """
        left: set[int] = {
            cls.LEFT_LATERAL_VENTRICLE,
            cls.LEFT_INFERIOR_LATERAL_VENTRICLE,
        }
        right: set[int] = {
            cls.RIGHT_LATERAL_VENTRICLE,
            cls.RIGHT_INFERIOR_LATERAL_VENTRICLE,
        }

        if not lateral_only:
            left.add(cls.THIRD_VENTRICLE)
            left.add(cls.FOURTH_VENTRICLE)
            right.add(cls.THIRD_VENTRICLE)
            right.add(cls.FOURTH_VENTRICLE)

        if hemisphere == "left":
            return left
        elif hemisphere == "right":
            return right
        else:
            return left | right

    @classmethod
    def get_frontal_lobe_labels(cls, hemisphere: str | None = None) -> set[int]:
        """
        Return all frontal lobe label values, optionally filtered by hemisphere.
        """
        left: set[int] = {
            1002,
            1003,
            1012,
            1014,
            1017,
            1018,
            1019,
            1020,
            1024,
            1026,
            1027,
            1028,
        }
        right: set[int] = {
            2002,
            2003,
            2012,
            2014,
            2017,
            2018,
            2019,
            2020,
            2024,
            2026,
            2027,
            2028,
        }
        if hemisphere == "left":
            return left
        elif hemisphere == "right":
            return right
        else:
            return left | right

    @classmethod
    def get_temporal_lobe_labels(cls, hemisphere: str | None = None) -> set[int]:
        """
        Return all temporal lobe label values, optionally filtered by hemisphere.
        """
        left: set[int] = {1006, 1007, 1009, 1015, 1016, 1030, 1034}
        right: set[int] = {2006, 2007, 2009, 2015, 2016, 2030, 2034}
        if hemisphere == "left":
            return left
        elif hemisphere == "right":
            return right
        else:
            return left | right
