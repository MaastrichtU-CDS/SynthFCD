"""
Base test classes for multiple modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
import pytest

from synthfcd.utils import dcopy_arrays


class RequiresImages(Protocol):
    def __call__(
        self,
        image: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray: ...


class RequiresMask(Protocol):
    def __call__(
        self,
        mask: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray: ...


class RequiresMasks(Protocol):
    def __call__(
        self,
        masks: dict[str, np.ndarray],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, np.ndarray] | tuple[np.ndarray, dict[str, Any]] | np.ndarray: ...


class RequiresSpacing(Protocol):
    def __call__(
        self,
        spacing: tuple[float | int, float | int, float | int],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray: ...


class TestImageValidation(ABC):
    """
    Base test class for checking robustness to invalid input images.
    """

    @abstractmethod
    def op_under_test(self) -> RequiresImages:
        """
        The operation under test.
        """

    @pytest.fixture
    def image(self) -> np.ndarray:
        random_state = np.random.default_rng(seed=42)
        return random_state.random((40, 40, 40))

    @pytest.mark.parametrize(
        "wrong_image",
        [
            np.random.random((40, 40, 40)).tolist(),
            None,
            "not_an_image",
            (1, 2, 3),
            5,
            {"image": np.random.random((40, 40, 40))},
        ],
    )
    def test_invalid_image_type(self, wrong_image: Any) -> None:
        """
        Test that a `TypeError` is raised if the image is not a numpy array.
        """
        with pytest.raises(TypeError):
            self.op_under_test()(image=wrong_image)

    @pytest.mark.parametrize(
        "wrong_image",
        [
            np.random.random((40, 40, 40)).astype(bool),
            np.random.random((40, 40)),
            np.random.random((1, 40, 40, 40)),
        ],
    )
    def test_invalid_image_value(self, wrong_image: np.ndarray) -> None:
        """
        Test that a `ValueError` is raised if the image is not a 3D floating point
        array.
        """
        with pytest.raises(ValueError):
            self.op_under_test()(image=wrong_image)


class TestMaskValidation(ABC):
    """
    Base test class for checking robustness to invalid input masks.
    """

    @abstractmethod
    def op_under_test(self) -> RequiresMask:
        """
        The operation under test.
        """

    @pytest.fixture
    def mask(self) -> np.ndarray:
        random_state = np.random.default_rng(seed=42)
        return random_state.random((40, 40, 40)) < 0.5

    @pytest.mark.parametrize(
        "wrong_mask",
        [
            np.zeros((40, 40, 40), dtype=bool).tolist(),
            None,
            "not_a_mask",
            (1, 2, 3),
            5,
            {"mask": np.zeros((40, 40, 40), dtype=bool)},
        ],
    )
    def test_invalid_mask_type(self, wrong_mask: Any) -> None:
        """
        Test that a `TypeError` is raised if the mask is not a boolean numpy array.
        """
        with pytest.raises(TypeError):
            self.op_under_test()(mask=wrong_mask)

    @pytest.mark.parametrize(
        "wrong_mask",
        [
            np.zeros((40, 40, 40)),
            np.zeros((40, 40, 40), dtype=int),
            np.zeros((40, 40, 40), dtype=float),
            np.zeros((40, 40), dtype=bool),
        ],
    )
    def test_invalid_mask_value(self, wrong_mask: np.ndarray) -> None:
        """
        Test that a `ValueError` is raised if the mask is not a boolean 3D numpy array.
        """
        with pytest.raises(ValueError):
            self.op_under_test()(mask=wrong_mask)

    def test_warning_empty_mask(self) -> None:
        """
        Test that a `UserWarning` is raised if the mask is all zeros.
        """
        with pytest.warns(UserWarning):
            self.op_under_test()(mask=np.zeros((40, 40, 40), dtype=bool))


class TestMasksValidation(ABC):
    """
    Base test class for checking robustness to invalid input masks.
    """

    @pytest.fixture
    def masks(self) -> dict[str, np.ndarray]:
        random_state = np.random.default_rng(seed=42)
        return {
            "gm_mask": random_state.random((40, 40, 40)) < 0.5,
            "wm_mask": random_state.random((40, 40, 40)) < 0.5,
            "boundary_mask": random_state.random((40, 40, 40)) < 0.5,
            "ventricles_mask": random_state.random((40, 40, 40)) < 0.5,
        }

    @abstractmethod
    def op_under_test(self) -> RequiresMasks:
        """
        The operation under test.
        """

    def test_non_dict(self) -> None:
        """
        Test that the function raises an error if the input is not a dictionary.
        """
        func = self.op_under_test()
        with pytest.raises(TypeError):
            func(
                masks=np.zeros((40, 40, 40)),  # type: ignore
            )

    def test_missing_keys(self, masks: dict[str, np.ndarray]) -> None:
        """
        Test that the function raises an error if the input `masks` dictionary is
        missing a required key.
        """
        func = self.op_under_test()
        for key in masks:
            curr_masks = dcopy_arrays(masks)
            del curr_masks[key]
            with pytest.raises(ValueError):
                func(masks=curr_masks)

    def test_non_numpy_array(self, masks: dict[str, np.ndarray]) -> None:
        """
        Test that the function raises an error if the input `masks` dictionary contains
        a non-numpy array.
        """
        func = self.op_under_test()
        for key in masks:
            curr_masks = dcopy_arrays(masks)
            for test_value in [
                curr_masks[key].tolist(),
                None,
                "not_an_array",
                (1, 2, 3),
                5,
                {"key": curr_masks[key]},
            ]:
                curr_masks[key] = test_value
                with pytest.raises(TypeError):
                    func(masks=curr_masks)

    def test_non_3d(self, masks: dict[str, np.ndarray]) -> None:
        """
        Test that the function raises an error if the input `masks` dictionary contains
        a non-3D array.
        """
        func = self.op_under_test()
        for key in masks:
            curr_masks = dcopy_arrays(masks)
            for test_value in [
                curr_masks[key].reshape((64, 1000)),
                curr_masks[key].reshape((1, 40, 40, 40)),
            ]:
                curr_masks[key] = test_value
                with pytest.raises(ValueError):
                    func(masks=curr_masks)

    def test_non_boolean(self, masks: dict[str, np.ndarray]) -> None:
        """
        Test that the function raises an error if the input `masks` dictionary contains
        a non-boolean array.
        """
        func = self.op_under_test()
        for key in masks:
            curr_masks = dcopy_arrays(masks)
            curr_masks[key] = curr_masks[key].astype(float)
            with pytest.raises(ValueError):
                func(masks=curr_masks)

    def test_different_shapes(self, masks: dict[str, np.ndarray]) -> None:
        """
        Test that the function raises an error if the input `masks` dictionary contains
        arrays with different shapes.
        """
        func = self.op_under_test()
        for key in masks:
            curr_masks = dcopy_arrays(masks)
            curr_masks[key] = curr_masks[key].reshape((1, 40, 40, 40))
            with pytest.raises(ValueError):
                func(masks=curr_masks)


class TestSpacingValidation(ABC):
    """
    Base test class for checking robustness to invalid input spacing.
    """

    @abstractmethod
    def op_under_test(self) -> RequiresSpacing:
        """
        The operation under test.
        """

    @pytest.mark.parametrize(
        "wrong_spacing",
        [
            "not_a_tuple",
            None,
            3,
            False,
            {"spacing": (1, 2, 3.56)},
            [1, 2, 3],
        ],
    )
    def test_invalid_spacing_type(self, wrong_spacing: Any) -> None:
        """
        Test that a `TypeError` is raised if the spacing is not a tuple.
        """
        with pytest.raises(TypeError):
            self.op_under_test()(spacing=wrong_spacing)

    @pytest.mark.parametrize(
        "wrong_spacing",
        [
            (True, 2, 3),
            (1, 2, "3"),
            (1, 2),
            (1, 2, 3, 4),
            (-1, 2, 3),
            (0, 2, 3),
        ],
    )
    def test_invalid_spacing_value(self, wrong_spacing: tuple[Any, ...]) -> None:
        """
        Test that a `ValueError` is raised if the spacing is not a tuple of 3 floats or
        integers.
        """
        with pytest.raises(ValueError):
            self.op_under_test()(spacing=wrong_spacing)
