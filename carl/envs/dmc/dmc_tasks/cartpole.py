# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cartpole Domain."""

from typing import Dict, Optional, Tuple, Union

import collections

import dm_env  # type: ignore
import numpy as np
from dm_control import mujoco  # type: ignore
from dm_control.rl import control  # type: ignore
from dm_control.suite import base, common  # type: ignore
from dm_control.utils import containers, rewards  # type: ignore

from carl.envs.dmc.dmc_tasks.utils import adapt_context  # type: ignore
from carl.utils.types import Context

_DEFAULT_TIME_LIMIT = 10


SUITE = containers.TaggedTasks()


def get_model_and_assets() -> Tuple[bytes, Dict]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("cartpole.xml"), common.ASSETS


@SUITE.add("benchmarking")  # type: ignore
def balance_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Cartpole Balance task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Balance(swing_up=False, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")  # type: ignore
def balance_sparse_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the sparse reward variant of the Cartpole Balance task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Balance(swing_up=False, sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")  # type: ignore
def swingup_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the Cartpole Swing-Up task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Balance(swing_up=True, sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")  # type: ignore
def swingup_sparse_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the sparse reward variant of the Cartpole Swing-Up task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Balance(swing_up=True, sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cartpole domain."""

    def cart_position(self) -> np.float64:
        """Returns the position of the cart."""
        return self.named.data.qpos["slider"][0]

    def angular_vel(self) -> np.ndarray:
        """Returns the angular velocity of the pole."""
        return self.data.qvel[1:]

    def pole_angle_cosine(self) -> np.ndarray:
        """Returns the cosine of the pole angle."""
        return self.named.data.xmat[2:, "zz"]

    def bounded_position(self) -> np.ndarray:
        """Returns the state, with pole angle split into sin/cos."""
        return np.hstack(
            (self.cart_position(), self.named.data.xmat[2:, ["zz", "xz"]].ravel())
        )


class Balance(base.Task):
    """A Cartpole `Task` to balance the pole.

    State is initialized either close to the target configuration or at a random
    configuration.
    """

    _CART_RANGE = (-0.25, 0.25)
    _ANGLE_COSINE_RANGE = (0.995, 1)

    def __init__(
        self,
        swing_up: bool,
        sparse: bool,
        random: Union[np.random.RandomState, int, None] = None,
    ) -> None:
        """Initializes an instance of `Balance`.

        Args:
          swing_up: A `bool`, which if `True` sets the cart to the middle of the
            slider and the pole pointing towards the ground. Otherwise, sets the
            cart to a random position on the slider and the pole to a random
            near-vertical position.
          sparse: A `bool`, whether to return a sparse or a smooth reward.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._sparse = sparse
        self._swing_up = swing_up
        super().__init__(random=random)

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode.

        Initializes the cart and pole according to `swing_up`, and in both cases
        adds a small random initial velocity to break symmetry.

        Args:
          physics: An instance of `Physics`.
        """
        nv = physics.model.nv
        if self._swing_up:
            physics.named.data.qpos["slider"] = 0.01 * self.random.randn()
            physics.named.data.qpos["hinge_1"] = np.pi + 0.01 * self.random.randn()
            physics.named.data.qpos[2:] = 0.1 * self.random.randn(nv - 2)
        else:
            physics.named.data.qpos["slider"] = self.random.uniform(-0.1, 0.1)
            physics.named.data.qpos[1:] = self.random.uniform(-0.034, 0.034, nv - 1)
        physics.named.data.qvel[:] = 0.01 * self.random.randn(physics.model.nv)
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation of the (bounded) physics state."""
        obs = collections.OrderedDict()
        obs["position"] = physics.bounded_position()
        obs["velocity"] = physics.velocity()
        return obs

    def _get_reward(self, physics: Physics, sparse: bool) -> np.float64:
        if sparse:
            cart_in_bounds = rewards.tolerance(
                physics.cart_position(), self._CART_RANGE
            )
            angle_in_bounds = rewards.tolerance(
                physics.pole_angle_cosine(), self._ANGLE_COSINE_RANGE
            ).prod()
            return cart_in_bounds * angle_in_bounds
        else:
            upright = (physics.pole_angle_cosine() + 1) / 2
            centered = rewards.tolerance(physics.cart_position(), margin=2)
            centered = (1 + centered) / 2
            small_control = rewards.tolerance(
                physics.control(), margin=1, value_at_margin=0, sigmoid="quadratic"
            )[0]
            small_control = (4 + small_control) / 5
            small_velocity = rewards.tolerance(physics.angular_vel(), margin=5).min()
            small_velocity = (1 + small_velocity) / 2
            return upright.mean() * small_control * small_velocity * centered

    def get_reward(self, physics: Physics) -> np.float64:
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        return self._get_reward(physics, sparse=self._sparse)
