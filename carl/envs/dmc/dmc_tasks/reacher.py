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

"""Reacher Domain."""

from typing import Dict, Optional, Tuple, Union

import collections

import dm_env  # type: ignore
import numpy as np
from dm_control import mujoco  # type: ignore
from dm_control.rl import control  # type: ignore
from dm_control.suite import base, common  # type: ignore
from dm_control.suite.utils import randomizers  # type: ignore
from dm_control.utils import containers, rewards  # type: ignore

from carl.envs.dmc.dmc_tasks.utils import adapt_context  # type: ignore
from carl.utils.types import Context

_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = 0.05
_SMALL_TARGET = 0.015


SUITE = containers.TaggedTasks()


def get_model_and_assets() -> Tuple[bytes, Dict]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("reacher.xml"), common.ASSETS


@SUITE.add("benchmarking", "easy")  # type: ignore
def easy_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns reacher with sparse reward with 5e-2 tol and randomized target."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Reacher(target_size=_BIG_TARGET, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


@SUITE.add("benchmarking")  # type: ignore
def hard_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns reacher with sparse reward with 1e-2 tol and randomized target."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Reacher(target_size=_SMALL_TARGET, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Reacher domain."""

    def finger_to_target(self) -> np.ndarray:
        """Returns the vector from target to finger in global coordinates."""
        return (
            self.named.data.geom_xpos["target", :2]
            - self.named.data.geom_xpos["finger", :2]
        )

    def finger_to_target_dist(self) -> np.float64:
        """Returns the signed distance between the finger and target surface."""
        return np.linalg.norm(self.finger_to_target())


class Reacher(base.Task):
    """A reacher `Task` to reach the target."""

    def __init__(
        self, target_size: float, random: Union[np.random.RandomState, int, None] = None
    ) -> None:
        """Initialize an instance of `Reacher`.

        Args:
          target_size: A `float`, tolerance to determine whether finger reached the
              target.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._target_size = target_size
        super().__init__(random=random)

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode."""
        physics.named.model.geom_size["target", 0] = self._target_size
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)

        # Randomize target position
        angle = self.random.uniform(0, 2 * np.pi)
        radius = self.random.uniform(0.05, 0.20)
        physics.named.model.geom_pos["target", "x"] = radius * np.sin(angle)
        physics.named.model.geom_pos["target", "y"] = radius * np.cos(angle)

        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation of the state and the target position."""
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["to_target"] = physics.finger_to_target()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics: Physics) -> np.float64:
        radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
        return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))
