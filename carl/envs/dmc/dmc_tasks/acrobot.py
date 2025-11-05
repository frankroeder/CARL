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

"""Acrobot Domain."""

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
    return common.read_model("acrobot.xml"), common.ASSETS


@SUITE.add("benchmarking")  # type: ignore
def swingup_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns Acrobot balance task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Balance(sparse=False, random=random)
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
    """Returns Acrobot sparse balance."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Balance(sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Acrobot domain."""

    def horizontal(self) -> np.ndarray:
        """Returns horizontal (x) component of body frame z-axes."""
        return self.named.data.xmat[["upper_arm", "lower_arm"], "xz"]

    def vertical(self) -> np.ndarray:
        """Returns vertical (z) component of body frame z-axes."""
        return self.named.data.xmat[["upper_arm", "lower_arm"], "zz"]

    def to_target(self) -> np.float64:
        """Returns the distance from the tip to the target."""
        tip_to_target = (
            self.named.data.site_xpos["target"] - self.named.data.site_xpos["tip"]
        )
        return np.linalg.norm(tip_to_target)

    def orientations(self) -> np.ndarray:
        """Returns the sines and cosines of the pole angles."""
        return np.concatenate((self.horizontal(), self.vertical()))


class Balance(base.Task):
    """An Acrobot `Task` to swing up and balance the pole."""

    def __init__(
        self, sparse: bool, random: Union[np.random.RandomState, int, None] = None
    ) -> None:
        """Initializes an instance of `Balance`.

        Args:
          sparse: A `bool` specifying whether to use a sparse (indicator) reward.
          random: Optional, either a `numpy.random.RandomState` instance, an
            integer seed for creating a new `RandomState`, or None to select a seed
            automatically (default).
        """
        self._sparse = sparse
        super().__init__(random=random)

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode.

        Shoulder and elbow are set to a random position between [-pi, pi).

        Args:
          physics: An instance of `Physics`.
        """
        physics.named.data.qpos[["shoulder", "elbow"]] = self.random.uniform(
            -np.pi, np.pi, 2
        )
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation of pole orientation and angular velocities."""
        obs = collections.OrderedDict()
        obs["orientations"] = physics.orientations()
        obs["velocity"] = physics.velocity()
        return obs

    def _get_reward(self, physics: Physics, sparse: bool) -> np.float64:
        target_radius = physics.named.model.site_size["target", 0]
        return rewards.tolerance(
            physics.to_target(), bounds=(0, target_radius), margin=0 if sparse else 1
        )

    def get_reward(self, physics: Physics) -> np.float64:
        """Returns a sparse or a smooth reward, as specified in the constructor."""
        return self._get_reward(physics, sparse=self._sparse)
