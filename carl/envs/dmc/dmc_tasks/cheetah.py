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

"""Cheetah Domain."""

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

# Running speed above which reward is 1.
_RUN_SPEED = 10


SUITE = containers.TaggedTasks()


def get_model_and_assets() -> Tuple[bytes, Dict]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    return common.read_model("cheetah.xml"), common.ASSETS


@SUITE.add("benchmarking")  # type: ignore
def run_context(
    context: Context = {},
    time_limit: int = _DEFAULT_TIME_LIMIT,
    random: Union[np.random.RandomState, int, None] = None,
    environment_kwargs: Optional[Dict] = None,
) -> dm_env:
    """Returns the run task."""
    xml_string, assets = get_model_and_assets()
    if context != {}:
        xml_string = adapt_context(xml_string=xml_string, context=context)
    physics = Physics.from_xml_string(xml_string, assets)
    task = Cheetah(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs
    )


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Cheetah domain."""

    def speed(self) -> np.float64:
        """Returns the horizontal speed of the Cheetah."""
        return self.named.data.sensordata["torso_subtreelinvel"][0]


class Cheetah(base.Task):
    """A `Task` to train a running Cheetah."""

    def initialize_episode(self, physics: Physics) -> None:
        """Sets the state of the environment at the start of each episode."""
        # The indexing below assumes that all joints have a single DOF.
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = self.random.uniform(lower, upper)

        # Stabilize the model before the actual simulation.
        physics.step(nstep=200)

        physics.data.time = 0
        super().initialize_episode(physics)

    def get_observation(self, physics: Physics) -> collections.OrderedDict:
        """Returns an observation of the state, ignoring horizontal position."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance.
        obs["position"] = physics.data.qpos[1:].copy()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics: Physics) -> np.float64:
        """Returns a reward to the agent."""
        return rewards.tolerance(
            physics.speed(),
            bounds=(_RUN_SPEED, float("inf")),
            margin=_RUN_SPEED,
            value_at_margin=0,
            sigmoid="linear",
        )
