# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import gzip
import json
from typing import TYPE_CHECKING, Any, List, Optional, Union

import attr
import numpy as np
from gym import spaces

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

try:
    from habitat.datasets.object_nav.personalized_object_nav_dataset import (
        PersonalizedObjectNavDatasetV1,
    )
    from habitat.datasets.object_nav.object_nav_dataset import (
        ObjectNavDatasetV1,
    )
except ImportError:
    pass

if TYPE_CHECKING:
    from omegaconf import DictConfig


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[str] = None
    object_id: Optional[Union[List, str]] = None
    floor_id: Optional[str] = None
    description: Optional[Union[List, str]] = None
    owner: Optional[str] = None
    summary: Optional[str] = None
    extracted_summary: Optional[List[str]] = None
    query: Optional[List[str]] = None

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float] = None


@attr.s(auto_attribs=True, kw_only=True)
class ObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_name: Optional[str] = None
    object_name_id: Optional[int] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None
    
    # Adding Personalized Content
    floor_id: Optional[str] = None
    description: Optional[Union[List, str]] = None
    owner: Optional[str] = None
    summary: Optional[str] = None
    extracted_summary: Optional[List[str]] = None
    query: Optional[List[str]] = None

@registry.register_sensor
class ObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            goal_spec that specifies which id use for goal specification,
            goal_spec_max_val the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "ObjectNavDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = self.config.goal_spec_max_val - 1
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: ObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:
        if len(episode.goals) == 0:
            logger.error(
                f"No goal specified for episode {episode.episode_id}."
            )
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(
                f"First goal should be ObjectGoal, episode {episode.episode_id}."
            )
            return None
        category_name = episode.object_category
        if self.config.goal_spec == "TASK_CATEGORY_ID":
            # return np.array(
            #     [self._dataset.category_to_task_category_id[category_name]],
            #     dtype=np.int64,
            # )

            if type(episode.object_id) is list:
                # obj_ids = ",".join([id.split("_")[-1] for id in episode.object_id])   #TODO: FIX
                return None
            else:
                obj_ids = episode.object_id.split("_")[-1]
            return np.array(
                [self._dataset.category_to_task_category_id[category_name], obj_ids],
                dtype=np.int64,
            )

            #return episode.description[0]               #TODO CHANGED: Added. 
        elif self.config.goal_spec == "OBJECT_ID":
            obj_goal = episode.goals[0]
            assert isinstance(obj_goal, ObjectGoal)  # for type checking
            return np.array([obj_goal.object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong goal_spec specified for ObjectGoalSensor."
            )


@registry.register_task(name="Personalized_ObjectNav-v1")
class ObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """

def get_task_description(id_list, scene_name, 
                         info_dir = "habitat-lab/data/datasets/eai_pers/active/val/easy/content/"):
    '''
    Returns the task description given the task_id and the object_id_number.
    The task_id helps to get the object category, and the object_id_number 
    tells us the specific instance of the category in the scene. This (instance) is 
    important when dealing with personalized objects.
    '''

    #Obtain scene info
    info_path = os.path.join(info_dir, f"{scene_name}.json.gz")
    with gzip.open(info_path, "r") as f:
        info = json.load(f)

    #Extract the task and obj_id number
    task_id, obj_id_num = id_list[0], id_list[1]
    print(f"Task ID: {task_id}, Obj_id_num: {obj_id_num}")

    #Obtain the object category name
    cat_to_task_id = info["category_to_task_category_id"]
    cat_name = [k for k in cat_to_task_id if cat_to_task_id[k] == task_id]
    assert len(cat_name) == 1, "Task Category mapping is not unique!"
    cat_name = cat_name[0]

    #Scan the episode info and get the corresponding description for the specific object instance
    for ep in info["episodes"]:

        if ep["object_category"] == cat_name:
            ep_obj_ids = ep["object_id"]

            if type(ep_obj_ids) is list:    #In case of multiple instances, find the correct instance using instance ID

                ep_obj_id_nums = [int(ep_obj_id.split("_")[-1]) for ep_obj_id in ep_obj_ids]
                if obj_id_num in ep_obj_id_nums:

                    obj_arg = ep_obj_id_nums.index(obj_id_num)
                    return ep["description"][obj_arg][0]

            else:

                ep_obj_id_num = int(ep_obj_ids.split("_")[-1])
                if obj_id_num == ep_obj_id_num:
                    return ep["description"][0]

    return None