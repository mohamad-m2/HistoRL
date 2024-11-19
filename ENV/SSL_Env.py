import numpy as np
from tensordict import TensorDict
from ENV.BASE_Env import Base, RESET
from ENV.utils import (
    generate_goal,
    generate_goal_environment_fixed,
    save_dict_versatile,
    load_dict_versatile,
    ParameterValueNotFound
)
from typing import Optional, Union
from ENV import Reward
from ENV.Observation import ZoomLocation
import torch
from ENV import LOGGER
from enum import Enum
import json
import warnings
import inspect
import os
import copy

FILENAME = "ENV_META_DATA.json"
FOLDER = "SSL_META-DATA_FOLDER"


class FixedENV(int, Enum):
    FREE = 0
    HALF = 1
    FIXED = 2


class GoalGenerationError(Exception):
    pass


class FolderDoesntExists(Exception):
    pass

#TODO add probs to different levels
class SSL_Env(Base):
    """
    this class is based on the Base class, it uses its functionality from defining specs, reset and step
    it induces a specific taks to solve creating the environement and goal for that environment,
    """

    GRACE = 100

    def __handel_kwargs(self, **kwargs):
        """
        Handle extra rguments which are centered around a fixed environment or a variable one,
        saving the environement info makes sense only when it is fixed at least partially.

            -FIXED: both goal and environment space are fixed
            -HALF: only environment space is fixed
            -FREE: nothing is fixed

        """
        if "fix" in kwargs:
            self.meta_dictionary = None
            
            if(kwargs["fix"] in FixedENV.__members__.values()):
                self.fixed_env = kwargs["fix"]
            else:
                LOGGER.error(
                    f"parameter value {kwargs['fix']} in reward initialization isnt defined"
                )
                raise ParameterValueNotFound(
                    f"parameter value reset not found, possible values are { FixedENV.__members__.values()}, instead got {kwargs['fix']}"
                )

            if self.fixed_env != FixedENV.FREE:

                self.save = (
                    kwargs["save"] if "save" in kwargs else None
                )  # handel goal saving directory
                if self.save != None:

                    if not os.path.exists(self.save):
                        os.mkdir(self.save)
                    self.save = os.path.join(self.save, FOLDER)
                    if not os.path.exists(self.save):
                        os.mkdir(self.save)

                self.load = (
                    kwargs["load"] if "load" in kwargs else None
                )  # handel goal loading directory
                if self.load != None:
                    if not os.path.exists(self.load):
                        LOGGER.error(
                            f"folder given for loading doesnt exists {self.load}"
                        )
                        raise FolderDoesntExists(
                            f"folder given for loading doesnt exists {self.load}"
                        )
                    else:
                        self.load = os.path.join(self.load, FOLDER)
                        if not os.path.exists(self.load):
                            LOGGER.error(
                                f"folder given for loading doesnt exists {self.load}"
                            )
                            raise FolderDoesntExists(
                                f"folder given for loading doesnt exists {self.load}"
                            )

                # self.meta_dictionary contain information about the fixed environement and goal
                self.__load_meta_data(FILENAME)
                if self.meta_dictionary != None:
                    self.__save_meta_data(FILENAME)

        else:
            self.meta_dictionary = None
            self.fixed_env = FixedENV.FREE
        if 'probs' in kwargs:
            self.probs = kwargs['probs']
        else: 
            self.probs = None

    def __init__(
        self,
        path: str,
        *,
        reward: Reward.reward,
        high_level: Union[list,int] = 0,
        low_level: Union[list,int] = 2,
        seed: Optional[int] = None,
        reset: str = RESET.DEFAULT,
        image_Size: int = 224,
        device: str = "cpu",
        thumb_included:bool=False,
        zoom_dynamics=ZoomLocation.CENTER,
        **kwargs,
    ):
        """
        Parameters
        ----------
        path : str
            path to data
        reward : Reward.reward
            reward object from Reward module
        high_level : int, optional
            high magnification/resolution level for goals, by default 0
        low_level : int, optional
            low magnification/resolution level for initial state , by default 2
        seed : Optional[int], optional
            random seed, by default None
        reset : str, optional
            reset type enum from Base ENV, by default 'default'
        image_Size : int, optional
            the obsereved image size, by default 224
        device : str, optional
             by default 'cpu'
        """
        self.reward = reward
        self.high_level = [high_level] if isinstance(high_level,int) else high_level
        self.high_level.sort()
        self.low_level = [low_level] if isinstance(low_level,int) else low_level 
        self.low_level.sort()
        self.episode_length = 40

        self.__include_Thumb_In_Observation= thumb_included

        super().__init__(path, seed, reset, image_Size, device,zoom_dynamics)
        assert (
            min(self.low_level) > max(self.high_level)
        ), "low_level (magnification) shoud be higher than high_level (magnification)"

        LOGGER.info(
            f"SSL_ENV with high_level images from level {self.high_level} and low_level images from {self.low_level}"
        )
        self.__handel_kwargs(**kwargs)
        if(self.__include_Thumb_In_Observation):
            self._reset=self.thumbnail_reset
            self._step=self.thumbnail_step

    def __save_meta_data(self, File: str):

        if self.save == None:
            return
        data = copy.deepcopy(self.meta_dictionary)
        save_dict_versatile(data, self.save, "pth")
        with open(os.path.join(self.save, File), "w") as json_file:
            json.dump(data, json_file, indent=4)

    def __load_meta_data(self, File: str):
        if self.load == None:
            return None

        with open(os.path.join(self.load, File), "r") as json_file:
            data = load_dict_versatile(
                json.load(json_file), self.load, "pth", self.device
            )
            if self.fixed_env == FixedENV.HALF:
                self.meta_dictionary = {}
                self.meta_dictionary["low_meta_data"] = data["low_meta_data"]
                self.meta_dictionary["low_image"] = data["low_image"]
            else:
                self.meta_dictionary = data

    def __generate_goal(self, param: Optional[dict] = None) -> dict:
        """
        function that return the goal and initial state for the RL pretext task
        Parameters

        Parameters
        ----------
        param : Optional[dict], optional
            when provided the function will use it to only generate a goal for the environment provided, by default None

        Returns
        -------
        dict
            contain the initial state and goal for the RL task

        Raises
        ------
        GoalGenerationError
            when encoutering a problem with generating task (too much background might cause this error)

        """

        if param == None:
            dictionary = generate_goal(
                img=self.env,
                levels=self.observation.levels,
                size=self.size,
                relative_position=False,
                high_res_level=self.high_level,
                low_res_level=self.low_level,
                grace=SSL_Env.GRACE,probs_high=self.probs
            )
        else:
            dictionary = generate_goal_environment_fixed(
                img=self.env,
                levels=self.observation.levels,
                size=self.size,
                relative_position=False,
                high_res_level=self.high_level,
                environment_data=param,
                grace=SSL_Env.GRACE,probs_high=self.probs
            )

        if dictionary == {}:
            LOGGER.error("something wrong with goal generation")
            raise GoalGenerationError("something wrong with goal generation")

        else:

            dictionary["high_image"] = self.observation._post_process(
                dictionary["high_image"]
            )
            if param == None:
                dictionary["low_image"] = self.observation._post_process(
                    dictionary["low_image"]
                )

        return dictionary

    def get_goal(self) -> dict:
        """
        function responsible for returning the env and goal in different FixedENV cases
        """
        if self.fixed_env == FixedENV.FREE:
            return self.__generate_goal()
        elif self.fixed_env == FixedENV.FIXED:
            if (
                self.meta_dictionary == None
                or "high_image" not in self.meta_dictionary.keys()
            ):
                self.meta_dictionary = self.__generate_goal(self.meta_dictionary)
                self.__save_meta_data(FILENAME)
                return self.meta_dictionary
            else:
                return self.meta_dictionary
        else:
            if self.meta_dictionary == None:
                initialize = self.__generate_goal()

                self.meta_dictionary = {}
                self.meta_dictionary["low_meta_data"] = initialize["low_meta_data"]
                self.meta_dictionary["low_image"] = initialize["low_image"]
                self.__save_meta_data(FILENAME)
                return initialize
            else:

                return self.__generate_goal(self.meta_dictionary)

    def _terminate(
        self,
        tensordict: TensorDict,
        future_observation: Reward.Tensor,
        reward: Optional[Union[float, int, torch.Tensor]] = None,
    ) -> bool:
        coods = tensordict["cood"]
        next_obs_coods = future_observation[1]
        if reward > 0.5: #and (coods == next_obs_coods).all():
            terminate = torch.ones(1, dtype=torch.bool).to(self.device)
        else:
            terminate = torch.zeros(1, dtype=torch.bool).to(self.device)
        return terminate

    def _create_goal(self) -> dict:
        """this function is responsible for setting up the paramters and goal for the task, bounding the env

        Returns
        -------
        dict
            goal and tensordict for bounding the env
        """
        goal = self.get_goal()
        bound = TensorDict(
            {
                "zoom": TensorDict(
                    {"min": 0, "max": goal["low_meta_data"]["l_level"]}, []
                ),
                "end": goal["low_meta_data"]["dim"],
                "start": goal["low_meta_data"]["cood"],
            },
            [],
        )

        if self.reward.type == Reward.RewardType.COORDINATE:
            return {
                "goal": {
                    "image": goal["high_image"],
                    "cood": torch.cat(
                        (torch.tensor(goal["cood"]), torch.tensor([goal["high_meta_data"]["h_level"]]))
                    ),
                },
                "tensordict": bound,
            }
        else:
            return {"goal": goal["high_image"], "tensordict": bound}

    def _define_goal(self):
        if self.reward.type == Reward.RewardType.COORDINATE:
            return {"goal": {"type": ["image", "cood"], "cood": {"size": 3}}}
        else:
            return {"goal": {"type": "image"}}

    def _define_actions(self):

        X = {"type": "bounded", "low": -1, "high": 1}
        Y = {"type": "bounded", "low": -1, "high": 1}
        ZOOM = {"type": "discrete", "values": [-1, 0, 1]}
        SELECT = {"type": "binary"}
        terminate_search = {"type": "counter"}

        return {
            "X": X,
            "Y": Y,
            "ZOOM": ZOOM,
            "SELECT": SELECT,
            "terminate_search": terminate_search,
        }

    def set_episode_maximum_length(self, length):
        if length > 100:
            warnings.warn(
                f"episode length set too long, you may want to adjust it to lower values {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            )
            LOGGER.warning(
                f"episode length set too long, you may want to adjust it to lower values {self.__class__.__name__}.{inspect.currentframe().f_code.co_name}"
            )
        self.episode_length = length

    def _reset_counter(self):
        self.grace = self.episode_length

    def _define_reward(self):
        return {"extrinsic": {"type": "bounded", "high": 1.0, "low": -1.0}}

    def _form_decision(self, tensordict, future_observation):

        if self.reward.type == Reward.RewardType.COORDINATE:
            return self.reward(
                goal=tensordict["goal"]["cood"],
                state=tensordict["cood"],
                next_state=future_observation[1],
                select=tensordict["action"]["SELECT"],
            )
        else:
            return self.reward(
                goal=tensordict["goal"],
                state=tensordict["image"],
                next_state=future_observation,
                select=tensordict["action"]["SELECT"],
            )

    def _define_observation(self):
        observation={"image": [3, self.size, self.size]}
        if(self.__include_Thumb_In_Observation):
            observation['thumbnail']={'type':'bounded','low':0,'high':1,'size':torch.Size((3,self.size,self.size))}
        if self.reward.type == Reward.RewardType.COORDINATE:
            observation['cood']=3
        return observation

    def thumbnail_reset(self, tensordict=None):
        '''
        reset function when thumbnail should be included in the observation
        '''
        tensordict=super()._reset(tensordict)
        tensordict['thumbnail']=self.thumbnail_image.to(self.device)
        return tensordict
    def thumbnail_step(self, tensordict):
        
        '''
        step function when thumbnail should be included in the observation
        '''
        x=super()._step(tensordict)
        x['thumbnail']=tensordict['thumbnail']
        return x


class Simple_SSL_ENV(SSL_Env):
    """
    Have the same functionality with SSL ENV but take a procedure to simpllify and discretize action space by using moving_Action_Definition

    moving_Action_Definition: is a list for movement that  will be taken across x and y, example: [-1,-0.5,-0.25,0,0.25.0.5,1]

    """

    def __init__(
        self,
        path: str,
        *,
        reward: Reward.reward,
        moving_Action_Definition: list,
        high_level: int = 0,
        low_level: int = 2,
        seed: Optional[int] = None,
        reset: str = RESET.DEFAULT,
        image_Size: int = 224,
        device: str = "cpu",
        zoom_dynamics=ZoomLocation.CENTER,
        **kwargs,
    ):
        self.__move_list = moving_Action_Definition
        if 0 not in self.__move_list:
            self.__move_list.append(0)
            self.__move_list.sort()
            LOGGER.warning(
                f"0 wasn't included in the moving_Action_Definition {self.__class__.__name__}.{inspect.currentframe().f_code.co_name} it was forcebly added"
            )

            warnings.warn(
                f"0 wasn't included in the moving_Action_Definition {self.__class__.__name__}.{inspect.currentframe().f_code.co_name} it was forcebly added"
            )

        super().__init__(
            path,
            reward=reward,
            high_level=high_level,
            low_level=low_level,
            seed=seed,
            reset=reset,
            image_Size=image_Size,
            device=device,
            zoom_dynamics=zoom_dynamics,
            **kwargs,
        )

    def _define_actions(self):

        X = {
            "type": "discrete",
            "values": [torch.tensor(i, dtype=torch.float32) for i in self.__move_list],
        }
        Y = {
            "type": "discrete",
            "values": [torch.tensor(i, dtype=torch.float32) for i in self.__move_list],
        }
        ZOOM = {
            "type": "discrete",
            "values": [
                torch.tensor(-1, dtype=torch.int16),
                torch.tensor(0, dtype=torch.int16),
                torch.tensor(1, dtype=torch.int16),
            ],
        }
        terminate_search = {"type": "counter"}
        return {"X": X, "Y": Y, "ZOOM": ZOOM, "terminate_search": terminate_search}

    def _form_decision(
        self, tensordict, future_observation
    ):  # TODO check if i can get rid of this part

        if self.reward.type == Reward.RewardType.COORDINATE:
            return self.reward(
                goal=tensordict["goal"]["cood"],
                state=tensordict["cood"],
                next_state=future_observation[1],
                select=0,
            )
        else:
            return self.reward(
                goal=tensordict["goal"],
                state=tensordict["image"],
                next_state=future_observation,
                select=0,
            )
