from torchrl.envs import EnvBase
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    DiscreteTensorSpec,
    BinaryDiscreteTensorSpec,
)
import torch
from typing import Optional
import openslide
import os
from typing import Union
import numpy as np
from tensordict.tensordict import TensorDict
from ENV import Observation
from ENV.utils import eliminate_wrong_extension, COOD_Required,ParameterValueNotFound
from abc import ABC, abstractmethod
from ENV import LOGGER
from enum import Enum


class RESET(str, Enum):
    DEFAULT = "max"
    RANDOMIZE = "random"


class CoordinateOutOfRange(Exception):
    pass


HIGH = 1000000


# action should be defined in WSI_Env
class Base(EnvBase, ABC):
    """
    this clase is responsible for implementing the base of WSI environments, it requires the implementation of reward calculation,
    actions specification and auxiliary action dynamic implementation, the class itself is responsible for implementing all
    communicatory methods with env base class and dynamics of primary actions such as moving in the
    x or y direction and zooming in and out
    """

    batch_locked = False  # type: ignore

    # supported whole slides types
    Supported_types = [".svs", ".tif", ".dcm", ".scn", ".tiff", ".ndpi"]

    def __init__(
        self,
        path: Union[str, list],
        seed: Optional[int] = None,
        reset: str = RESET.DEFAULT,
        image_Size: int = 224,
        device: str = "cpu",
        zoom_dynamics=Observation.ZoomLocation.CENTER
    ):
        """
        initialize all the parameters and specs for the base environment

        Parameters
        ----------
        path : Union[str,list]
            a path to a WSI, a folder containing WSIs or a list of WSIs path
        seed : Optional[int], optional
            random seed , by default None
        reset : str, optional
            different possibilities to reset the env, by default RESET.DEFAULT
        image_Size : int, optional
            observed image size, by default 224
        device : str, optional
            device, by default 'cpu'

        Raises
        ------
        Exception
            wrong format for path
        """

        super().__init__(device=device, batch_size=torch.Size([]))

        self.image_list = self._conrol_path(path, Base.Supported_types)

        if seed == None:
            seed = np.random.randint(0, 2**31 - 1)
        self.set_seed(seed)

        self.observation_class = Observation.observation
        self.zoom_dynamics=zoom_dynamics
        if reset in RESET.__members__.values():
            self.reset_level = reset   
        else:
            LOGGER.error(
                f"parameter value {reset} in reward initialization isnt defined"
            )
            raise ParameterValueNotFound(
                f"parameter value reset not found, possible values are { RESET.__members__.values()}, instead got {reset}"
            )
        self.grace = -1
        self.counter = False
        self.size = image_Size
        self.coord_required = COOD_Required.LEVEL0

        LOGGER.info(
            f"Base ENV: reset_level {self.reset_level}, cood {self.coord_required}, size {self.size}"
        )

        actions = self.set_action_space().to(self.device)

        goal = self.set_goal_space()
        if goal != None:
            goal = goal.to(self.device)
        reward = self.set_reward_space().to(self.device)
        state = self.set_observation_space().to(self.device)
        self._make_spec(state, actions, goal, reward)

        LOGGER.info(
            f"Base ENV: actions {actions} \ngoal {goal} \nreward {reward} \nstate  {state}"
        )

    def _conrol_path(self, path: Union[str, list], supported: list):

        if isinstance(path, list):
            image_list = path
            image_list = eliminate_wrong_extension(image_list, supported)
        elif os.path.isdir(path):
            image_list = list(os.listdir(path))
            image_list = eliminate_wrong_extension(image_list, supported)
        elif os.path.splitext(path)[1] in supported:
            image_list = [path]
        else:
            LOGGER.error(f"file extension not supported {path}")
            raise Exception("file extension is not supported")

        return image_list

    @abstractmethod
    def _define_actions(self):
        """
        this function should define the action space for the ENV
        and weather they are bounder or not along their boundaries:
        hence it should return a tensor dict as follow:
        {
            'X': {'type': 'bounded' or 'discrete' or 'continuous'} if 'bounded' it should have a low and high keys,
                        if discrete t should have a list of possible values
            'Y':{'type': 'bounded' or 'discrete' or 'continuous'} if 'bounded' it should have a low and high keys,
                        if discrete t should have a list of possible values
            'ZOOM':{'type': 'discrete'} if discrete t should have a list of possible values

            'SELECT': {'type': 'discrete' or 'binary'} if discrete t should have a list of possible values

            'terminate_search': {'type', 'binary' or 'counter'}
            others: {'type': any of the above following the same rule except for counter and should functionality should
            be implemented in form_decision as moving and zooming, selecting and terminating will be handeled in base step}
            'anything' will be extended in the
        }
        """
        return {}

    @abstractmethod
    def _reset_counter(self):
        """
        set the counter initial value for the desired initial count using self.grace= desire value
        """
        pass

    @abstractmethod
    def _form_decision(self, tensordict: TensorDict, future_observation: torch.Tensor):
        """
        take the result of movement, deploy further actions and calculate reward.  Produce the final step output of the
        step functionality (tensordict should comply with spec)
        SHOULD out put a reward whihc is either a tensor or a tensordict following its definition
        """
        return {}

    @abstractmethod
    def _define_goal(self) -> dict:
        """
        return

        set a goal spec
        {
        'goal':{'type': 'cood' or 'image' or 'vector'} if it is a vector a 'size' key is needed, cood will be 2 and image will
        follow the size of image
        can be a list of all the above
        }
        if not goal conditioned RL specify None for type
        """
        return {"goal": {"type": None, "size": None}}

    @abstractmethod
    def _create_goal(self) -> dict:
        """
        a function that creates a goal,
        it should return a dictionary as follow {
        'goal': the goal to be reached -> tensor or tensordict depending on the goal structure
        'tensordict': if there is a need to bound the observation to a subset containing the goal otherwise the model will use the whole
        virtual slide and  search it all

        }
        if 'goal' is none, it is skipped and considered non-goal conditioned RL
        """
        return {"goal": None}

    @abstractmethod
    def _define_reward(self) -> dict:
        """
        return

        set a return spec
        {
        used to define intrinsic and extrinsic reward
        'extrinsic / intrinsic':{'type':'bounded' or 'continuous'} if it is bounded a high and a low bounds
        }
        """
        return {}

    @abstractmethod
    def _define_observation(self) -> dict:
        """
        change the observation class if needed
        rewrite coordinate required if needed

        retrun
        observations { 'cood', 'others'
        }
        """
        return {}

    @abstractmethod
    def _terminate(
        self,
        tensordict: TensorDict,
        future_observation: torch.Tensor,
        reward: Optional[Union[float, int, torch.Tensor]] = None,
    ) -> bool:
        """this function should be implmenmted in children classes assesing the termination condition and it should return true when the episode should be terminated

        Parameters
        ----------
        tensordict : TensorDict
            previous state information
        future_observation : torch.Tensor
            what is my next observation
        reward : Optional[Union[float,int,torch.Tensor]], optional
            the reward that has been just achieved, by default None
        """
        pass

    def _set_seed(self, seed: Optional[int]):
        gen = torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = gen

    def _make_spec(self, state, actions, goal, reward):
        self.observation_spec = state
        if goal != None:
            self.observation_spec["goal"] = goal
        self.state_spec = self.observation_spec.clone()

        self.action_spec = actions

        self.reward_spec = reward

    def set_action_space(self):
        dictionary = self._define_actions()
        assert "X" in dictionary, "X should be present in _define_actions"
        assert "Y" in dictionary, "Y should be present in _define_actions"
        assert "ZOOM" in dictionary, "ZOOM should be present in _define_actions"

        tensordict = TensorDict({}, [])
        specs = []
        for i in dictionary.keys():
            assert "type" in dictionary[i], f"type not specified for {i}"
            spec, extra = Base.__convert_to_Spec(dictionary[i])
            if isinstance(extra, tuple):
                tensordict[i] = TensorDict({"min": extra[0], "max": extra[1]}, [])
            elif isinstance(extra, list):
                tensordict[i] = extra
            if spec != None:
                specs.append((i, spec))
            else:
                self.counter = True
        self.action_limits = tensordict.to(self.device)

        return CompositeSpec({("action", element[0]): element[1] for element in specs})

    def set_goal_space(self):
        dictionary = self._define_goal()
        assert "goal" in dictionary, "goal should be present in _define_goal"
        types = dictionary["goal"]
        assert "type" in types, "type hould be present in goal"

        if isinstance(types["type"], list):
            specs = []
            for j in types["type"]:

                if j.lower() == "image":
                    spec, _ = Base.__convert_to_Spec(
                        {
                            "type": "bounded",
                            "low": 0,
                            "high": 1,
                            "size": torch.Size((3, self.size, self.size)),
                        }
                    )
                    specs.append((j.lower(), spec))
                elif j.lower() == "cood":
                    spec, _ = Base.__convert_to_Spec(
                        {
                            "type": "bounded",
                            "low": 0,
                            "high": HIGH,
                            "size": torch.Size((types["cood"]["size"],)),
                        }
                    )
                    specs.append((j.lower(), spec))
                elif j.lower() == "vector":
                    spec, _ = Base.__convert_to_Spec(
                        {
                            "type": "bounded",
                            "low": types["low"] if "low" in types else 0,
                            "high": types["high"] if "high" in types else 1,
                            "size": torch.Size(types["vector"]["size"]),
                        }
                    )
                    specs.append((j.lower(), spec))

            return CompositeSpec({element[0]: element[1] for element in specs})

        elif isinstance(types["type"], str):
            if types["type"].lower() == "image":
                spec, _ = Base.__convert_to_Spec(
                    {
                        "type": "bounded",
                        "low": 0,
                        "high": 1,
                        "size": torch.Size((3, self.size, self.size)),
                    }
                )
                return spec
            elif types["type"].lower() == "cood":
                spec, _ = Base.__convert_to_Spec(
                    {
                        "type": "bounded",
                        "low": 0,
                        "high": HIGH,
                        "size": torch.Size((types["cood"]["size"],)),
                    }
                )
                return spec
            elif types["type"].lower() == "vector":
                spec, _ = Base.__convert_to_Spec(
                    {
                        "type": "bounded",
                        "low": types["low"] if "low" in types else 0,
                        "high": types["high"] if "high" in types else 1,
                        "size": torch.Size(types["vector"]["size"]),
                    }
                )
                return spec

            else:
                raise Exception(
                    "no recognizable type was defined for goals : recognizable types are image, cood, and vector"
                )
        return None

    def set_reward_space(self):
        dictionary = self._define_reward()
        specs = []
        for j in dictionary.keys():
            assert "type" in dictionary[j], f"type hould be present in {j}"
            spec, _ = Base.__convert_to_Spec(dictionary[j])
            specs.append((j, spec))
        if len(specs) == 1:
            return specs[0][1]
        return CompositeSpec({element[0]: element[1] for element in specs})

    def set_observation_space(self):
        dictionary = self._define_observation()
        specs = []
        specs.append(
            (
                "image",
                Base.__convert_to_Spec(
                    {
                        "type": "bounded",
                        "low": 0,
                        "high": 1,
                        "size": torch.Size((3, self.size, self.size)),
                    }
                )[0],
            )
        )
        for j in dictionary.keys():
            if j.lower() == "image":
                continue
            elif j.lower() == "cood":
                spec, _ = Base.__convert_to_Spec(
                    {
                        "type": "bounded",
                        "low": 0,
                        "high": HIGH,
                        "size": torch.Size((dictionary[j],)),
                    }
                )
                specs.append((j.lower(), spec))
                if dictionary[j] == 2:
                    self.coord_required = COOD_Required.LEVEL1
                elif dictionary[j] == 3:
                    self.coord_required = COOD_Required.LEVEL2
                else:
                    raise CoordinateOutOfRange("coordinate value should be 2 or 3")

            else:
                spec, _ = Base.__convert_to_Spec(dictionary[j])
                specs.append((j.lower(), spec))

        return CompositeSpec({element[0]: element[1] for element in specs})

    @staticmethod
    def __convert_to_Spec(diction):
        """
        method for converting dictionary like structure to spec -> compositespec, discrete and continuous...
        """
        type = diction["type"].lower()
        extra = 0
        if type == "bounded":
            specs = BoundedTensorSpec(
                minimum=diction["low"],
                maximum=diction["high"],
                shape=diction["size"] if "size" in diction else 1,
                dtype=torch.float32,
            )
            extra = (diction["low"], diction["high"])
        elif type == "continuous":
            specs = UnboundedContinuousTensorSpec(shape=1, dtype=torch.float32)
        elif type == "discrete":
            extra = diction["values"]
            specs = DiscreteTensorSpec(n=len(extra), shape=torch.Size([1]))
        elif type == "binary":
            specs = BinaryDiscreteTensorSpec(n=1)
        else:
            specs = None
        return specs, extra

    def _reset_slide(self):
        """
        reset the whole slide with the intial state viewS
        Returns
        -------
        path
            the chosen slide path
        """
        path = np.random.choice(self.image_list)
        self.env = openslide.open_slide(path)
        if self.reset_level == "max" or self.reset_level == "random":
            self.coordinate = [0, 0]
            self.level = "max"
        else:
            self.level = np.random.randint(
                int(np.log2(int(self.env.level_downsamples[-1])))
            )
            self.coordinate = np.random.randint(
                self.env.dimensions[0]
            ), np.random.randint(self.env.dimensions[1])
        self.observation = self.observation_class(
            self.size, self.env, self.coordinate, self.level, self.coord_required,self.zoom_dynamics
        )
        return path

    def __initialize_start(self):
        """
        This function saves thumbnail and its coordinate in a class property
        """
        if self.coord_required:
            self.thumbnail_image, self.thumbnail_coordinates = (
                self.observation.get_current()
            )

        else:
            self.thumbnail_image = self.observation.get_current()

        if self.reset_level == RESET.RANDOMIZE:

            minlevel, maxlevel = self.observation._get_min_max_levels()
            level = np.random.randint(minlevel, maxlevel + 1)  # randomize initial level

            self.observation._set_level(level)

            start = self.observation._get_start()
            dim = self.observation._get_dim()

            x, y = np.random.randint(start[0], dim[0]), np.random.randint(
                start[1], dim[1]
            )  # randomize initial state coordinates

            self.observation._set_coordinate(torch.tensor([x, y]))

    def _reset(self, tensordict=None):
        """
        reset and populate the first observation
        """

        self._reset_slide()

        self._reset_counter()
        goal = self._create_goal()
        elements = []

        if goal["goal"] != None:

            if "tensordict" in goal:

                self.observation.bounded_env(
                    type=Observation.BoundingType.CUSTOM, tensor_dict=goal["tensordict"]
                )

            goal = goal["goal"]

            elements.append(("goal", goal))

        self.__initialize_start()  # initialize the starting position of the episode
        if self.coord_required:
            observation, cood = self.observation.get_current()
            elements.append(("image", observation))
            elements.append(("cood", cood))

        else:
            observation = self.observation.get_current()
            elements.append(("image", observation))

        # add everything to a tensordict
        return TensorDict({element[0]: element[1] for element in elements}, []).to(
            self.device
        )

    def _step(self, tensordict):
        """
        this is the step function, it is responsible of retrieving the next observation and reward, it uses a couple of functionality
        deployed in high level classes like _form_decision for reward retrieving and it behaves as env._step()
        """
        tensordict = tensordict.to(self.device)
        actions = tensordict["action"]

        x = actions["X"].squeeze(-1)

        if "X" in self.action_limits.keys():
            if isinstance(self.action_limits["X"], TensorDict):
                x = x.clamp(self.action_limits["X", "min"], self.action_limits["X", "max"])  # type: ignore
            else:
                x = self.action_limits["X"][x.item()]  # type: ignore

        y = actions["Y"].squeeze(-1)

        if "Y" in self.action_limits.keys():
            if isinstance(self.action_limits["Y"], TensorDict):
                y = y.clamp(self.action_limits["Y", "min"], self.action_limits["Y", "max"])  # type: ignore
            else:
                y = self.action_limits["Y"][y.item()]  # type: ignore

        zoom = self.action_limits["ZOOM"][actions["ZOOM"].item()]  # type: ignore

        elements = []
        if "goal" in tensordict.keys():
            elements.append(("goal", tensordict["goal"]))

        if self.coord_required:
            future_observation, cood = self.observation.step([x, y, zoom])
            elements.append(("image", future_observation.to(self.device)))
            elements.append(("cood", cood.to(self.device)))
            reward = self._form_decision(
                tensordict, (future_observation, cood.to(self.device))
            )
            terminate = self._terminate(
                tensordict, (future_observation, cood.to(self.device)), reward
            )
        else:
            future_observation = self.observation.step([x, y, zoom]).to(self.device)
            elements.append(("image", future_observation))
            reward = self._form_decision(tensordict, future_observation)
            terminate = self._terminate(tensordict, future_observation, reward)

        elements.append(("reward", reward))
        elements.append(("terminated", terminate))

        if self.counter:
            if self.grace > 0:
                self.grace -= 1
                elements.append(("done", torch.zeros(1, dtype=torch.bool)))
            else:
                elements.append(("done", torch.ones(1, dtype=torch.bool)))

        tensordict_returned = TensorDict(
            {element[0]: element[1] for element in elements}, []
        )

        return tensordict_returned.to(self.device)
