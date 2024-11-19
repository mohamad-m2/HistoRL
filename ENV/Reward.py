from torch import Tensor
import torch
from torch import nn
from typing import Callable, Optional
from enum import Enum
from ENV import LOGGER
from ENV.utils import ParameterValueNotFound
EPSILON=0.1
AOI_THRESHOLD = 0.5  # this threshold is used in AOI type reward calcualtion
ACTION_COST = 0.02  # used in AOI type reward reward calculation
GENERIC_THRESHOLD = 1.0  # this threshold is used to decide a discrete reward (ENCODER type)

COOD_THRESHOLD = 100


class RewardType(str, Enum):
    COORDINATE = "cood"


class VaeRewardValues(float, Enum):
    POS = 0  # 1.0
    NEG = 0  # -1.0
    CLOSE = 1  # 0.5
    FAR = 0  # -0.5
    USELESS = 0  # -0.1


def distance():
    loss = nn.MSELoss(reduction="sum")

    def calculate_distance(x, y):
        return torch.sqrt(loss(x, y))

    return calculate_distance


class reward:

    def __init__(
        self,
        custom: Optional[Callable] = None,
        type: str = RewardType.COORDINATE,
        device: str = "cpu",
    ):
        """
        this class is responsible for reward calculation given the set of parameters needed following the calcualtion chosen

        Parameters
        ----------
        custom : Optional[Callable], optional
            custom function to calculate reward, by default None
        type : str, optional
            the type of the in-built reward calculation functions to be used , by default RewardType.ENCODER
        model : Optional[Callable], optional
            the model used in calculation of the reward in case needed, by default None
        device : str, optional
            the device used , by default 'cpu'

        Raises
        ------
        ParameterValueNotFound
            if the value of the type isn't found
        """
        self.device = device
        self.type = type
        if custom != None:
            self.reward_func = custom
        else:
            self.threshold = GENERIC_THRESHOLD
            if type.lower() == RewardType.COORDINATE:
                self.reward_func = self.__cood_reward
                self.loss = distance()
                self.base_threshold = COOD_THRESHOLD
            else:
                LOGGER.error(
                    f"parameter value {type} in reward initialization isnt defined"
                )
                raise ParameterValueNotFound(
                    "parameter value type not found, possible values are {RewardType.__m=embers__.values()}, instead got {type}"
                )

        LOGGER.info(f"reward class, type of reward claculation: {type}")

    def __call__(self, *args, **kargs):
        return self.reward_func(*args, **kargs)

    def __cood_reward(
        self, *, goal: Tensor, state: Tensor, next_state: Tensor, select: int
    ) -> Tensor:
        assert (
            isinstance(state, tuple) == False
        ), "reward is defined to take coordinate only"
        assert (
            isinstance(next_state, tuple) == False
        ), "reward is defined to take coordinate only"

       
        if next_state[-1] == goal[-1] or len(state) < 3:
            self.threshold = self.base_threshold * (2 ** goal[-1])
            distance=self.loss(next_state,goal)
            if(distance<self.threshold):

                reward =torch.clamp(torch.tensor([25* (2 ** goal[-1])/(distance+EPSILON)]), min=-1,max=1)
            else:
                reward = torch.tensor([0])
        else:
            reward = torch.tensor([0]) 

        return reward
