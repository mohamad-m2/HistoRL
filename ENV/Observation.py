from __future__ import annotations
import torch
from torch import Tensor
import torchvision
import numpy as np
import openslide
from typing import Union, Optional, Tuple
from tensordict.tensordict import TensorDict
from abc import abstractmethod
from ENV import LOGGER
from ENV.utils import ParameterValueNotFound, COOD_Required,memory_mapped_openslide
from enum import Enum
from collections import OrderedDict
import os
class ZoomLocation(str, Enum):
    """
    this class includes the dyanmics to the zooming operation: zoom to the center or up-left corner
    """

    CENTER = "center"
    CORNER = "up-left"


class BoundingType(str, Enum):
    CURRENT = "current"
    CUSTOM = "custom"
    CURRENT_MIN_LEVEL = "custom_min_level"

class MEM_Exception(Exception):
    pass
class observation:

    def __init__(
        self,
        size: int,
        env: Union[openslide.OpenSlide, openslide.ImageSlide],
        coordinate: Union[tuple, list, Tensor],
        level: Union[int, str],
        coord_required: Union[int, COOD_Required] = COOD_Required.LEVEL0,
        ZoomDynamics: Union[str, ZoomLocation] = ZoomLocation.CENTER,
    ) -> None:
        """_summary_

        Parameters
        ----------
        size : int
            size of the observation
        env : Union[openslide.OpenSlide , openslide.ImageSlide]
            the WSI where the dynamics takes place
        coordinate : Union[tuple,list,Tensor]
            starting coordinates of the observation
        level : Union[int,str]
            starting level of the observation, int or 'max'
        coord_required : bool, optional
            return the current observation coordinate alongside its image or not, by default False
        """
        self.__size = size

        if(coord_required in COOD_Required.__members__.values()):
            self.return_coord = coord_required
        else:
            LOGGER.error(
                f"parameter value {coord_required} in reward initialization isnt defined"
            )
            raise ParameterValueNotFound(
                f"parameter value reset not found, possible values are { COOD_Required.__members__.values()}, instead got {coord_required}"
            )

        self.__start = torch.tensor(
            [0, 0]
        )  # the top left coordinate of the observable space
        self.__dim = torch.tensor(env.dimensions)  # dimensions of the observable space
        self.__coordinate = (
            torch.tensor(coordinate)
            if isinstance(coordinate, Tensor) == False
            else coordinate.clone().detach()
        )


        if(ZoomDynamics in ZoomLocation.__members__.values()):
            self.__ZoomDynamics = ZoomDynamics
        else:
            LOGGER.error(
                f"parameter value {ZoomDynamics} in reward initialization isnt defined"
            )
            raise ParameterValueNotFound(
                f"parameter value reset not found, possible values are { ZoomLocation.__members__.values()}, instead got {ZoomDynamics}"
            )

        self.env = env

        self.levels, self.upper_bound = (
            self.__create_levels()
        )  # upper_bound is the highest level in the levels dictionary
        self.__max_level = self.upper_bound

        self.__min_level = 0  # min and max levels are the space magnification bounds

        self._set_level(level)

        LOGGER.debug(
            f"---Observation class----- \n\t size:{self.__size} \n\t start:{self.__start} \n\t dim:{self.__dim}\
                    \n\t level:{self.__level} \n\t min_level:{self.__min_level} \n\t max_level: {self.__max_level} \n\t coord_required:{coord_required}"
        )

    def _set_coordinate(self, coordinate: Union[tuple, list, Tensor]):
        self.__coordinate = (
            torch.tensor(coordinate)
            if isinstance(coordinate, Tensor) == False
            else coordinate.clone().detach()
        )

    def _get_min_max_levels(self) -> Tuple[int, int]:
        return self.__min_level, self.__max_level

    def _get_coordinate(self) -> Tensor:
        return self.__coordinate

    def _get_size(self) -> int:
        return self.__size

    def _set_zoom_dynamics(self, ZoomDynamics: Union[ZoomLocation, str]):
        if isinstance(ZoomDynamics, str):
            assert (
                ZoomDynamics in COOD_Required.__members__.values()
            ), f"cood_required should belong to {ZoomLocation.__members__.values()}, it should take either the value of either a key or a value"
            self.__ZoomDynamics = ZoomDynamics
        else:
            self.__ZoomDynamics = ZoomDynamics

    def _set_level(self, level: Union[int, str]):
        if isinstance(level, str) and level.lower() == "max":
            self.__level = torch.tensor(self.__max_level)
        else:
            assert isinstance(
                level, int
            ), "level can only take the value max as a string otherwise it should be a digit"
            self.__level = torch.tensor(level)
        self.__level = torch.clamp(self.__level, self.__min_level, self.__max_level)

    def _get_level(self) -> int:
        return self.__level

    def _get_dim(self) -> Tensor:
        return self.__dim

    def _get_start(self) -> Tensor:
        return self.__start

    def _get_bottom_right_coordinate(self):

        size_current_view = self.levels[self.__level.item()][1]
        multiplier = int(
            self.env.level_downsamples[self.levels[self.__level.item()][0]]
        )

        temp = size_current_view * multiplier

        new_coordinate = self.__start + (torch.tensor([1, 1]) * temp)
        new_coordinate = torch.clamp(new_coordinate, self.__start, self.__dim)

    def __create_levels(self) -> tuple[dict, int]:
        """__create_levels map the levels of the images to a new levels set with x2 zoom factor between them
        by setting a level and size for read_region (openslide) function.

        Returns
        -------
        tuple[dict,int]
            return a dictionary of the new levels and int correpsonding to the highest level of them all
        """

        levels = {}
        counter = 0
        previous_downsample_factor = 1

        for level, downsample_factor in enumerate(self.env.level_downsamples):
            _ = downsample_factor
            size_increase_factor = (
                2  # the magnification/ downsampling between our level is set to x2
            )

            while (
                _ > previous_downsample_factor * 2
            ):  # the while loop set the intermediate levels between
                # previous_downsample_factor and downsample_factor
                _ = _ // 2

                levels[counter] = (level - 1, self.__size * size_increase_factor)
                counter += 1
                size_increase_factor *= 2

            # set the current downsample_factor in its correct place in the dictionary
            levels[counter] = (level, self.__size)
            counter += 1

            previous_downsample_factor = downsample_factor

        size = self.__size * 2

        # fill the rest by using the last available level and an increased size
        while (
            size < self.env.level_dimensions[-1][0]
            and size < self.env.level_dimensions[-1][1]
        ):
            levels[counter] = (self.env.level_count - 1, size)
            counter += 1
            size *= 2

        return levels, counter

    def __custom_bounded_env(self, tensor_dict: TensorDict) -> None:
        """custom bounding for the enviroment

        Parameters
        ----------
        tensor_dict : TensorDict
            should contain both levels and dimensions of the bounded enviroment
        """

        self.__min_level = tensor_dict["zoom", "min"].item()
        self.__max_level = tensor_dict["zoom", "max"].item()
        self.__dim = torch.clamp(
            tensor_dict["end"], torch.tensor([0, 0]), torch.tensor(self.env.dimensions)
        )
        self.__start = torch.clamp(
            tensor_dict["start"],
            torch.tensor([0, 0]),
            torch.tensor(self.env.dimensions) - self.__size,
        )

    def __curent_view_bounded_env(self) -> None:
        """
        this function bounds the enviroment given the current view
        """
        self.__max_level = self.__level.item()
        self.__min_level = 0
        self.__start = self.__coordinate

        self.__dim = self._get_bottom_right_coordinate()

    def __custom_min_level_bounded_env(self, min_level: int) -> None:
        """
        this function bound the enviroment given the current view while setting a boound on min_level

        Parameters
        ----------
        min_level : int
            min level to be customized
        """
        self.__min_level = min_level
        self.__max_level = self.__level.item()
        self.__start = self.__coordinate

        self.__dim = self._get_bottom_right_coordinate()

    def bounded_env(
        self,
        type: str = BoundingType.CURRENT,
        *,
        custom_min_level: Optional[int] = None,
        tensor_dict: Optional[TensorDict] = None,
    ):
        """_summary_

        Parameters
        ----------
        type : str, optional
            the type of bounding, by default CURRENT
        custom_min_level : Optional[int], optional
            the min level for CURRENT_MIN_LEVEL type , by default None
        tensor_dict : Optional[TensorDict], optional
            tensordict with full description for CUSTOM type, by default None

        Raises
        ------
        ParameterValueNotFound
            the value passed to type is not found
        """

        if type == BoundingType.CUSTOM:
            assert tensor_dict != None and isinstance(
                tensor_dict, TensorDict
            ), "You need to pass a tensordict"
            self.__custom_bounded_env(tensor_dict)

        elif type == BoundingType.CURRENT:
            self.__curent_view_bounded_env()

        elif type == BoundingType.CURRENT_MIN_LEVEL:
            assert custom_min_level != None and isinstance(
                custom_min_level, int
            ), "You need to pass a custom_min_level"
            self.__custom_min_level_bounded_env(custom_min_level)
        else:
            LOGGER.error(f"parameter value {type} in bounded_env isnt defined")
            raise ParameterValueNotFound(
                f"parameter value type not found, possible values are {BoundingType.CURRENT_MIN_LEVEL,BoundingType.CUSTOM,BoundingType.CURRENT}"
            )

        LOGGER.debug(
            f"---Observation class Bounding, New bounds----- \n\t start:{self.__start} \n\t dim:{self.__dim}\
                \n\t level:{self.__level} \n\t min_level:{self.__min_level} \n\t max_level: {self.__max_level}"
        )
        self.get_current()
    @abstractmethod
    def transform(self, img: Tensor) -> Tensor:
        """
        this function can be overwriten for transfroming the observation before providing it to the agent
        """
        return img

    def _post_process(self, img) -> torch.Tensor:
        """_summary_

        Parameters
        ----------
        img : _type_
            the observed image

        Returns
        -------
        torch.Tensor
            post_processed observation: RGB transfromed, Resized to fit size
        """
        img = torchvision.transforms.ToTensor()(img)
        if img.shape[0] > 3:
            img = img[:3, :, :]
        img = torchvision.transforms.Resize((self.__size, self.__size))(img)
        return self.transform(img)

    def _move(self, move: Optional[Tensor] = 0):
        """
        Parameters
        ----------
        move : Optional[Tensor], optional
            movement value on x and y , by default 0

        Returns
        -------
            image after moving
        """

        size_current_view = self.levels[self.__level.item()][1]
        multiplier = int(
            self.env.level_downsamples[self.levels[self.__level.item()][0]]
        )

        temp = size_current_view * multiplier
        new_coordinate = self.__coordinate + (move * temp)

        # fixing the new coordinate such that the observation is still in the scope of the wsi

        new_coordinate = torch.clamp(new_coordinate, self.__start, self.__dim - temp)

        self.__coordinate = new_coordinate

        img = self.env.read_region(
            new_coordinate.numpy().astype(np.int32),
            self.levels[self.__level.item()][0],
            (size_current_view, size_current_view),
        ).convert("RGB")
        return img

    def get_current(self) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """
        extract current view while making sure it respects the enviroment bounds and correct it otherwise

        Returns
        -------
        Union[Tensor,tuple[Tensor,Tensor]]
            return image or image and coordinate of the current observation.
        """

        self.__level = torch.clamp(self.__level, self.__min_level, self.__max_level)

        if self.__level == self.upper_bound:
            img = self.env.get_thumbnail(self.env.level_dimensions[-1])
        else:
            img = self._move()

        # transform image
        if self.return_coord == COOD_Required.LEVEL1:
            return self._post_process(img), self.__coordinate
        elif self.return_coord == COOD_Required.LEVEL2:
            return self._post_process(img), torch.cat(
                (self.__coordinate, torch.tensor([self.__level], dtype=torch.float32))
            )
        return self._post_process(img)

    def step(self, action: Union[list, Tensor]) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """perform a step in the environmentm updating th internal parameters and returning the next observation

        Parameters
        ----------
        action : Union[list,Tensor]
            contain x,y, and zoom

        Returns
        -------
        Union[Tensor,tuple[Tensor,Tensor]]
             return image or image and coordinate of the next observation.
        """
        action = torch.tensor(action,dtype=torch.float32)

        # 3rd element should be the zooming action

        zoom = action[2]

        current_level = self.__level.item()  # used to identify changes in level later on

        self.__level += int(zoom.item())
        self.__level = torch.clamp(self.__level, self.__min_level, self.__max_level)

        if self.__level == self.upper_bound:
            img = self.env.get_thumbnail(self.env.level_dimensions[-1])

        else:
            zoomed=abs(current_level-self.__level.item())
            # add a default movement from up-left corner to center (during zoom in and out) when self.__ZoomDynamics==ZoomLocation.CENTER
            if (
                self.__ZoomDynamics == ZoomLocation.CENTER
                and zoomed >0
            ):
                if zoom > 0: # add movement so the zoomed in patch was in the center when zooming out
                    action[0] -= (1-0.5**zoomed)/2 
                    action[1] -= (1-0.5**zoomed)/2
                else: # add movement to the center of the image when zooming in 
                    action[0] += 2**(zoomed-1)-0.5 
                    action[1] += 2**(zoomed-1)-0.5

            img = self._move(action[:2])

        # transform image
        if self.return_coord == COOD_Required.LEVEL1:
            return self._post_process(img), self.__coordinate
        elif self.return_coord == COOD_Required.LEVEL2:
            return self._post_process(img), torch.cat(
                (self.__coordinate, torch.tensor([self.__level], dtype=torch.float32))
            )
        return self._post_process(img)


class memory_mapped_observation(observation):
    ''' a class  that handels fast observation, by using the fully memory mapped sub-envs,
        saving some for future access with a queue os size BUFFER_SIZE.
    Raises
    ------
    MEM_Exception
        _description_
    '''
    BUFFER_SIZE=4
    DICTIONARY=OrderedDict()
    BITES_BOUNDS=100*2**20
    def __init__(self, size, env, coordinate, level, coord_required = COOD_Required.LEVEL0, ZoomDynamics = ZoomLocation.CENTER):
        
        super().__init__(size, env, coordinate, level, coord_required, ZoomDynamics)
        self.__env=env

    def bounded_env(self, type = BoundingType.CURRENT, *, custom_min_level = None, tensor_dict = None):
        '''_summary_
        implements Bounded_env of class Observation
        
        additionally it maps the bounded nev to memory using 'memory_mapped_openslide', and saves 
        these mapped sub-envs in a queue for fast access
        Raises
        ------
        MEM_Exception
            raise exception when the bounded env is too big to be mapped into main memory
        '''
        super().bounded_env(type, custom_min_level=custom_min_level, tensor_dict=tensor_dict)
        self.env=self.__env
        min_level,max_level = self._get_min_max_levels()
        start=self._get_start()
        dim=self._get_dim()

        key= f'{os.path.basename(self.env._filename)},{start},{dim},{min_level},{max_level}'
        
        
        if(key in self.DICTIONARY):

            self.env=self.DICTIONARY[key]
        else:
            width=dim[0]-start[0]
            height=dim[1]-start[1]
            if(width*height*3>self.BITES_BOUNDS):
                raise MEM_Exception(f"this Sub-ENV is too big to be mapped into main memory width {width} height {height}, check the Bites_support of the class {self.__class__.__name__}")
            
            if(len(self.DICTIONARY)>self.BUFFER_SIZE):
                self.DICTIONARY.popitem(last=False)
            
            self.DICTIONARY[key] =memory_mapped_openslide(self.__env, start, dim, min_level, max_level, self.levels)
    
            self.env = self.DICTIONARY[key] 
        
    def _move(self, move: Optional[Tensor] = 0):
        """
        Parameters
        ----------
        move : Optional[Tensor], optional
            movement value on x and y , by default 0

        Returns
        -------
            image after moving
        """
        ##################################### calculation of the new corrdinate stay the same
        size_current_view = self.levels[self._get_level().item()][1]
        multiplier = int(
            self.env.level_downsamples[self.levels[self._get_level().item()][0]]
        )

        temp = size_current_view * multiplier
        new_coordinate = self._get_coordinate() + (move * temp)

        # fixing the new coordinate such that the observation is still in the scope of the wsi

        new_coordinate = torch.clamp(new_coordinate, self._get_start(), self._get_dim() - temp)

        self._set_coordinate(new_coordinate)
        #####################################


        # what changes is the read_region dynamic, in this class mem is sacrificed for performance
        # hence there  are images stored for every level 
        img = self.env.read_region(
            new_coordinate.numpy().astype(np.int32),
            self._get_level().item(),
            (self._get_size(), self._get_size()),
        )
        return img