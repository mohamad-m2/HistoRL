import warnings
import numpy as np
import os
import torchvision
from ENV import LOGGER
from torch import Tensor
import openslide
import torch
from typing import Union
from PIL import Image
from enum import Enum
import pickle

EXTENSION_DICT = {
    "pth": (Tensor, torch.save, torch.load),
    "npy": (np.ndarray, np.save, np.load),
}


class COOD_Required(int, Enum):
    LEVEL0 = 0  # not required
    LEVEL1 = 1  # only cood
    LEVEL2 = 2  # cood and level


class ParameterValueNotFound(Exception):
    pass


def save_dict_versatile(data: dict, path: str, extension: str):
    current_type = EXTENSION_DICT[extension][0]
    saving_function = EXTENSION_DICT[extension][1]
    for key in data:
        if isinstance(data[key], dict):
            data[key] = save_dict_versatile(data[key], path, extension)
        elif isinstance(data[key], current_type):
            saving_function(data[key], os.path.join(path, key + "." + extension))
            data[key] = key + "." + extension
    return data


def load_dict_versatile(data: dict, path: str, extension: str, device: str):
    loading_function = EXTENSION_DICT[extension][2]
    for key in data:
        if isinstance(data[key], dict):
            data[key] = load_dict_versatile(data[key], path, extension, device)
        elif isinstance(data[key], str) and extension in data[key]:
            data[key] = loading_function(os.path.join(path, data[key])).to(device)
    return data


def save_annotations(file: str, data: dict) -> None:
    with open(file, "wb") as poly_file:
        pickle.dump(data, poly_file, pickle.HIGHEST_PROTOCOL)


def load_annotations(file: str) -> dict:
    with open(file, "rb") as poly_file:
        loaded_polygon = pickle.load(poly_file)
    return loaded_polygon


def filter(img: Union[Tensor, Image.Image]) -> bool:
    # filter out background patches using background ratio, std and mean value of pixles
    img = np.array(img)[:, :, :4]

    std = img[:, :, 0].std(), img[:, :, 1].std(), img[:, :, 2].std()
    max_channel_std = max(*std)
    white_ratio = np.sum(img.mean(axis=2) > 220) / (img.shape[0] * img.shape[1])

    if img.mean() > 230 or max_channel_std < 10 or white_ratio > 0.6:
        return True
    return False


def eliminate_wrong_extension(imgs_paths: list, supported: list) -> list:
    """filter out paths that arent supported

    Parameters
    ----------
    imgs_paths : list
        list of images
    supported : list
        supported extension

    Returns
    -------
    list
        a list containing only the supported images
    """
    cleaned = []
    for i in imgs_paths:
        if os.path.splitext(i)[1] in supported:
            cleaned.append(i)
        else:
            LOGGER.debug(f"eliminate_wrong_extension, utils, ENV: {i} eliminated")
    return cleaned


def correct_resolution_levels(
    highest_key: int, low_res_level: int, high_res_level: int
):

    if high_res_level < 0:
        high_res_level = 0

    if low_res_level > highest_key:
        low_res_level = highest_key + 1

    if low_res_level < 2:
        low_res_level = 2
    if high_res_level >= low_res_level:
        high_res_level = low_res_level - 1

    return low_res_level, high_res_level


def __extract_low_resolution_image(
    img: Union[openslide.OpenSlide, openslide.ImageSlide],
    levels: dict,
    low_res_level: int,
    key: int,
    grace: int,
):

    # paramreter extraction for reading images
    l_level = levels[low_res_level][0] if (low_res_level <= key) else None
    l_size = levels[low_res_level][1] if (low_res_level <= key) else None
    l_size_relative = (
        int(l_size * img.level_downsamples[l_level]) if (low_res_level <= key) else None
    )

    to_return = {
        "low_meta_data": {
            "l_level": low_res_level,
        }
    }
    low_img = None
    counter = grace

    while counter >= 0:
        if low_res_level == key + 1:

            low_img = img.get_thumbnail(img.level_dimensions[-1])
            to_return["low_meta_data"]["cood"] = (0, 0)
            to_return["low_meta_data"]["dim"] = img.dimensions
            break

        else:
            cood = (
                np.random.randint(0, img.dimensions[0] - l_size_relative),
                np.random.randint(0, img.dimensions[1] - l_size_relative),
            )
            low_img = img.read_region(cood, l_level, (l_size, l_size))

            if not filter(low_img):
                end = (cood[0] + l_size_relative, cood[1] + l_size_relative)
                to_return["low_meta_data"]["cood"] = cood
                to_return["low_meta_data"]["dim"] = end

                break
        counter -= 1

    if counter < 0:

        warnings.warn("one item missing bcz of white backgrounds in low res images")
        LOGGER.warn(
            f"generate_goal, utils, ENV: one item missing bcz of white backgrounds in low res images (grace={grace})"
        )
        return {}
    to_return["low_image"] = low_img
    return to_return


def __extract_high_resolution_image(
    img: Union[openslide.OpenSlide, openslide.ImageSlide],
    levels: dict,
    high_res_level: int,
    start: tuple,
    end: tuple,
    relative_position: bool,
    size: int,
    grace: int,
):

    h_level = levels[high_res_level][0]
    h_size = levels[high_res_level][1]
    h_size_relative = int(h_size * img.level_downsamples[h_level])
    assert (
        h_size_relative < end[0] - start[0]
    ), "high_res_level should be smaller than low_res_level"

    counter = grace
    x = 0
    y = 0
    high_img = None
    to_return = {
        "high_meta_data": {
            "h_level": high_res_level,
        }
    }
    # extract a random high level patch (goal)
    while counter >= 0:

        cood = (
            np.random.randint(start[0], end[0] - h_size_relative),
            np.random.randint(start[1], end[1] - h_size_relative),
        )

        high_img = img.read_region(cood, h_level, (h_size, h_size))

        if not filter(high_img):
            if (
                relative_position
            ):  # this option return the cood relative to the env (it was forgotten) #TODO check if i can use it
                x = size * (cood[0] - start[0]) / (end[0] - start[0])
                y = size * (cood[1] - start[1]) / (end[1] - start[1])
            else:
                x, y = cood[0] * 1.0, cood[1] * 1.0
            break
        counter -= 1

    if counter < 0:

        warnings.warn("one item missing bcz of white backgrounds in high res images")
        LOGGER.warn(
            f"generate_goal, utils, ENV: one item missing bcz of white backgrounds in high res images (grace={grace})"
        )
        return {}

    to_return["high_meta_data"]["cood"] = cood
    to_return["high_meta_data"]["dim"] = (x + h_size, y + h_size)
    to_return["high_image"] = high_img
    to_return["cood"] = (x, y)
    return to_return


def generate_goal(
    img: Union[openslide.OpenSlide, openslide.ImageSlide],
    levels: dict,
    size: int,
    relative_position: bool = False,
    low_res_level: list = [2],
    high_res_level: list = [0],
    grace: int = 20, **kwargs
):
    """generate an intitial state and goal for the RL pretext-task

    Parameters
    ----------
    img : Union[openslide.OpenSlide , openslide.ImageSlide]
        wsi image
    levels : dict
        the levels dictionary
    size : int
        size of observation
    low_res_level : int, optional
        level where initial state is located, by default 2
    high_res_level : int, optional
        goal level, by default 0
    grace : int, optional
        nb of times to try, by default 20

    Returns
    -------
    dict
        dictionar containing goal and initial state info
    """

    key = max(levels.keys())


    high_res_level=np.random.choice(high_res_level) if 'probs_high' not in kwargs else np.random.choice(high_res_level,p=kwargs['probs_high'])
    low_res_level=np.random.choice(low_res_level) if 'probs_low' not in kwargs else np.random.choice(low_res_level,p=kwargs['probs_low'])
    high_res_level=np.int32(high_res_level).item()
    low_res_level=np.int32(low_res_level).item()

    low_res_level, high_res_level = correct_resolution_levels(
        key, low_res_level, high_res_level
    )

    low_meta_data = __extract_low_resolution_image(
        img, levels, low_res_level, key, grace
    )
    if low_meta_data == {}:
        return {}

    high_meta_data = __extract_high_resolution_image(
        img,
        levels,
        high_res_level,
        low_meta_data["low_meta_data"]["cood"],
        low_meta_data["low_meta_data"]["dim"],
        relative_position,
        size,
        grace,
    )
    if high_meta_data == {}:
        return {}

    return low_meta_data | high_meta_data


def generate_goal_environment_fixed(
    img: Union[openslide.OpenSlide, openslide.ImageSlide],
    levels: dict,
    size: int,
    environment_data: dict,
    relative_position: bool = False,
    high_res_level: list = [0],
    grace: int = 20,
    **kwargs
):
    """generate a goal for the RL pretext-task given the initial state

    Parameters
    ----------
    img : Union[openslide.OpenSlide , openslide.ImageSlide]
        wsi image
    levels : dict
        the levels dictionary
    size : int
        size of observation
    environment_data : dict
        necessary data about the enviroment size and location top generate the goal
    high_res_level : int, optional
        goal level, by default 0
    grace : int, optional
        nb of times to try, by default 20

    Returns
    -------
    dict
        dictionar containing goal and initial state info
    """
    high_res_level=np.random.choice(high_res_level) if 'probs_high' not in kwargs else np.random.choice(high_res_level,p=kwargs['probs_high'])

    to_combine = __extract_high_resolution_image(
        img,
        levels,
        high_res_level,
        environment_data["low_meta_data"]["cood"],
        environment_data["low_meta_data"]["dim"],
        relative_position,
        size,
        grace,
    )
    if to_combine == {}:
        return {}
    return to_combine | environment_data

class memory_mapped_openslide:
    #TODO handel creation of standalone memory mapped images
    '''
    This class creates memory-mapped images of OpenSlide images at different levels.
    It does not create its own slide from input data,
    but instead reads a portion of a Whole Slide Image (WSI) and maps it into memory.

    Key Features:

    -Reads a part of a WSI and maps it into memory

    -Allows reading of regions from the mapped part using WSI-level information
    (level, coordinates)

    -Designed to maintain compatibility with existing classes that handle WSIs,
    by minimizing changes to function calls when switching to memory-mapped parts.
    

    The class doesnt handel input checking ( coordinate and size compatibility
    with the maped reagion ) in order to minimze computation head on read-region.
    '''
    def __create_img(self,img:Union[openslide.OpenSlide, openslide.ImageSlide],
                     level:int,
                     width:int,
                     height:int):

        """
        Private method to create a  memory map of specific level of the openslide as numpy array image.
        wdith and height are at level 0.

        Parameters
        ----------
        img : Union[openslide.OpenSlide, openslide.ImageSlide]
            openslide image
        level : int
            level of which this image should belong to
        width : int
            width at level 0
        height : int
            height at level 0
        Returns
        -------
        dict
            dictionary containing the memory mapped image and the rescaling factor for coordinates
        """
        actual_width=np.ceil(width/(2**level)).astype(np.int32)
        actual_height=np.ceil(height/(2**level)).astype(np.int32)

        # these directly read from the levels of The WSI.
        factor=int(self.level_downsamples[self.levels[level][0]])
        width_tmp=np.ceil(width/factor).astype(np.int32)
        height_tmp=np.ceil(height/factor).astype(np.int32)
        ###################

        data=img.read_region(self.start,self.levels[level][0],(width_tmp,height_tmp)).convert("RGB")
        data=data.resize((actual_width, actual_height))

        return {'image':np.array(data),'rescale':2**level}

    def __init__(self, img: Union[openslide.OpenSlide, openslide.ImageSlide],
                 start:Union[tuple[int,int],Tensor,np.ndarray],
                 dimensions:Union[tuple[int,int],Tensor,np.ndarray],
                 start_level: int,
                 end_level: int,
                 levels:dict
                 ):
        self.levels=levels
        self.level_downsamples=img.level_downsamples
        self.start=np.array(start)
        dimensions=np.array(dimensions)
        self.images={}
        
        width=dimensions[0]-self.start[0]
        height=dimensions[1]-self.start[1]
        for i in range(start_level,end_level+1):
            self.images[i]=self.__create_img(img,i,width,height)
        self.images['thumbnail']= self.images[end_level]['image']
    def read_region(self, coordinates: tuple[int,int], level: int, size: tuple[int,int])->np.ndarray:
        '''_summary_

        Parameters
        ----------
        coordinates : tuple[int,int]
            where to read from in the WSI
        level : int
            level to read from
        size : tuple[int,int]
            the size of the region to read

        Returns
        -------
        np.ndarray
            the requested section 
        '''
        coorinates=(coordinates-self.start)/self.images[level]['rescale']
        coorinates=coorinates.astype(np.int32)
        return self.images[level]['image'][coorinates[1]:coorinates[1]+size[1],coorinates[0]:coorinates[0]+size[0]]
        
    def get_thumbnail(self,size: tuple[int,int]):
        return self.images['thumbnail']

        
