from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .shiny_blender import ShinyBlenderDataset



dataset_dict = {'blender': BlenderDataset,
                'shiny': ShinyBlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF}