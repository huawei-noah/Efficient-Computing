'''
The losses come from DIST and DKD
https://github.com/megvii-research/mdistiller
https://github.com/hunto/DIST_KD

Modifications by Zhiwei Hao (haozhw@bit.edu.cn) and Jianyuan Guo (jianyuan_guo@outlook.com)
'''

from .bkd import BinaryKLDiv
from .correlation import Correlation
from .crd import CRD
from .dist import DIST
from .dkd import DKD
from .kd import KLDiv
from .review import ReviewKD
from .rkd import RKD
