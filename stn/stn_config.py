from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import six
import os
import os.path as osp
import math
import argparse


parser = argparse.ArgumentParser(description="Softmax loss classification")
# data
parser.add_argument('--tps_inputsize', nargs='+', type=int, default=[32, 128])
parser.add_argument('--tps_outputsize', nargs='+', type=int, default=[32, 128])
#-----------------------------------1-------------------------------------------#
# parser.add_argument('--STN_ON', action='store_true', #store_true 是指带触发action时为真，不触发则为假
#                     help='add the stn head.')
parser.add_argument('--STN_ON', type=bool, default=True, #store_true 是指带触发action时为真，不触发则为假
                    help='add the stn head.')

parser.add_argument('--tps_margins', nargs='+', type=float, default=[0.05,0.05])
parser.add_argument('--stn_activation', type=str, default='none')
parser.add_argument('--num_control_points', type=int, default=20)
parser.add_argument('--stn_with_dropout', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', default=True, type=bool,
                    help='whether use cuda support.')

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args