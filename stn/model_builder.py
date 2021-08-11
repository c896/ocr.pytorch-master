#------------------------------------------#
from torch import nn
from torch.nn import functional as F
from stn.tps_spatial_transformer import TPSSpatialTransformer
from stn.stn_head import STNHead
from stn.stn_config import get_args
import sys
global_args = get_args(sys.argv[1:])

class ModelBuilder(nn.Module):
    def __init__(self, STN_ON=False):
        super(ModelBuilder, self).__init__()

        self.STN_ON = STN_ON
        self.tps_inputsize = global_args.tps_inputsize

        if self.STN_ON:
            self.tps = TPSSpatialTransformer(
            output_image_size=tuple(global_args.tps_outputsize),
            num_control_points=global_args.num_control_points,
            margins=tuple(global_args.tps_margins))
            self.stn_head = STNHead(
            in_planes=3,
            num_ctrlpoints=global_args.num_control_points,
            activation=global_args.stn_activation)

    def forward(self, input_dict):
        # return_dict = {}
        # return_dict['output'] = {}
        
        x = input_dict['images']
        # h,w=x.shape[-2:]
        # self.tps_inputsize=[32,int(w/h*32)]
        # rectification #倾斜校正
        if self.STN_ON:
        # input images are downsampled before being fed into stn_head.
            stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            x, _ = self.tps(x, ctrl_points)
        # if not self.training:
        #     # save for visualization
        #     return_dict['output']['ctrl_points'] = ctrl_points
        #     return_dict['output']['rectified_images'] = x

        return x,ctrl_points