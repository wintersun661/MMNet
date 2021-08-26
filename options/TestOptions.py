from options import BaseOptions
# options used when training with train.py


class OptionParser(BaseOptions.OptionParser):
    def __init__(self):
        super(OptionParser, self).__init__()
        self.name = "framework_train"

    def initialize(self):
        super().initialize()

        #

        # training hyperparameters
        self.parser.add_argument('--ckp_name', type=str, default="./checkpoints/ckp_pascal_fcnres101.pth")
        self.parser.add_argument('--log_path',type=str,default='./logs/')
        self.parser.add_argument('--resize', type=str, default="224,320")
        self.parser.add_argument('--max_kps_num', type=int, default=50)
        self.parser.add_argument('--split_type', type=str, default="test")

        # evaluation alpha for pck
        self.parser.add_argument('--alpha', type=float, default=0.1)

        # evaluation resolution level
        self.parser.add_argument('--resolution', type=int, default=2)
