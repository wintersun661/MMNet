from options import BaseOptions
# options used when training with train.py


class OptionParser(BaseOptions.OptionParser):
    def __init__(self):
        super(OptionParser, self).__init__()
        self.name = "framework_train"

    def initialize(self):
        super().initialize()

        # training hyperparameters
        self.parser.add_argument('--epoch', type=int, default=20)
        self.parser.add_argument('--resize', type=str, default="224,320")
        self.parser.add_argument('--max_kps_num', type=int, default=50)
        self.parser.add_argument('--split_type', type=str, default="trn")

        # optimizer
        self.parser.add_argument('--optimizer_type', type=str, default="SGD")
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, default=0.999)
        self.parser.add_argument('--momentum', type=float, default=0.9)

        # loss
        self.parser.add_argument('--loss_type', type=str, default="BCE")