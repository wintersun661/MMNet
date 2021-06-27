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

        # optimizer
        self.parser.add_argument('--optimizer_type', type=str, default="SGD")
        self.parser.add_argument('--lr', type=float, default=1e-3)
        self.parser.add_argument('--momentum', type=float, default=0.9)

        # loss
        self.parser.add_argument('--loss_type', type=str, default="MSE")

        # data files
        self.parser.add_argument('--train_file', type=str, default='train.csv')
        self.parser.add_argument(
            '--validation_file', type=str, default='validation.csv')
        self.parser.add_argument('--test_file', type=str, default="test.csv")
