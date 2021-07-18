import argparse

import os


class OptionParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.name = "framework_base"
        self.initialized = False

    def initialize(self):
        # system description
        self.parser.add_argument('--name', type=str, default=self.name)

        # model type
        self.parser.add_argument('--model', type=str, default="original")

        # task specific setting
        self.parser.add_argument('--benchmark', type=str, default="pfpascal")
        self.parser.add_argument('--thresh_type', type=str, default="auto")
        self.parser.add_argument(
            '--backbone_name', type=str, default="resnet101")
        # resnet101 resnet50 resnext-101 fcn-resnet101
        self.parser.add_argument('--ms_rate', type=int, default=4)
        self.parser.add_argument('--feature_channel', type=int, default=21)

        # hyper-parameters shared by train & test
        self.parser.add_argument('--batch', type=int, default=5)

        # device set
        self.parser.add_argument('--gpu', type=str, default="0")

        # data source
        self.parser.add_argument(
            '--data_path', type=str, default="/data/Datasets_SCOT/")

        # checkpoint path
        self.parser.add_argument(
            '--checkpoint_path', type=str, default="./checkpoints_debug")

        # visualization path
        self.parser.add_argument(
            '--visualization_path', type=str, default='visualization')

        # model selection
        self.parser.add_argument(
            '--model_type', type=str, default="MMNet"
        )

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
