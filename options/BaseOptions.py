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

        # hyper-parameters shared by train & test
        self.parser.add_argument('--batch', type=int, default=10)

        # device set
        self.parser.add_argument('--gpu', type=str, default="0")

        # data source
        self.parser.add_argument('--data_path', type=str, default="./dataset")

        # checkpoint path
        self.parser.add_argument(
            '--checkpoint_path', type=str, default="./checkpoints")

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        return self.opt
