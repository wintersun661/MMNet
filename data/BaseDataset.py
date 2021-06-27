import torch

import csv


def readFile(filePath):
    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        feature_dict = []
        data_list = []
        for row in csv_reader:
            if line_count == 0:
                for name in row:
                    feature_dict.append(name)
                line_count += 1
            else:
                data_list.append(row)

    return feature_dict, data_list


class CustomizedDataset(torch.utils.data.Dataset):

    def __init__(self, filePath):
        super(CustomizedDataset, self).__init__()
        self.filePath = filePath
        self.feature_dict, self.data_list = readFile(self.filePath)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        X = []
        for i in range(len(self.feature_dict)-1):
            X.append(float(self.data_list[index][i]))

        X = torch.FloatTensor(X)
        y = torch.FloatTensor([float(self.data_list[index][-1])])

        return X, y
