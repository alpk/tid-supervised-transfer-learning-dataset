import torch.nn as nn
from . import util as u

class DANN(nn.Module):

    def __init__(self, feature_extractor, num_classes, feature_size):
        super(DANN, self).__init__()

        self.feature_extractor = feature_extractor
        self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(256, 100))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(feature_size, num_classes))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(feature_size, 2))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        # self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        # self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


    def forward(self, input_data, alpha):

        # input_data = input_data.expand(input_data.s, 3, 28, 28)
        _, features = self.feature_extractor(input_data)
        # feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = u.GradientReverseFunction.apply(features, alpha)
        class_output = self.class_classifier(features)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output, features