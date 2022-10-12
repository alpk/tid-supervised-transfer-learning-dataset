import torchvision.models as models
from torch.nn import Parameter
from models.GCN import *
import torch
import torch.nn as nn

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class multilabel_gcn(nn.Module):
    def __init__(self, model, adj, num_classes, in_channel=300, t=0,  attribute_class=0):
        super(multilabel_gcn, self).__init__()

        #self.features = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.data_bn = model.data_bn
        self.l1 = model.l1
        self.l2 = model.l2
        self.l3 = model.l3
        self.l4 = model.l4
        self.l5 = model.l5
        self.l6 = model.l6
        self.l7 = model.l7
        self.l8 = model.l8
        self.l9 = model.l9
        self.l10 = model.l10
        self.fc1 = model.fc1
        self.fc2 = model.fc2
        self.fc_attr1 = model.fc_attr1
        self.fc_attr2 = model.fc_attr2

        """
        self.attribute_class = 0
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.fc_attr1 = nn.Linear(256, 256)
        self.fc_attr2 = nn.Linear(256, self.attribute_class)
        """

        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 128)
        self.gc2 = GraphConvolution(128, 256)
        self.relu = nn.LeakyReLU(0.2)

        #adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(adj).float())


    def forward(self, x, inp,keep_prob=0.9):

        x = x.unsqueeze(-1)
        x = x.permute(0,2,3,1,4)
        #  N is batch size, C = 3 is size of input feature, T = 300 is number of frame, V = 25 is the number of joint, M = 2 is number of people in one frame
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x, 1.0)
        x = self.l2(x, 1.0)
        x = self.l3(x, 1.0)
        x = self.l4(x, 1.0)
        x = self.l5(x, 1.0)
        x = self.l6(x, 1.0)
        x = self.l7(x, keep_prob)
        x = self.l8(x, keep_prob)
        x = self.l9(x, keep_prob)
        x = self.l10(x, keep_prob)

        # N*M,C,T,V
        c_new = x.size(1)

        # print(x.size())
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        attr = self.fc_attr2(self.fc_attr1(x))

        x =  self.fc2(self.fc1(x))

        x_flat = x.view(x.size(0), -1)

        inp = self.pooling(x)




        adj = gen_adj(self.A).detach()
        x2 = self.gc1(inp, adj)
        x2 = self.relu(x2)
        x2 = self.gc2(x2, adj)

        x2 = x2.transpose(0, 1)
        x2 = torch.matmul(x2, x2)

        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


"""
def multilabel_gcn(args, num_classes, num_channels, joint_groups,  t, adj=None, in_channel=300):

    if args.model_type == 'GCN':
        feature_size = 256
        model = Model(num_class=num_classes, num_point=num_channels,
                      num_person=1, groups=16, in_channels=3, graph='graph.sign_27.Graph',
                      graph_args={'labeling_mode': 'spatial',
                                  'joint_groups': joint_groups,
                                  'num_point': num_channels})
    else:
        assert False, 'Unsupported Model type'

    return 
"""