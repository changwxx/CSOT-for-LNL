import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, head_type="mlp", feat_dim=128, num_classes=10, drop=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.drop = drop
        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.head = nn.Linear(512*block.expansion, 512*block.expansion)

        dim_in = 512*block.expansion
        if head_type == 'linear':
            self.conhead = nn.Linear(dim_in, feat_dim)
        elif head_type == 'mlp':
            self.conhead = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head_type))
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, lin=0, lout=5, feat=False, confeat=False):
        out = x
        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)
        if lin < 2 and lout > 0:
            out = self.layer1(out)
            if self.drop > 0.0:
                out = self.dropout(out)
        if lin < 3 and lout > 1:
            out = self.layer2(out)
            if self.drop > 0.0:
                out = self.dropout(out)
        if lin < 4 and lout > 2:
            out = self.layer3(out)
            if self.drop > 0.0:
                out = self.dropout(out)
        if lin < 5 and lout > 3:
            out = self.layer4(out)
            if self.drop > 0.0:
                out = self.dropout(out)
        if (not feat) and (not confeat):
            if lout > 4:
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                if self.drop > 0.0:
                    out = self.dropout(out)
                    print("Dropout 1", self.drop)
                out = self.head(out)
                if self.drop > 0.0:
                    out = self.relu(out)
                    out = self.dropout(out)
                    print("Dropout 2", self.drop)
                out = self.linear(out)
            return out
        elif feat and (not confeat):
            if lout > 4:
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.head(out)
                out_feat = out
                out = self.linear(out)
            return out, out_feat
        elif (not feat) and confeat:
            if lout > 4:
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.head(out)
                con_feat = self.conhead(out)
                out = self.linear(out)
            return out, con_feat
        else:
            if lout > 4:
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                out = self.head(out)
                out_feat = out
                con_feat = self.conhead(out)
                out = self.linear(out)
            return out, out_feat, con_feat



def ResNet18(num_classes=10, drop=0.0):
    return ResNet(PreActBlock, [2,2,2,2], num_classes=num_classes, drop=drop)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())


class Net(nn.Module):
    def __init__(self, func, pred_dim=128, **karg):
        super(Net, self).__init__()
        self.encoder = func(**karg)
        self.feat_dim = self.encoder.linear.in_features
        self.cls = nn.Linear(self.encoder.linear.in_features, karg["num_classes"])
        # self.cls = ProtoCLS(self.encoder.linear.in_features, karg["num_classes"])
        # https://github.com/facebookresearch/simsiam
        self.predictor = nn.Sequential(nn.Linear(self.feat_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, self.feat_dim)) # output layer

    def forward(self, x, ssl=False):
        if not ssl:
            _, x = self.encoder(x, feat=True)
            feat = x.view(x.size(0), -1)
            out = self.cls(feat)
            return feat, out
        else:
            """
            Input:
                x1: first views of images
                x2: second views of images
            Output:
                p1, p2, z1, z2: predictors and targets of the network
                See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
            """
            # compute features for one view
            _, z1 = self.encoder(x[0], feat=True) # NxC
            _, z2 = self.encoder(x[1], feat=True) # NxC

            z1 = z1.view(z1.size(0), -1)
            z2 = z2.view(z2.size(0), -1)

            out1 = self.cls(z1)
            out2 = self.cls(z2)

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            return z1, z2, out1, out2, p1, p2 

        



class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """
    def __init__(self, in_dim, out_dim):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.weight_norm()

    def forward(self, x):
        norm_feat = F.normalize(x)
        out = self.fc(norm_feat) 
        return norm_feat, out
    
    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))


class CLS(nn.Module):
    """
    a classifier made up of projection head and prototype-based classifier
    """
    def __init__(self, in_dim, out_dim, hidden_mlp=2048, feat_dim=256, temp=0.05):
        super(CLS, self).__init__()
        self.projection_head = nn.Sequential(
                            nn.Linear(in_dim, hidden_mlp),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_mlp, feat_dim))
        self.ProtoCLS = ProtoCLS(feat_dim, out_dim, temp)

    def forward(self, x):
        before_lincls_feat = self.projection_head(x)
        after_lincls = self.ProtoCLS(before_lincls_feat)
        return before_lincls_feat, after_lincls