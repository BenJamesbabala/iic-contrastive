import torch.nn as nn
import torch.nn.functional as F
from .base import BasicBlock, ResNet, ResNetTrunk

__all__ = ['ClusterNetMulHead']

#Represents an IIC head, which is composed by a number of sub_head
class ClusterNetMulHeadHead(nn.Module):
  def __init__(self, config, output_k, num_sub_heads=1, softmax=True):
    super(ClusterNetMulHeadHead, self).__init__()
    self.batchnorm_track = config.batchnorm_track
    self.num_sub_heads = num_sub_heads
    #Default IIC loss requires softmax
    if softmax:
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, output_k),
            nn.Softmax(dim=1)) for _ in range(self.num_sub_heads)])
    #SimCLR does not
    else:
        self.heads = nn.ModuleList([nn.Linear(512 * BasicBlock.expansion, output_k) for _ in range(self.num_sub_heads)])

  def forward(self, x):
    results = []
    for i in range(self.num_sub_heads):
      results.append(self.heads[i](x))
    return results  

#A trunked ResNet34
class ClusterNetTrunk(ResNetTrunk):
  def __init__(self, config):
    super(ClusterNetTrunk, self).__init__()

    self.batchnorm_track = config.batchnorm_track

    block = BasicBlock
    layers = [3, 4, 6, 3]

    in_channels = config.in_channels
    self.inplanes = 64
    self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64, track_running_stats=self.batchnorm_track)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    if config.input_sz == 96:
      avg_pool_sz = 7
    elif config.input_sz == 64:
      avg_pool_sz = 5
    elif config.input_sz == 32:
      avg_pool_sz = 3
    else:
      avg_pool_sz = 2
    self.avgpool = nn.AvgPool2d(avg_pool_sz, stride=1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

#Complete ResNet34 with three heads
class ClusterNetMulHead(ResNet):
  def __init__(self, config):

    super(ClusterNetMulHead, self).__init__()
    self.batchnorm_track = config.batchnorm_track
    self.trunk = ClusterNetTrunk(config)
    self.head_A = ClusterNetMulHeadHead(config, output_k=config.output_k_A, num_sub_heads=config.num_sub_heads)
    self.head_B = ClusterNetMulHeadHead(config, output_k=config.output_k_B, num_sub_heads=config.num_sub_heads)
    self.head_C = ClusterNetMulHeadHead(config, output_k=config.output_k_C, num_sub_heads=1, softmax=False)    
    self._initialize_weights()

  def forward(self, x, head="B", trunk_features=False):
    # default is "B" for use by eval code
    # training script switches between A, B and C
    x = self.trunk(x)
    if trunk_features:  # for feature evaluation
      return x
    if head == "A":
      x = self.head_A(x)
    elif head == "B":
      x = self.head_B(x)
    elif head == "C":
      x = self.head_C(x) 
    else:
      assert (False)
    return x