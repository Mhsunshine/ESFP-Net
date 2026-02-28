from models.resnet_snn import MultiStepSEWResNet,MultiStepBasicBlock,multi_step_sew_resnet18,multi_step_sew_resnet50
# from spikingjelly.clock_driven import neuron, functional, surrogate
# from spikingjelly.clock_driven import functional, surrogate
import torch.nn as nn
# from models import neuron

class SNNModule(nn.Module):
    def __init__(self, in_channels, timesteps):
        super(SNNModule, self).__init__()
        self.encoder = get_encoder_snn(in_channels, timesteps)
        # out_encoder = 512 # sew_resnet_18

        #self.fc = nn.Linear(out_encoder, n_classes, bias=False)

    def forward(self, x):
        x = x.permute(2, 0, 1, 3, 4)
        # IMPORTANT: always apply reset_net before a new forward
        functional.reset_net(self.encoder)
        #functional.reset_net(self.fc)

        x = self.encoder(x)
        #x = self.fc(x)

        return x


def get_encoder_snn(in_channels: int, T: int):
    # resnet = MultiStepSEWResNet(
    #     block= MultiStepBasicBlock,
    #     layers=[2, 2, 2, 2],
    #     zero_init_residual=True,
    #     T=T,
    #     cnf="ADD",
    #     multi_step_neuron=neuron.MultiStepIFNode,
    #     detach_reset=True,
    #     surrogate_function=surrogate.ATan(),
    # )
    resnet = multi_step_sew_resnet18(multi_step_neuron = neuron.MultiStepIFNode)
    # resnet = multi_step_sew_resnet18(multi_step_neuron = neuron.MultiStepIFNode)

    if in_channels != 3:
        resnet.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet


def Sew_resnet():
    return SNNModule(2,8)