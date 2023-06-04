from typing import Union
import torch
from torch import nn
from torch.nn import functional as F

import torch
from torch.nn import functional as F


def lstm_pbm(dl_model, pb_model, param_func, x, z):
    """
    Differential parameter learning

    z (normalized input) -> lstm -> param -> + x (not normalized) -> pbm -> q
    Parameters will be denormalized in pbm model

    Parameters
    ----------
    dl_model
        lstm model
    pb_model
        physics-based model
    param_func
        function used to limit the range of params; now it is sigmoid or clamp function
    x
        not normalized data used for physical model; a sequence-first 3-dim tensor. [sequence, batch, feature]
    z
        normalized data used for DL model; a sequence-first 3-dim tensor. [sequence, batch, feature]

    Returns
    -------
    torch.Tensor
            one time forward result
    """
    gen = dl_model(z)
    if torch.isnan(gen).any():
        raise ValueError("Error: NaN values detected. Check your data firstly!!!")
    # we set all params' values in [0, 1] and will scale them when forwarding
    if param_func == "sigmoid":
        params_ = F.sigmoid(gen)
    elif param_func == "clamp":
        params_ = torch.clamp(gen, min=0.0, max=1.0)
    else:
        raise NotImplementedError(
            "We don't provide this way to limit parameters' range!! Please choose sigmoid or clamp"
        )
    # just get one-period values, here we use the final period's values
    params = params_[-1, :, :]
    # Please put p in the first location and pet in the second
    q = pb_model(x[:, :, : pb_model.feature_size], params)
    return q


def ann_pbm(dl_model, pb_model, param_func, x, z):
    """
    Differential parameter learning

    z (normalized input) -> ann -> param -> + x (not normalized) -> pbm -> q
    Parameters will be denormalized in pbm model

    Parameters
    ----------
    dl_model
        ann model
    pb_model
        physics-based model
    param_func
        function used to limit the range of params; now it is sigmoid or clamp function
    x
        not normalized data used for physical model; a sequence-first 3-dim tensor. [sequence, batch, feature]
    z
        normalized data used for DL model; a 2-dim tensor. [batch, feature]

    Returns
    -------
    torch.Tensor
        one time forward result
    """
    gen = dl_model(z)
    if torch.isnan(gen).any():
        raise ValueError("Error: NaN values detected. Check your data firstly!!!")
    # we set all params' values in [0, 1] and will scale them when forwarding
    if param_func == "sigmoid":
        params = F.sigmoid(gen)
    elif param_func == "clamp":
        params = torch.clamp(gen, min=0.0, max=1.0)
    else:
        raise NotImplementedError(
            "We don't provide this way to limit parameters' range!! Please choose sigmoid or clamp"
        )
    # Please put p in the first location and pet in the second
    q = pb_model(x[:, :, : pb_model.feature_size], params)
    return q


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.5):
        super(SimpleLSTM, self).__init__()
        self.linearIn = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, dropout=dr)
        self.linearOut = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x0 = F.relu(self.linearIn(x))
        out_lstm, (hn, cn) = self.lstm(x0)
        out = self.linearOut(out_lstm)
        return out


class SimpleAnn(torch.nn.Module):
    def __init__(
        self,
        nx: int,
        ny: int,
        hidden_size: Union[int, tuple, list] = None,
        dr: Union[float, tuple, list] = 0.0,
    ):
        """
        A simple multi-layer NN model with final linear layer

        Parameters
        ----------
        nx
            number of input neurons
        ny
            number of output neurons
        hidden_size
            a list/tuple which contains number of neurons in each hidden layer;
            if int, only one hidden layer except for hidden_size=0
        dr
            dropout rate of layers, default is 0.0 which means no dropout;
            here we set number of dropout layers to (number of nn layers - 1)
        """
        super(SimpleAnn, self).__init__()
        linear_list = torch.nn.ModuleList()
        dropout_list = torch.nn.ModuleList()
        if (
            hidden_size is None
            or (type(hidden_size) is int and hidden_size == 0)
            or (type(hidden_size) in [tuple, list] and len(hidden_size) < 1)
        ):
            linear_list.add_module("linear1", torch.nn.Linear(nx, ny))
        else:
            if type(hidden_size) is int:
                if type(dr) in [tuple, list]:
                    dr = dr[0]
                linear_list.add_module("linear1", torch.nn.Linear(nx, hidden_size))
                # dropout layer do not have additional weights, so we do not name them here
                dropout_list.append(torch.nn.Dropout(dr))
                linear_list.add_module("linear2", torch.nn.Linear(hidden_size, ny))
            else:
                linear_list.add_module("linear1", torch.nn.Linear(nx, hidden_size[0]))
                if type(dr) is float:
                    dr = [dr] * len(hidden_size)
                else:
                    if len(dr) != len(hidden_size):
                        raise ArithmeticError(
                            "We set dropout layer for each nn layer, please check the number of dropout layers"
                        )
                # dropout_list.add_module("dropout1", torch.nn.Dropout(dr[0]))
                dropout_list.append(torch.nn.Dropout(dr[0]))
                for i in range(len(hidden_size) - 1):
                    linear_list.add_module(
                        "linear%d" % (i + 1 + 1),
                        torch.nn.Linear(hidden_size[i], hidden_size[i + 1]),
                    )
                    dropout_list.append(
                        torch.nn.Dropout(dr[i + 1]),
                    )
                linear_list.add_module(
                    "linear%d" % (len(hidden_size) + 1),
                    torch.nn.Linear(hidden_size[-1], ny),
                )
        self.linear_list = linear_list
        self.dropout_list = dropout_list

    def forward(self, x):
        for i, model in enumerate(self.linear_list):
            if i == 0:
                if len(self.linear_list) == 1:
                    return model(x)
                out = F.relu(self.dropout_list[i](model(x)))
            else:
                if i == len(self.linear_list) - 1:
                    # in final layer, no relu again
                    return model(out)
                else:
                    out = F.relu(self.dropout_list[i](model(out)))


class KernelConv(nn.Module):
    def __init__(self, a, theta, kernel_size):
        """
        The convolution kernel for the convolution operation in routing module

        We use two-parameter gamma distribution to determine the unit hydrograph,
        which comes from [mizuRoute](http://www.geosci-model-dev.net/9/2223/2016/)

        Parameters
        ----------
        a
            shape parameter
        theta
            timescale parameter
        kernel_size
            the size of conv kernel
        """
        super(KernelConv, self).__init__()
        self.a = a
        self.theta = theta
        routa = self.a.repeat(kernel_size, 1).unsqueeze(-1)
        routb = self.theta.repeat(kernel_size, 1).unsqueeze(-1)
        self.uh_gamma = uh_gamma(routa, routb, len_uh=kernel_size)

    def forward(self, x):
        """
        1d-convolution calculation

        Parameters
        ----------
        x
            x is a sequence-first variable, so the dim of x is [seq, batch, feature]

        Returns
        -------
        torch.Tensor
            convolution
        """
        # dim: permute from [len_uh, batch, feature] to [batch, feature, len_uh]
        uh = self.uh_gamma.permute(1, 2, 0)
        # the dim of conv kernel in F.conv1d is out_channels, in_channels (feature)/groups, width (seq)
        # the dim of inputs in F.conv1d are batch, in_channels (feature) and width (seq),
        # each element in a batch should has its own conv kernel,
        # hence set groups = batch_size and permute input's batch-dim to channel-dim to make "groups" work
        inputs = x.permute(2, 1, 0)
        batch_size = x.shape[1]
        # conv1d in NN is different from the general convolution: it is lack of a flip
        outputs = F.conv1d(
            inputs, torch.flip(uh, [2]), groups=batch_size, padding=uh.shape[-1] - 1
        )
        # permute from [feature, batch, seq] to [seq, batch, feature]
        return outputs[:, :, : -(uh.shape[-1] - 1)].permute(2, 1, 0)


def uh_conv(x, uh_made) -> torch.Tensor:
    """
    Function for 1d-convolution calculation

    Parameters
    ----------
    x
        x is a sequence-first variable, so the dim of x is [seq, batch, feature]
    uh_made
        unit hydrograph from uh_gamma or other unit-hydrograph method

    Returns
    -------
    torch.Tensor
        convolution, [seq, batch, feature]; the length of seq is same as x's
    """
    uh = uh_made.permute(1, 2, 0)
    # the dim of conv kernel in F.conv1d is out_channels, in_channels (feature)/groups, width (seq)
    # the dim of inputs in F.conv1d are batch, in_channels (feature) and width (seq),
    # each element in a batch should has its own conv kernel,
    # hence set groups = batch_size and permute input's batch-dim to channel-dim to make "groups" work
    inputs = x.permute(2, 1, 0)
    batch_size = x.shape[1]
    # conv1d in NN is different from the general convolution: it is lack of a flip
    outputs = F.conv1d(
        inputs, torch.flip(uh, [2]), groups=batch_size, padding=uh.shape[-1] - 1
    )
    # cut to same shape with x and permute from [feature, batch, seq] to [seq, batch, feature]
    return outputs[:, :, : x.shape[0]].permute(2, 1, 0)


def uh_gamma(a, theta, len_uh=10):
    """
    A simple two-parameter Gamma distribution as a unit-hydrograph to route instantaneous runoff from a hydrologic model

    The method comes from mizuRoute -- http://www.geosci-model-dev.net/9/2223/2016/

    Parameters
    ----------
    a
        shape parameter
    theta
        timescale parameter
    len_uh
        the time length of the unit hydrograph

    Returns
    -------
    torch.Tensor
        the unit hydrograph, dim: [seq, batch, feature]

    """
    # dims of a: time_seq (same all time steps), batch, feature=1
    m = a.shape
    assert len_uh <= m[0]
    # aa > 0, here we set minimum 0.1 (min of a is 0, set when calling this func); First dimension of a is repeat
    aa = F.relu(a[0:len_uh, :, :]) + 0.1
    # theta > 0, here set minimum 0.5
    theta = F.relu(theta[0:len_uh, :, :]) + 0.5
    # len_f, batch, feature
    t = (
        torch.arange(0.5, len_uh * 1.0)
        .view([len_uh, 1, 1])
        .repeat([1, m[1], m[2]])
        .to(aa.device)
    )
    denominator = (aa.lgamma().exp()) * (theta**aa)
    # [len_f, m[1], m[2]]
    w = 1 / denominator * (t ** (aa - 1)) * (torch.exp(-t / theta))
    w = w / w.sum(0)  # scale to 1 for each UH
    return w
