# This is inspired by the authors's implementation of `Unsupervised Scalable Representation Learning for Multivariate Time Series`.
# In order to study in more details the architecture, we added:

# 1) The possibility of replacing the Average Poling layer by a fully-connected layer (by setting the global_average_pooling
#    attribute of CausalCNNEncoder() to False.
#    This step aims at squeezing the temporal dimension and aggregating all temporal information in a fixed-size vector (the embedding).
#    Our intuition is that the Global Average Pooling is used here as a regularizer, as it doesn't introduce any parameters
#    to optimize, contrary to a fully-connected layer. (More detailed information can be found in the article `Network In Network (Lin et al. 2014)`
#    Furthermore, using global average pooling is more convenient as the whole model is therefore independant from the time-series length,
#    whereas using a fully-connected layer to squeeze the temporal dimension requires to know the length of the input time series.
#    We still led some experiment to study performances change.

# 2) Another idea we might investigate is the introduction of attention mechanism after the Causal CNN blocks.

import torch


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2)

        # Residual connection
        self.upordownsample = (
            torch.nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """
    Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output
           channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, in_channels, channels, depth, out_channels, kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [
                CausalConvolutionBlock(
                    in_channels_block, channels, kernel_size, dilation_size
                )
            ]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [
            CausalConvolutionBlock(channels, out_channels, kernel_size, dilation_size)
        ]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the
           causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    def __init__(
        self,
        in_channels,
        channels,
        depth,
        reduced_size,
        out_channels,
        kernel_size,
        global_average_pooling=True,
        time_series_length=None,
    ):
        super(CausalCNNEncoder, self).__init__()

        # If using fully-connected layer to squeeze the temporal dimension, the length of the time series should be known.
        assert global_average_pooling == True or (
            global_average_pooling == False and time_series_length is not None
        )

        causal_cnn = CausalCNN(in_channels, channels, depth, reduced_size, kernel_size)

        if global_average_pooling:
            reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        else:
            reduce_size = torch.nn.Linear(time_series_length, 1)

        # Squeezes the third dimension (time)
        squeeze = SqueezeChannels()

        linear = torch.nn.Linear(reduced_size, out_channels)

        self.network = torch.nn.Sequential(causal_cnn, reduce_size, squeeze, linear)

    def forward(self, x):
        return self.network(x)


class Chomp1d(torch.nn.Module):
    """
    Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)