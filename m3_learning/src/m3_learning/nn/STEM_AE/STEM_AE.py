import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional Block with 3 convolutional layers, 1 layer normalization layer with ReLU and ResNet

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, t_size, n_step):
        """Initializes the convolutional block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(ConvBlock, self).__init__()
        self.cov1d_1 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov1d_2 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov1d_3 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.norm_3 = nn.LayerNorm(n_step)
        self.relu_4 = nn.ReLU()

    def forward(self, x):
        """Forward pass of the convolutional block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        out = self.cov1d_1(x)
        out = self.cov1d_2(out)
        out = self.cov1d_3(out)
        out = self.norm_3(out)
        out = self.relu_4(out)
        out = out.add(x_input)

        return out


class IdentityBlock(nn.Module):

    """Identity Block with 1 convolutional layers, 1 layer normalization layer with ReLU
    """

    def __init__(self, t_size, n_step):
        """Initializes the identity block

        Args:
            t_size (int): Size of the convolution kernel
            n_step (int): Input shape of normalization layer
        """

        super(IdentityBlock, self).__init__()
        self.cov1d_1 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass of the identity block

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """

        x_input = x
        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    """Encoder block

    Args:
        nn (nn.Module): Torch module class
    """

    def __init__(self, original_step_size, pool_list, embedding_size, conv_size):
        """Build the encoder

        Args:
            original_step_size (Int): the x and y size of input image 
            pool_list (List): the list of parameter for each 2D MaxPool layer 
            embedding_size (Int): the value for number of channels
            conv_size (Int): the value of filters number goes to each block 
        """

        super(Encoder, self).__init__()

        blocks = []

        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]

        number_of_blocks = len(pool_list)

        blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(nn.MaxPool2d(pool_list[0], stride=pool_list[0]))

        for i in range(1, number_of_blocks):
            original_step_size = [original_step_size[0]//pool_list[i-1], original_step_size[1]//pool_list[i-1]]
            blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(nn.MaxPool2d(pool_list[i], stride=pool_list[i]))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)

        original_step_size = [original_step_size[0]//pool_list[-1], original_step_size[1]//pool_list[-1]]
        input_size = original_step_size[0]*original_step_size[1]

        self.cov2d = nn.Conv2d(1, conv_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov2d_1 = nn.Conv2d(conv_size, 1, 3, stride=1, padding=1, padding_mode='zeros')

        self.relu_1 = nn.ReLU()

        self.dense = nn.Linear(input_size, embedding_size)

    def forward(self, x):
        """Forward pass of the encoder

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: output tensor
        """
        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        selection = self.relu_1(out)

        return selection
