import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Builds a contrastive loss function based on the cosine similarity between the latent vectors.

    Args:
        nn (nn.Module): Pytorch module
    """

    def __init__(self, cof1=1e-2):
        """Initializes the contrastive loss regularization

        Args:
            cof1 (float, optional): Regularization hyperparameter. Defaults to 1e-2.
        """
        super(ContrastiveLoss, self).__init__()
        self.cof1 = cof1

    def forward(self, latent):
        """Forward pass of the contrastive loss regularization

        Args:
            latent (Tensor): Activations of layer to apply the loss metric

        Returns:
            Tensor: Loss value
        """

        loss = 0
        beyond_0 = torch.where(torch.sum(latent, axis=1) != 0)[0]
        new_latent = latent[beyond_0]
        for i in beyond_0:
            loss += sum(F.cosine_similarity(
                latent[i].unsqueeze(0), new_latent)) - 1

        loss = self.cof1 * loss / (2.0 * latent.shape[0])

        return loss


class DivergenceLoss(nn.Module):
    def __init__(self, batch_size, cof1=1e-2):
        """Divergence regularization for the latent space.

        This regularization tries to make the latent vectors sparse and different from each other.

        Args:
            batch_size (Int): The batch size of each update
            cof1 (Tensor, optional): Hyperparameter. Defaults to 1e-2.
        """
        super(DivergenceLoss, self).__init__()
        self.batch_size = batch_size
        self.cof1 = cof1

    def forward(self, latent):
        
        """Forward pass of the divergence regularization

        Args:
            latent (Tensor): Activations of layer to apply the loss metric

        Returns:
            Tensor: Loss value
        """
        
        loss = 0

        for i in range(self.batch_size):
            no_zero = torch.where(latent[i].squeeze() != 0)[0]
            single = latent[i][no_zero]
            loss += self.cof1 * \
                torch.sum(abs(single.reshape(-1, 1) - single)) / 2.0

        loss = loss / self.batch_size

        return loss

    # def to(self, device):
    #     self.device = device
    #     super().to(device)
    #     return self


# def to(self, device):
#     """Function to move the model to a specific device

#         Args:
#             device : hardware device to use

#         Returns:
#             Torch: Torch object on device
#         """
#     self.device = device
#     super().to(device)
#     return self
