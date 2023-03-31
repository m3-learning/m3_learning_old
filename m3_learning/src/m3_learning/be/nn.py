import torch.nn as nn
import torch
from ..nn.SHO_fitter.SHO import SHO_fit_func_torch
from ..optimizers.AdaHessian import AdaHessian
from ..nn.random import random_seed
from torch.utils.data import DataLoader
import time


class SHO_Model(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        if ~hasattr(self.dataset, "SHO_scaler"):
            self.dataset.SHO_Scaler()

        # Input block of 1d convolution
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=8, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=8, out_channels=6, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=6, out_channels=4, kernel_size=5),
            nn.SELU(),
        )

        # fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(256, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
        )

        # 2nd block of 1d-conv layers
        self.hidden_x2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
        )

        # Flatten layer
        self.flatten_layer = nn.Flatten()

        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(26, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, 4),
        )

    def forward(self, x, n=-1):
        # output shape - samples, (real, imag), frequency
        x = torch.swapaxes(x, 1, 2)
        x = self.hidden_x1(x)
        xfc = torch.reshape(x, (n, 256))  # batch size, features
        xfc = self.hidden_xfc(xfc)
        # batch size, (real, imag), timesteps
        x = torch.reshape(x, (n, 2, 128))
        x = self.hidden_x2(x)
        cnn_flat = self.flatten_layer(x)
        encoded = torch.cat((cnn_flat, xfc), 1)  # merge dense and 1d conv.
        embedding = self.hidden_embedding(encoded)  # output is 4 parameters

        # corrects the scaling of the parameters
        unscaled_param = (
            embedding *
            torch.tensor(self.dataset.SHO_scaler.var_[0:4] ** 0.5).cuda()
            + torch.tensor(self.dataset.SHO_scaler.mean_[0:4]).cuda()
        )

        # passes to the pytorch fitting function
        fits = SHO_fit_func_torch(
            unscaled_param, self.dataset.wvec_freq, device="cuda")

        # extract and return real and imaginary
        real = torch.real(fits)
        real_scaled = (real - torch.tensor(self.dataset.real_scaler.mean).cuda()) / torch.tensor(
            self.dataset.real_scaler.std
        ).cuda()
        imag = torch.imag(fits)
        imag_scaled = (imag - torch.tensor(self.dataset.imag_scaler.mean).cuda()) / torch.tensor(
            self.dataset.imag_scaler.std
        ).cuda()
        out = torch.stack((real_scaled, imag_scaled), 2)
        return out


class SHO_NN_Model:

    def __init__(self,
                 model,
                 seed=42,
                 lr=0.1,
                 **kwargs):
        super().__init__()
        self.model = model
        self.seed = seed
        self.lr = lr
        self.__dict__.update(kwargs)

    def train(self,
              data_train,
              batch_size,
              epochs=5,
              loss_func=torch.nn.MSELoss(),
              optimizer='Adam',
              **kwargs):

        # Note that the seed will behave differently on different hardware targets (GPUs)
        random_seed(seed=self.seed)

        torch.cuda.empty_cache()

        # selects the optimizer
        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters())
        elif optimizer == "AdaHessian":
            optimizer = AdaHessian(self.model.parameters(), lr=0.1)

        # instantiate the dataloader
        train_dataloader = DataLoader(data_train, batch_size=batch_size)

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = 0.0
            total_num = 0

            self.model.train()

            for train_batch in train_dataloader:

                pred = self.model(train_batch.double().cuda())

                optimizer.zero_grad()

                loss = loss_func(train_batch.double().cuda(), pred)
                loss.backward(create_graph=True)
                train_loss += loss.item() * pred.shape[0]
                total_num += pred.shape[0]

                optimizer.step()

            train_loss /= total_num

            print("epoch : {}/{}, recon loss = {:.8f}".format(epoch +
                  1, epochs, train_loss))
            print("--- %s seconds ---" % (time.time() - start_time))

            torch.save(self.model.state_dict(),
                       'Trained Models/SHO Fitter/model.pth')
