import torch

def SHO_fit_func_torch(parms,
                       wvec_freq,
                       device='cpu'):
    
    """_summary_

    Returns:
        _type_: _description_
    """
    Amp = parms[:, 0].type(torch.complex128)
    w_0 = parms[:, 1].type(torch.complex128)
    Q = parms[:, 2].type(torch.complex128)
    phi = parms[:, 3].type(torch.complex128)
    wvec_freq = torch.tensor(wvec_freq)

    Amp = torch.unsqueeze(Amp, 1)
    w_0 = torch.unsqueeze(w_0, 1)
    phi = torch.unsqueeze(phi, 1)
    Q = torch.unsqueeze(Q, 1)

    wvec_freq = wvec_freq.to(device)

    numer = Amp * torch.exp((1.j) * phi) * torch.square(w_0)
    den_1 = torch.square(wvec_freq)
    den_2 = (1.j) * wvec_freq.to(device) * w_0 / Q
    den_3 = torch.square(w_0)

    den = den_1 - den_2 - den_3

    func = numer / den

    return func
