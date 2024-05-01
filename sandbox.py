import torch
from kan import KANLayer, KANLayerArguments


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


def main():
    device = get_device()
    dtype = torch.bfloat16

    args = KANLayerArguments(3, 5, device=device)

    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    print(f"Running on {device} in {dtype} precision.")

    layer = KANLayer(args)
    x = torch.normal(0, 1, size=(100, 3)).to(device)
    y, preacts, postacts, postspline = layer(x)

    print(f"y.shape: {y.shape}")  # torch.Size([100, 5])

if __name__ == "__main__":
    main()