import cpuinfo
import torch

def find_device():
        
    cpudata = cpuinfo.get_cpu_info()["brand_raw"]
    cpuname = cpudata.split(" ")[1]

    if cpuname == "M1":
        device = "mps"
    elif torch.cuda.device_count():
        device = "cuda"
    else:
        device = "cpu"

    print(f"You are running on a {device}")
    
    return device