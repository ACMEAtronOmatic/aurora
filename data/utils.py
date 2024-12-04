import torch

def check_gpu_memory():
    if torch.cuda.is_available():
        mem = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = mem - allocated
        gb = free * 1e-9
        return gb
    else:
        return None


def print_debug(debug=True, *args):
    if debug:
        msg = " ".join(map(str, args))
        print(msg)