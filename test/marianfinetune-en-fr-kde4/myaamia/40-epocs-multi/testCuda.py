import torch
print(torch.cuda.is_available())  # should be True
print(torch.cuda.current_device()) # should be 0 (internal index for visible GPUs)
print(torch.cuda.get_device_name(0))
