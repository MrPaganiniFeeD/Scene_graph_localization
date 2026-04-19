import torch
print(torch.cuda.is_available())          # Должно быть True
print(torch.cuda.get_device_capability()) # Должно быть (12, 0) или (12, x)