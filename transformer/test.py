import torch
import time 
# Check if a Metal device is available
if torch.backends.mps.is_available():
    print("MPS (Metal Performance Shaders) backend is available.")
else:
    print("MPS backend is not available.")


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tensor1 = torch.rand(10000, 50000)
tensor2 = torch.rand(50000, 200)

start = time.perf_counter()
print("cpu accelerated")
result1 = tensor1 @ tensor2
end = time.perf_counter()

print("time spent ", end-start)
print("gpu accelerated")
start = time.perf_counter()

tensor1 = tensor1.to(device)
tensor2 = tensor2.to(device)

result2 = tensor1 @ tensor2
print("time spent ", end-start)

result2 = result2.detach().cpu()
print("result of both runs", result2 == result1)

end = time.perf_counter()