import torch, cv2, av
print("CUDA available:", torch.cuda.is_available())
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("OpenCV:", cv2.__version__)
print("PyAV:", av.__version__)
print("CUDA available:", torch.cuda.is_available(), "| Torch CUDA:", torch.version.cuda)
print("OpenCV:", cv2.__version__)