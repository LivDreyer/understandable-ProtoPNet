# test_protopnet.py
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Import ProtoPNet from the ppnet folder
try:
    import ppnet
    print("ProtoPNet module imported successfully!")
except ModuleNotFoundError:
    print("Could not find ppnet module. Check folder structure and Python path.")
