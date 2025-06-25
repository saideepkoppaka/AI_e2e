# Google Colab Overview Example

# 1. Check if running in Google Colab
import sys
IN_COLAB = 'google.colab' in sys.modules
print(f"Running in Google Colab: {IN_COLAB}")

# 2. Mount Google Drive
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted!")

# 3. Check for GPU availability
import torch

gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

# 4. Install a package (e.g., tqdm) if in Colab
if IN_COLAB:
    !pip install tqdm

# 5. Use tqdm for a progress bar demonstration
from tqdm import tqdm
import time

print("Progress bar example:")
for i in tqdm(range(10)):
    time.sleep(0.2)

# 6. Display an image using matplotlib
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sample Sine Wave")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()

# 7. Upload a file (if in Colab)
if IN_COLAB:
    from google.colab import files
    print("Please select a file to upload:")
    uploaded = files.upload()
    print("Uploaded files:", uploaded.keys())
