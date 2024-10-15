# HelloCUDA
Hello World in CUDA

Certainly! Setting up a CUDA environment on a Linux machine in Azure is a great way to work with CUDA. Here's a step-by-step guide to get you started:

1. Create an Azure VM with GPU support:
   - Log into the Azure portal (portal.azure.com)
   - Create a new Virtual Machine
   - Choose a Linux distribution (Ubuntu is a popular choice)
   - Select a VM size that includes NVIDIA GPUs (look for N-series VMs)
   - Set up authentication and networking as needed

2. Connect to your VM:
   - Use SSH to connect to your VM from your local machine

3. Update the system:
```bash
sudo apt update && sudo apt upgrade -y
```

4. Install necessary tools:
```bash
sudo apt install build-essential
```

5. Install NVIDIA GPU drivers:
```bash
sudo apt install linux-headers-$(uname -r)
sudo apt install nvidia-driver-xxx  # replace xxx with the latest version number
```

6. Reboot the VM:
```bash
sudo reboot
```

7. Verify GPU driver installation:
```bash
nvidia-smi
```

8. Install CUDA Toolkit:
   - Go to NVIDIA's CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive
   - Choose the version you want (latest is usually best)
   - Follow the installation instructions for your chosen Linux distribution

   For example, for CUDA 11.8 on Ubuntu 20.04:

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

9. Set up environment variables:
   Add these lines to your ~/.bashrc file:

```bash
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

   Then, source the file:
```bash
source ~/.bashrc
```

10. Verify CUDA installation:
```bash
nvcc --version
```

Now that you have CUDA set up, let's create a simple CUDA program to test the environment:



```cuda
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

```

Save this code in a file named `cuda_test.cu`.

11. Compile and run the test program:
```bash
nvcc cuda_test.cu -o cuda_test
./cuda_test
```

If everything is set up correctly, you should see the output: "Hello World from GPU!"

12. For development, you might want to install an IDE. Visual Studio Code is a popular choice:
```bash
sudo snap install --classic code
```

13. Install CUDA development tools:
```bash
sudo apt install nvidia-cuda-toolkit
```

This setup provides you with a full CUDA development environment on an Azure Linux VM. You can now start developing and running CUDA programs.

Would you like me to explain any part of this process in more detail, or provide a more complex CUDA example to test your setup?
