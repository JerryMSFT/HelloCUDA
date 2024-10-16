# Setting up GPU-enabled VMs for CUDA Development

## Azure GPU VM Setup

1. Log in to the Azure portal (portal.azure.com)

2. Click on "Create a resource" and search for "Virtual machine"

3. Click "Create" under "Virtual machine"

4. Choose your subscription and resource group (create a new one if needed)

5. Name your VM and choose a region

6. For Image, select a Linux distribution (e.g., Ubuntu Server 20.04 LTS)

7. For Size, click "See all sizes" and filter for GPU. Choose an NC-series VM (e.g., NC6s_v3)

8. Set up authentication (SSH public key recommended)

9. In Networking, allow SSH (port 22) inbound

10. Review and create the VM

11. Once deployed, find the public IP address in the VM overview

12. SSH into your VM:
    ```
    ssh -i <path_to_private_key> azureuser@<vm_public_ip>
    ```

13. Follow the CUDA setup instructions provided earlier to install NVIDIA drivers and CUDA toolkit

## Oracle Cloud Infrastructure (OCI) GPU VM Setup

1. Log in to the OCI Console (cloud.oracle.com)

2. Open the navigation menu and click "Compute" > "Instances"

3. Click "Create Instance"

4. Name your instance

5. For Image, choose a GPU-compatible OS (e.g., Oracle Linux 7.9 or Ubuntu 20.04)

6. For Shape, click "Change shape"
   - Select "GPU" from the menu
   - Choose a GPU shape (e.g., VM.GPU2.1)

7. In Network, create a new VCN or use an existing one
   - Ensure a public subnet is selected
   - Select "Assign a public IPv4 address"

8. Add your SSH key (upload or paste public key)

9. Click "Create" to launch the instance

10. Once the instance is running, find the public IP address

11. SSH into your instance:
    ```
    ssh -i <path_to_private_key> opc@<instance_public_ip>
    ```

12. Install CUDA:
    - For Oracle Linux:
      ```
      sudo yum-config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
      sudo yum clean all
      sudo yum -y install nvidia-driver-latest-dkms cuda
      ```
    - For Ubuntu, follow the CUDA setup instructions provided earlier

13. Reboot the instance:
    ```
    sudo reboot
    ```

14. After reboot, verify the CUDA installation:
    ```
    nvidia-smi
    nvcc --version
    ```

Remember to stop or terminate your instances when not in use to avoid unnecessary charges.

# Instructions to Clone and Compile CUDA Program

1. Connect to your GPU-enabled VM (Azure or OCI) via SSH:
   ```
   ssh -i <path_to_private_key> username@<vm_public_ip>
   ```

2. Ensure Git is installed on your VM:
   ```
   sudo apt update
   sudo apt install git
   ```

3. Clone your Git repository:
   ```
   git clone https://github.com/JerryMSFT/HelloCUDA
   ```

4. Navigate to the cloned repository:
   ```
   cd HelloCUDA
   ```

5. Locate your CUDA source file. Assuming it's named `cuda_comprehensive_test.cu`, compile it using nvcc:
   ```
   nvcc -o cuda_comprehensive_test cuda_comprehensive_test.cu
   ```

6. Run the compiled program:
   ```
   ./cuda_comprehensive_test
   ```

7. If you need to make changes to the code, edit the file using a text editor like nano:
   ```
   nano cuda_comprehensive_test.cu
   ```
   Make your changes, save (Ctrl+O, then Enter), and exit (Ctrl+X).

8. After making changes, recompile the program (repeat step 5) and run it again.

Troubleshooting:
- If you encounter a "command not found" error for nvcc, ensure CUDA is properly installed and the PATH is set correctly.
- If you face compilation errors, double-check that your CUDA installation matches the version expected by your code.
- For runtime errors, verify that your GPU drivers are correctly installed using `nvidia-smi`.
