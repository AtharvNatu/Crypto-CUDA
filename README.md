Cryptography algorithms implemented in parallel fashion

This project makes use of CUDA to parallelize existing encryption algorithms in order to achieve higher throughput in less time.

Status - 

AES 128-bit/192-bit/256-bit Implementation


Software Requirements - 

1. CUDA Toolkit Version 11.7
2. Microsoft CL Compiler or GNU GCC
3. Microsoft Visual Studio Code
4. Timer Library from NVIDIA 


Steps to run the code:

1) Windows

**NOTE**: If your NVIDIA CUDA-enabled compute device has more than 4GB of VRAM (Video Memory), please use x64 Native Tools Command Prompt, as Developer Command Prompt is a 32-bit application that cannot access memory larger than 4GB.

a) If you have Microsoft Visual C++ installed along with CUDA Toolkit, then just go for running the batch file as Build.bat inside Developer Command Prompt.

b) If you do not have Microsoft Visual C++ installed, you need to install it using Visual Studio, not Visual Studio Code :)


2) Linux:

a) Linux users only need to install gcc (the GNU Compiler Collection), which is included in most distributions.Also install the CUDA Toolkit according to your distribution's package manager.

b) After the above step is done, just run the Build.sh file.


3) macOS: Unfortunately, as macOS has dropped NVIDIA support :(, Mac users cannot test this code, from macOS Mojave (10.14) onwards. You can, however, test it up to macOS 10.13 i.e., macOS High Sierra if you have a compatible NVIDIA card.


Interpreting the Output

Basically, what you will get on your command prompt or terminal is a complete AES encryption and decryption status performed on a 6K x 6K image. You will also be able to view your GPU's detailed information.

Thanks :D
