cls
nvcc -o AES AES.cu
AES.exe Data/Ventura.bmp Key.txt Data/EncryptedImage.BMP Data/DecryptedImage.BMP 1024
AES.exe Data/Ventura.bmp Key.txt Data/EncryptedImage.BMP Data/DecryptedImage.BMP 2048