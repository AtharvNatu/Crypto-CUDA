clear
nvcc -o AES AES.cu
./AES Data/Ventura.bmp Key.txt Data/EncryptedImage.BMP Data/DecryptedImage.BMP 1024
./AES Data/Ventura.bmp Key.txt Data/EncryptedImage.BMP Data/DecryptedImage.BMP 2048