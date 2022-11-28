// Standard Headers
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>

// CUDA Headers
#include <cuda.h>
#include "helper_timer.h"

using namespace std;

// Macros
#define BYTE unsigned char
#define LENGTH 16
#define MAX 256
#define FAILURE -1
#define SUCCESS 1

// Variable Declarations
class aes_block
{
    public:
        BYTE block[LENGTH];
};

BYTE sbox[] =
{   /*0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f */
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76, /*0*/ 
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0, /*1*/
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15, /*2*/
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75, /*3*/
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84, /*4*/
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf, /*5*/
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8, /*6*/ 
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2, /*7*/
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73, /*8*/
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb, /*9*/
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79, /*a*/
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08, /*b*/
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a, /*c*/
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e, /*d*/
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf, /*e*/
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16  /*f*/
};

FILE* encFile = NULL;
FILE* decFile = NULL;

ifstream input_file_stream, key_file_stream;

int threads_per_block;
int block_number;
int num_zero_pending;
int input_file_length;

aes_block* aes_block_array = NULL;
aes_block* cuda_aes_block_array = NULL;
BYTE *cuda_key = NULL;

BYTE key[LENGTH * (14 + 1)];
int key_length = 0;
int block_length = LENGTH;
int expanded_key_length;

char temp[LENGTH];

cudaDeviceProp dev_prop;
cudaError_t result = cudaSuccess;

// Code
void print_cuda_device_properties(void)
{
	// Code
	printf("\nCUDA INFORMATION : \n");
	printf("\n**************************************************************************************************\n");

	cudaError_t ret_cuda_rt;
	int dev_count;

	ret_cuda_rt = cudaGetDeviceCount(&dev_count);

	if (ret_cuda_rt != cudaSuccess)
		printf("\nCUDA Runtime API Error - cudaGetDeviceCount() Failed Due To %s\n", cudaGetErrorString(ret_cuda_rt));

	else if (dev_count == 0)
	{
		printf("\nNo CUDA Supported Devices Found On This System ... Exiting !!!\n");
		return;
	}

	else
	{
		printf("Total Number Of CUDA Supporting GPU Device/Devices On This System : %d\n", dev_count);

		for (int i = 0; i < dev_count; i++)
		{
			int driverVersion = 0, runtimeVersion = 0;

			ret_cuda_rt = cudaGetDeviceProperties(&dev_prop, i);

			if (ret_cuda_rt != cudaSuccess)
			{
				printf("%s in %s at line %d\n", cudaGetErrorString(ret_cuda_rt), __FILE__, __LINE__);
				return;
			}

			printf("\n");

			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			printf("================================================================================================\n");
			printf("***** CUDA DRIVER AND RUNTIME INFORMATION *****\n");
			printf("================================================================================================\n");
			printf("CUDA Driver Version					: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
			printf("CUDA Runtime Version					: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
			printf("\n");
			printf("================================================================================================\n");

			printf("***** GPU DEVICE GENERAL INFORMATION *****\n");
			printf("================================================================================================\n");
			printf("GPU Device Number					: %d\n", i);
			printf("GPU Device Name						: %s\n", dev_prop.name);
			printf("GPU Device Compute Compatibility			: %d.%d\n", dev_prop.major, dev_prop.minor);
			printf("GPU Device Clock Rate					: %d\n", dev_prop.clockRate);
			printf("GPU Device Type						: %s", dev_prop.integrated ? "Integrated (On-Board)\n" : "Discrete (Card)\n");
			printf("\n");
			printf("================================================================================================\n");

			printf("***** GPU DEVICE MEMORY INFORMATION *****\n");
			printf("================================================================================================\n");
			printf("GPU Device Total Memory					: %.0f GB = %.0f MB = %llu Bytes\n", ((float)dev_prop.totalGlobalMem / 1048576.0f) / 1024.0f, (float)dev_prop.totalGlobalMem / 1048576.0f, (unsigned long long)dev_prop.totalGlobalMem);
			printf("GPU Device Constant Memory				: %lu Bytes\n", (unsigned long)dev_prop.totalConstMem);
			printf("GPU Device Shared Memory Per SMProcessor		: %lu\n", (unsigned long)dev_prop.sharedMemPerBlock);
			printf("\n");
			printf("================================================================================================\n");

			printf("***** GPU DEVICE MULTIPROCESSOR INFORMATION *****\n");
			printf("================================================================================================\n");
			printf("GPU Device Number Of SMProcessors			: %d\n", dev_prop.multiProcessorCount);
			printf("GPU Device Number Of Registers Per SMProcessor		: %d\n", dev_prop.regsPerBlock);
			printf("\n");
			printf("================================================================================================\n");

			printf("***** GPU DEVICE THREAD INFORMATION *****\n");
			printf("================================================================================================\n");
			printf("GPU Device Maximum Number Of Threads Per SMProcessor	: %d\n", dev_prop.maxThreadsPerMultiProcessor);
			printf("GPU Device Maximum Number Of Threads Per Block		: %d\n", dev_prop.maxThreadsPerBlock);
			printf("GPU Device Threads In Warp				: %d\n", dev_prop.warpSize);
			printf("GPU Device Maximum Thread Dimensions			: %d, %d, %d\n", dev_prop.maxThreadsDim[0], dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
			printf("GPU Device Maximum Grid Dimensions			: %d, %d, %d\n", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1], dev_prop.maxGridSize[2]);
			printf("\n");
			printf("================================================================================================\n");

			printf("***** GPU DEVICE DRIVER INFORMATION *****\n");
			printf("================================================================================================\n");
			printf("GPU Device Has ECC Support				: %s\n", dev_prop.ECCEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			printf("GPU Device CUDA Driver Mode ( TCC Or WDDM )		: %s\n", dev_prop.tccDriver ? "TCC ( Tesla Compute Cluster Driver )" : "WDDM ( Windows Display Driver Model )");
#endif

			printf("================================================================================================\n");

			printf("\n**************************************************************************************************\n");

		}
	}
}

__device__ void sub_bytes(BYTE state[], BYTE sbox[])
{
    for (int i = 0; i < LENGTH; i++)
        state[i] = sbox[state[i]];
}

__device__ void add_round_key(BYTE state[], BYTE round_key[])
{
    for (int i = 0; i < LENGTH; i++)
        state[i] = state[i] ^ round_key[i];
}

__device__ void shift_rows(BYTE state[], BYTE shift_tab[])
{
    BYTE h[LENGTH];
    memcpy(h, state, LENGTH);

    for (int i = 0; i < LENGTH; i++)
        state[i] = h[shift_tab[i]];
}

__device__ void mix_columns(BYTE state[], BYTE time[])
{
    for (int i = 0; i < LENGTH; i = i + 4)
    {
        BYTE s0 = state[i + 0];
        BYTE s1 = state[i + 1];
        BYTE s2 = state[i + 2];
        BYTE s3 = state[i + 3];

        BYTE h = s0 ^ s1 ^ s2 ^ s3;

        state[i + 0] = state[i + 0] ^ h ^ time[s0 ^ s1];
        state[i + 1] = state[i + 1] ^ h ^ time[s1 ^ s2];
        state[i + 2] = state[i + 2] ^ h ^ time[s2 ^ s3];
        state[i + 3] = state[i + 3] ^ h ^ time[s3 ^ s0];
    }
}

__device__ void mix_columns_inverse(BYTE state[], BYTE time[])
{
    for (int i = 0; i < LENGTH; i = i + 4)
    {
        BYTE s0 = state[i + 0];
        BYTE s1 = state[i + 1];
        BYTE s2 = state[i + 2];
        BYTE s3 = state[i + 3];

        BYTE h = s0 ^ s1 ^ s2 ^ s3;
        BYTE xh = time[h];
        BYTE h1 = time[time[xh ^ s0 ^ s2]] ^ h;
        BYTE h2 = time[time[xh ^ s1 ^ s3]] ^ h;

        state[i + 0] = state[i + 0] ^ h1 ^ time[s0 ^ s1];
        state[i + 1] = state[i + 1] ^ h2 ^ time[s1 ^ s2];
        state[i + 2] = state[i + 2] ^ h1 ^ time[s2 ^ s3];
        state[i + 3] = state[i + 3] ^ h2 ^ time[s3 ^ s0];
    }
}

__device__ void aes_init(BYTE sbox[], BYTE shift_row_tab[], BYTE sbox_inverse[], BYTE time[], BYTE shift_row_tab_inverse[])
{
    shift_row_tab[0]=0;
    shift_row_tab[1]=5;
    shift_row_tab[2]=10;
    shift_row_tab[3]=15;
    shift_row_tab[4]=4;
    shift_row_tab[5]=9;
    shift_row_tab[6]=14;
    shift_row_tab[7]=3;
    shift_row_tab[8]=8;
    shift_row_tab[9]=13;
    shift_row_tab[10]=2;
    shift_row_tab[11]=7;
    shift_row_tab[12]=12;
    shift_row_tab[13]=1;
    shift_row_tab[14]=6;
    shift_row_tab[15]=11;

    sbox[0] = 0x63;sbox[1] = 0x7c;sbox[2] = 0x77;sbox[3] = 0x7b;sbox[4] = 0xf2;sbox[5] = 0x6b;sbox[6] = 0x6f;sbox[7] = 0xc5;sbox[8] = 0x30;sbox[9] = 0x1;sbox[10] = 0x67;sbox[11] = 0x2b;sbox[12] = 0xfe;sbox[13] = 0xd7;sbox[14] = 0xab;sbox[15] = 0x76;
    sbox[16] = 0xca;sbox[17] = 0x82;sbox[18] = 0xc9;sbox[19] = 0x7d;sbox[20] = 0xfa;sbox[21] = 0x59;sbox[22] = 0x47;sbox[23] = 0xf0;sbox[24] = 0xad;sbox[25] = 0xd4;sbox[26] = 0xa2;sbox[27] = 0xaf;sbox[28] = 0x9c;sbox[29] = 0xa4;sbox[30] = 0x72;sbox[31] = 0xc0;
    sbox[32] = 0xb7;sbox[33] = 0xfd;sbox[34] = 0x93;sbox[35] = 0x26;sbox[36] = 0x36;sbox[37] = 0x3f;sbox[38] = 0xf7;sbox[39] = 0xcc;sbox[40] = 0x34;sbox[41] = 0xa5;sbox[42] = 0xe5;sbox[43] = 0xf1;sbox[44] = 0x71;sbox[45] = 0xd8;sbox[46] = 0x31;sbox[47] = 0x15;
    sbox[48] = 0x4;sbox[49] = 0xc7;sbox[50] = 0x23;sbox[51] = 0xc3;sbox[52] = 0x18;sbox[53] = 0x96;sbox[54] = 0x5;sbox[55] = 0x9a;sbox[56] = 0x7;sbox[57] = 0x12;sbox[58] = 0x80;sbox[59] = 0xe2;sbox[60] = 0xeb;sbox[61] = 0x27;sbox[62] = 0xb2;sbox[63] = 0x75;
    sbox[64] = 0x9;sbox[65] = 0x83;sbox[66] = 0x2c;sbox[67] = 0x1a;sbox[68] = 0x1b;sbox[69] = 0x6e;sbox[70] = 0x5a;sbox[71] = 0xa0;sbox[72] = 0x52;sbox[73] = 0x3b;sbox[74] = 0xd6;sbox[75] = 0xb3;sbox[76] = 0x29;sbox[77] = 0xe3;sbox[78] = 0x2f;sbox[79] = 0x84;
    sbox[80] = 0x53;sbox[81] = 0xd1;sbox[82] = 0x0;sbox[83] = 0xed;sbox[84] = 0x20;sbox[85] = 0xfc;sbox[86] = 0xb1;sbox[87] = 0x5b;sbox[88] = 0x6a;sbox[89] = 0xcb;sbox[90] = 0xbe;sbox[91] = 0x39;sbox[92] = 0x4a;sbox[93] = 0x4c;sbox[94] = 0x58;sbox[95] = 0xcf;
    sbox[96] = 0xd0;sbox[97] = 0xef;sbox[98] = 0xaa;sbox[99] = 0xfb;sbox[100] = 0x43;sbox[101] = 0x4d;sbox[102] = 0x33;sbox[103] = 0x85;sbox[104] = 0x45;sbox[105] = 0xf9;sbox[106] = 0x2;sbox[107] = 0x7f;sbox[108] = 0x50;sbox[109] = 0x3c;sbox[110] = 0x9f;sbox[111] = 0xa8;
    sbox[112] = 0x51;sbox[113] = 0xa3;sbox[114] = 0x40;sbox[115] = 0x8f;sbox[116] = 0x92;sbox[117] = 0x9d;sbox[118] = 0x38;sbox[119] = 0xf5;sbox[120] = 0xbc;sbox[121] = 0xb6;sbox[122] = 0xda;sbox[123] = 0x21;sbox[124] = 0x10;sbox[125] = 0xff;sbox[126] = 0xf3;sbox[127] = 0xd2;
    sbox[128] = 0xcd;sbox[129] = 0xc;sbox[130] = 0x13;sbox[131] = 0xec;sbox[132] = 0x5f;sbox[133] = 0x97;sbox[134] = 0x44;sbox[135] = 0x17;sbox[136] = 0xc4;sbox[137] = 0xa7;sbox[138] = 0x7e;sbox[139] = 0x3d;sbox[140] = 0x64;sbox[141] = 0x5d;sbox[142] = 0x19;sbox[143] = 0x73;
    sbox[144] = 0x60;sbox[145] = 0x81;sbox[146] = 0x4f;sbox[147] = 0xdc;sbox[148] = 0x22;sbox[149] = 0x2a;sbox[150] = 0x90;sbox[151] = 0x88;sbox[152] = 0x46;sbox[153] = 0xee;sbox[154] = 0xb8;sbox[155] = 0x14;sbox[156] = 0xde;sbox[157] = 0x5e;sbox[158] = 0xb;sbox[159] = 0xdb;
    sbox[160] = 0xe0;sbox[161] = 0x32;sbox[162] = 0x3a;sbox[163] = 0xa;sbox[164] = 0x49;sbox[165] = 0x6;sbox[166] = 0x24;sbox[167] = 0x5c;sbox[168] = 0xc2;sbox[169] = 0xd3;sbox[170] = 0xac;sbox[171] = 0x62;sbox[172] = 0x91;sbox[173] = 0x95;sbox[174] = 0xe4;sbox[175] = 0x79;
    sbox[176] = 0xe7;sbox[177] = 0xc8;sbox[178] = 0x37;sbox[179] = 0x6d;sbox[180] = 0x8d;sbox[181] = 0xd5;sbox[182] = 0x4e;sbox[183] = 0xa9;sbox[184] = 0x6c;sbox[185] = 0x56;sbox[186] = 0xf4;sbox[187] = 0xea;sbox[188] = 0x65;sbox[189] = 0x7a;sbox[190] = 0xae;sbox[191] = 0x8;
    sbox[192] = 0xba;sbox[193] = 0x78;sbox[194] = 0x25;sbox[195] = 0x2e;sbox[196] = 0x1c;sbox[197] = 0xa6;sbox[198] = 0xb4;sbox[199] = 0xc6;sbox[200] = 0xe8;sbox[201] = 0xdd;sbox[202] = 0x74;sbox[203] = 0x1f;sbox[204] = 0x4b;sbox[205] = 0xbd;sbox[206] = 0x8b;sbox[207] = 0x8a;
    sbox[208] = 0x70;sbox[209] = 0x3e;sbox[210] = 0xb5;sbox[211] = 0x66;sbox[212] = 0x48;sbox[213] = 0x3;sbox[214] = 0xf6;sbox[215] = 0xe;sbox[216] = 0x61;sbox[217] = 0x35;sbox[218] = 0x57;sbox[219] = 0xb9;sbox[220] = 0x86;sbox[221] = 0xc1;sbox[222] = 0x1d;sbox[223] = 0x9e;
    sbox[224] = 0xe1;sbox[225] = 0xf8;sbox[226] = 0x98;sbox[227] = 0x11;sbox[228] = 0x69;sbox[229] = 0xd9;sbox[230] = 0x8e;sbox[231] = 0x94;sbox[232] = 0x9b;sbox[233] = 0x1e;sbox[234] = 0x87;sbox[235] = 0xe9;sbox[236] = 0xce;sbox[237] = 0x55;sbox[238] = 0x28;sbox[239] = 0xdf;
    sbox[240] = 0x8c;sbox[241] = 0xa1;sbox[242] = 0x89;sbox[243] = 0xd;sbox[244] = 0xbf;sbox[245] = 0xe6;sbox[246] = 0x42;sbox[247] = 0x68;sbox[248] = 0x41;sbox[249] = 0x99;sbox[250] = 0x2d;sbox[251] = 0xf;sbox[252] = 0xb0;sbox[253] = 0x54;sbox[254] = 0xbb; sbox[255] = 0x16;

    for (int i = 0; i < MAX; i++)
        sbox_inverse[sbox[i]] = i;
    
    for (int i = 0; i < LENGTH; i++)
        shift_row_tab_inverse[shift_row_tab[i]] = i;
    
    for (int i = 0; i < 128; i++)
    {
        time[i] = i << 1;
        time[128 + i] = (i << 1) ^ 0x1b;
    }
}

__device__ void aes_init_inverse(BYTE sbox[], BYTE shift_row_tab[], BYTE sbox_inverse[], BYTE time[], BYTE shift_row_tab_inverse[])
{
    shift_row_tab[0]=0;
    shift_row_tab[1]=5;
    shift_row_tab[2]=10;
    shift_row_tab[3]=15;
    shift_row_tab[4]=4;
    shift_row_tab[5]=9;
    shift_row_tab[6]=14;
    shift_row_tab[7]=3;
    shift_row_tab[8]=8;
    shift_row_tab[9]=13;
    shift_row_tab[10]=2;
    shift_row_tab[11]=7;
    shift_row_tab[12]=12;
    shift_row_tab[13]=1;
    shift_row_tab[14]=6;
    shift_row_tab[15]=11;

    sbox_inverse[0] = 0x52;sbox_inverse[1] = 0x9;sbox_inverse[2] = 0x6a;sbox_inverse[3] = 0xd5;sbox_inverse[4] = 0x30;sbox_inverse[5] = 0x36;sbox_inverse[6] = 0xa5;sbox_inverse[7] = 0x38;sbox_inverse[8] = 0xbf;sbox_inverse[9] = 0x40;sbox_inverse[10] = 0xa3;sbox_inverse[11] = 0x9e;sbox_inverse[12] = 0x81;sbox_inverse[13] = 0xf3;sbox_inverse[14] = 0xd7;sbox_inverse[15] = 0xfb;
    sbox_inverse[16] = 0x7c;sbox_inverse[17] = 0xe3;sbox_inverse[18] = 0x39;sbox_inverse[19] = 0x82;sbox_inverse[20] = 0x9b;sbox_inverse[21] = 0x2f;sbox_inverse[22] = 0xff;sbox_inverse[23] = 0x87;sbox_inverse[24] = 0x34;sbox_inverse[25] = 0x8e;sbox_inverse[26] = 0x43;sbox_inverse[27] = 0x44;sbox_inverse[28] = 0xc4;sbox_inverse[29] = 0xde;sbox_inverse[30] = 0xe9;sbox_inverse[31] = 0xcb;
    sbox_inverse[32] = 0x54;sbox_inverse[33] = 0x7b;sbox_inverse[34] = 0x94;sbox_inverse[35] = 0x32;sbox_inverse[36] = 0xa6;sbox_inverse[37] = 0xc2;sbox_inverse[38] = 0x23;sbox_inverse[39] = 0x3d;sbox_inverse[40] = 0xee;sbox_inverse[41] = 0x4c;sbox_inverse[42] = 0x95;sbox_inverse[43] = 0xb;sbox_inverse[44] = 0x42;sbox_inverse[45] = 0xfa;sbox_inverse[46] = 0xc3;sbox_inverse[47] = 0x4e;
    sbox_inverse[48] = 0x8;sbox_inverse[49] = 0x2e;sbox_inverse[50] = 0xa1;sbox_inverse[51] = 0x66;sbox_inverse[52] = 0x28;sbox_inverse[53] = 0xd9;sbox_inverse[54] = 0x24;sbox_inverse[55] = 0xb2;sbox_inverse[56] = 0x76;sbox_inverse[57] = 0x5b;sbox_inverse[58] = 0xa2;sbox_inverse[59] = 0x49;sbox_inverse[60] = 0x6d;sbox_inverse[61] = 0x8b;sbox_inverse[62] = 0xd1;sbox_inverse[63] = 0x25;
    sbox_inverse[64] = 0x72;sbox_inverse[65] = 0xf8;sbox_inverse[66] = 0xf6;sbox_inverse[67] = 0x64;sbox_inverse[68] = 0x86;sbox_inverse[69] = 0x68;sbox_inverse[70] = 0x98;sbox_inverse[71] = 0x16;sbox_inverse[72] = 0xd4;sbox_inverse[73] = 0xa4;sbox_inverse[74] = 0x5c;sbox_inverse[75] = 0xcc;sbox_inverse[76] = 0x5d;sbox_inverse[77] = 0x65;sbox_inverse[78] = 0xb6;sbox_inverse[79] = 0x92;
    sbox_inverse[80] = 0x6c;sbox_inverse[81] = 0x70;sbox_inverse[82] = 0x48;sbox_inverse[83] = 0x50;sbox_inverse[84] = 0xfd;sbox_inverse[85] = 0xed;sbox_inverse[86] = 0xb9;sbox_inverse[87] = 0xda;sbox_inverse[88] = 0x5e;sbox_inverse[89] = 0x15;sbox_inverse[90] = 0x46;sbox_inverse[91] = 0x57;sbox_inverse[92] = 0xa7;sbox_inverse[93] = 0x8d;sbox_inverse[94] = 0x9d;sbox_inverse[95] = 0x84;
    sbox_inverse[96] = 0x90;sbox_inverse[97] = 0xd8;sbox_inverse[98] = 0xab;sbox_inverse[99] = 0x0;sbox_inverse[100] = 0x8c;sbox_inverse[101] = 0xbc;sbox_inverse[102] = 0xd3;sbox_inverse[103] = 0xa;sbox_inverse[104] = 0xf7;sbox_inverse[105] = 0xe4;sbox_inverse[106] = 0x58;sbox_inverse[107] = 0x5;sbox_inverse[108] = 0xb8;sbox_inverse[109] = 0xb3;sbox_inverse[110] = 0x45;sbox_inverse[111] = 0x6;
    sbox_inverse[112] = 0xd0;sbox_inverse[113] = 0x2c;sbox_inverse[114] = 0x1e;sbox_inverse[115] = 0x8f;sbox_inverse[116] = 0xca;sbox_inverse[117] = 0x3f;sbox_inverse[118] = 0xf;sbox_inverse[119] = 0x2;sbox_inverse[120] = 0xc1;sbox_inverse[121] = 0xaf;sbox_inverse[122] = 0xbd;sbox_inverse[123] = 0x3;sbox_inverse[124] = 0x1;sbox_inverse[125] = 0x13;sbox_inverse[126] = 0x8a;sbox_inverse[127] = 0x6b;
    sbox_inverse[128] = 0x3a;sbox_inverse[129] = 0x91;sbox_inverse[130] = 0x11;sbox_inverse[131] = 0x41;sbox_inverse[132] = 0x4f;sbox_inverse[133] = 0x67;sbox_inverse[134] = 0xdc;sbox_inverse[135] = 0xea;sbox_inverse[136] = 0x97;sbox_inverse[137] = 0xf2;sbox_inverse[138] = 0xcf;sbox_inverse[139] = 0xce;sbox_inverse[140] = 0xf0;sbox_inverse[141] = 0xb4;sbox_inverse[142] = 0xe6;sbox_inverse[143] = 0x73;
    sbox_inverse[144] = 0x96;sbox_inverse[145] = 0xac;sbox_inverse[146] = 0x74;sbox_inverse[147] = 0x22;sbox_inverse[148] = 0xe7;sbox_inverse[149] = 0xad;sbox_inverse[150] = 0x35;sbox_inverse[151] = 0x85;sbox_inverse[152] = 0xe2;sbox_inverse[153] = 0xf9;sbox_inverse[154] = 0x37;sbox_inverse[155] = 0xe8;sbox_inverse[156] = 0x1c;sbox_inverse[157] = 0x75;sbox_inverse[158] = 0xdf;sbox_inverse[159] = 0x6e;
    sbox_inverse[160] = 0x47;sbox_inverse[161] = 0xf1;sbox_inverse[162] = 0x1a;sbox_inverse[163] = 0x71;sbox_inverse[164] = 0x1d;sbox_inverse[165] = 0x29;sbox_inverse[166] = 0xc5;sbox_inverse[167] = 0x89;sbox_inverse[168] = 0x6f;sbox_inverse[169] = 0xb7;sbox_inverse[170] = 0x62;sbox_inverse[171] = 0xe;sbox_inverse[172] = 0xaa;sbox_inverse[173] = 0x18;sbox_inverse[174] = 0xbe;sbox_inverse[175] = 0x1b;
    sbox_inverse[176] = 0xfc;sbox_inverse[177] = 0x56;sbox_inverse[178] = 0x3e;sbox_inverse[179] = 0x4b;sbox_inverse[180] = 0xc6;sbox_inverse[181] = 0xd2;sbox_inverse[182] = 0x79;sbox_inverse[183] = 0x20;sbox_inverse[184] = 0x9a;sbox_inverse[185] = 0xdb;sbox_inverse[186] = 0xc0;sbox_inverse[187] = 0xfe;sbox_inverse[188] = 0x78;sbox_inverse[189] = 0xcd;sbox_inverse[190] = 0x5a;sbox_inverse[191] = 0xf4;
    sbox_inverse[192] = 0x1f;sbox_inverse[193] = 0xdd;sbox_inverse[194] = 0xa8;sbox_inverse[195] = 0x33;sbox_inverse[196] = 0x88;sbox_inverse[197] = 0x7;sbox_inverse[198] = 0xc7;sbox_inverse[199] = 0x31;sbox_inverse[200] = 0xb1;sbox_inverse[201] = 0x12;sbox_inverse[202] = 0x10;sbox_inverse[203] = 0x59;sbox_inverse[204] = 0x27;sbox_inverse[205] = 0x80;sbox_inverse[206] = 0xec;sbox_inverse[207] = 0x5f;
    sbox_inverse[208] = 0x60;sbox_inverse[209] = 0x51;sbox_inverse[210] = 0x7f;sbox_inverse[211] = 0xa9;sbox_inverse[212] = 0x19;sbox_inverse[213] = 0xb5;sbox_inverse[214] = 0x4a;sbox_inverse[215] = 0xd;sbox_inverse[216] = 0x2d;sbox_inverse[217] = 0xe5;sbox_inverse[218] = 0x7a;sbox_inverse[219] = 0x9f;sbox_inverse[220] = 0x93;sbox_inverse[221] = 0xc9;sbox_inverse[222] = 0x9c;sbox_inverse[223] = 0xef;
    sbox_inverse[224] = 0xa0;sbox_inverse[225] = 0xe0;sbox_inverse[226] = 0x3b;sbox_inverse[227] = 0x4d;sbox_inverse[228] = 0xae;sbox_inverse[229] = 0x2a;sbox_inverse[230] = 0xf5;sbox_inverse[231] = 0xb0;sbox_inverse[232] = 0xc8;sbox_inverse[233] = 0xeb;sbox_inverse[234] = 0xbb;sbox_inverse[235] = 0x3c;sbox_inverse[236] = 0x83;sbox_inverse[237] = 0x53;sbox_inverse[238] = 0x99;sbox_inverse[239] = 0x61;
    sbox_inverse[240] = 0x17;sbox_inverse[241] = 0x2b;sbox_inverse[242] = 0x4;sbox_inverse[243] = 0x7e;sbox_inverse[244] = 0xba;sbox_inverse[245] = 0x77;sbox_inverse[246] = 0xd6;sbox_inverse[247] = 0x26;sbox_inverse[248] = 0xe1;sbox_inverse[249] = 0x69;sbox_inverse[250] = 0x14;sbox_inverse[251] = 0x63;sbox_inverse[252] = 0x55;sbox_inverse[253] = 0x21;sbox_inverse[254] = 0xc;sbox_inverse[255] = 0x7d;

    for (int i = 0; i < LENGTH; i++)
        shift_row_tab_inverse[shift_row_tab[i]] = i;
    
    for (int i = 0; i < 128; i++)
    {
        time[i] = i << 1;
        time[128 + i] = (i << 1) ^ 0x1b;
    }
}

__global__ void aes_encrypt(aes_block aes_block_array[], BYTE key[], int key_length, int block_number)
{
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ BYTE shift_row_tab[LENGTH];
    __shared__ BYTE shift_row_tab_inverse[LENGTH];
    __shared__ BYTE sbox[MAX];
    __shared__ BYTE sbox_inverse[MAX];
    __shared__ BYTE time[MAX];

    if (global_thread_index < block_number)
    {
        if (threadIdx.x == 0)
            aes_init(sbox, shift_row_tab, sbox_inverse, time, shift_row_tab_inverse);
        
        __syncthreads();

        BYTE block[LENGTH];
        for (int i = 0; i < LENGTH; i++)
            block[i] = aes_block_array[global_thread_index].block[i];

        int length = key_length;
        int i;

        add_round_key(block, &key[0]);
        for (i = LENGTH; i < length - LENGTH; i = i + LENGTH)
        {
            sub_bytes(block, sbox);
            shift_rows(block, shift_row_tab);
            mix_columns(block, time);
            add_round_key(block, &key[i]);
        }
        sub_bytes(block, sbox);
        shift_rows(block, shift_row_tab);
        add_round_key(block, &key[i]);

        for (int j = 0; j < LENGTH; j++)
            aes_block_array[global_thread_index].block[i] = block[i];
    }
}

__global__ void aes_decrypt(aes_block aes_block_array[], BYTE key[], int key_length, int block_number)
{
    int global_thread_index = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ BYTE shift_row_tab[LENGTH];
    __shared__ BYTE shift_row_tab_inverse[LENGTH];
    __shared__ BYTE sbox[MAX];
    __shared__ BYTE sbox_inverse[MAX];
    __shared__ BYTE time[MAX];

    if (global_thread_index < block_number)
    {
        if (threadIdx.x == 0)
            aes_init_inverse(sbox, shift_row_tab, sbox_inverse, time, shift_row_tab_inverse);
        
        __syncthreads();

        BYTE block[LENGTH];
        for (int i = 0; i < LENGTH; i++)
            block[i] = aes_block_array[global_thread_index].block[i];

        int length = key_length;	
        int i;

        add_round_key(block, &key[length - LENGTH]);
        shift_rows(block, shift_row_tab_inverse);
        sub_bytes(block, sbox_inverse);
        for (i = length - 32; i >= LENGTH; i = i - LENGTH)
        {
            add_round_key(block, &key[i]);
            mix_columns_inverse(block, time);
            shift_rows(block, shift_row_tab_inverse);
            sub_bytes(block, sbox_inverse);
        }
        add_round_key(block, &key[0]);

        for (int j = 0; j < LENGTH; j++)
            aes_block_array[global_thread_index].block[i] = block[i];
    }
}

int expand_key(BYTE key[], int key_length)
{
	int length = key_length;
	int Rcon = 1;
	int ks;

	BYTE temp_array_1[4], temp_array_2[4];

	switch(length)
	{
		case 16:
			ks = 16 * (10 + 1);
		break;

		case 24:
			ks = 16 * (12 + 1);
		break;

		case 32:
			ks = 16 * (14 + 1);
		break;

		default:
			printf("Expand Key : Key Lengths Permitted Are -> 16, 24 or 32 Bytes Only !!!\n");
		break;
	}

	for (int i = length; i < ks; i = i + 4)
	{
		memcpy(temp_array_1, &key[i - 4], 4);

		if (i % length == 0)
		{
			temp_array_2[0] = sbox[temp_array_1[1]] ^ Rcon;
			temp_array_2[1] = sbox[temp_array_1[2]];
			temp_array_2[2] = sbox[temp_array_1[3]];
			temp_array_2[3] = sbox[temp_array_1[0]];

			memcpy(temp_array_1, temp_array_2, 4);

			if ((Rcon <<= 1) >= 256)
				Rcon = Rcon ^ 0x11b;
		}

		else if ((length > 24) && (i % length == 16))
		{
			temp_array_2[0] = sbox[temp_array_1[1]];
			temp_array_2[1] = sbox[temp_array_1[2]];
			temp_array_2[2] = sbox[temp_array_1[3]];
			temp_array_2[3] = sbox[temp_array_1[0]];

			memcpy(temp_array_1, temp_array_2, 4);
		}

		for (int j = 0; j < 4; j++)
			key[i + j] = key[i + j - length] ^ temp_array_1[j];
		
	}

	return ks;
}

void print_file_data(BYTE array[], int length, FILE* fp, int file)
{
	// Local Variables
	int flag = 0;

	// Code
	switch(file)
	{
		case 1:
			for (int i = 0; i < length; i++)
				fprintf(fp, "%02x", array[i]);
			fprintf(fp, "\n");
		break;

		case 2:
			for (int i = 0; i < length; i++)
			{
				fprintf(fp, "%c", array[i]);
				if (array[i] == '\n')
					flag++;
			}
		break;

		case 3:
			for (int i = 0; i < length; i++)
			{
				if (array[i] == '\0')
					return;
				fprintf(fp, "%c", array[i]);
				if (array[i] == '\n')
					flag++;
			}
		break;
	}
}

int get_file_data(char* argv[])
{
    // Function Declaration
    void cleanup(void);

    // Code
	input_file_stream.open(argv[1], ifstream::binary);
	if (!input_file_stream)
		return FAILURE;
	input_file_stream.seekg(0, ios::end);
	input_file_length = input_file_stream.tellg();
	input_file_stream.seekg(0, ios::beg);

	block_number = input_file_length / LENGTH;
	num_zero_pending = input_file_length % LENGTH;

	key_file_stream.open(argv[2]);
	while (key_file_stream.peek() != EOF)
	{
		key_file_stream >> key[key_length];
		if (key_file_stream.eof())
			break;
		key_length++;
	}

	switch(key_length)
	{
		case 16:
		case 24:
		case 32:
		break;

		default:
			printf("Key Length Should Be 128, 192 or 256 bits !!!\n");
			return FAILURE;
	}

	expanded_key_length = expand_key(key, key_length);

	if (num_zero_pending != 0)
		aes_block_array = new aes_block[block_number + 1];
	else
		aes_block_array = new aes_block[block_number];
	
	encFile = fopen(argv[3], "wb");
	decFile = fopen(argv[4], "wb");

	for (int i = 0; i < block_number; i++)
	{
		input_file_stream.read(temp, LENGTH);
		for (int j = 0; j < LENGTH; j++)
			aes_block_array[i].block[j] = (unsigned char)temp[j];
	}

	if (num_zero_pending != 0)
	{
		input_file_stream.read(temp, num_zero_pending);
		for (int j = 0; j < LENGTH; j++)
			aes_block_array[block_number].block[j] = (unsigned char)temp[j];
		for (int j = 1; j < LENGTH - num_zero_pending; j++)
			aes_block_array[block_number].block[LENGTH - j] = '\0';
	
		block_number++;
	}

	char* num_threads = argv[5];
	int number_of_threads = atoi(num_threads);

    dim3 ThreadsPerBlock(number_of_threads);
    dim3 BlocksPerGrid(256);

    result = cudaMalloc((void **)&cuda_aes_block_array, block_number * sizeof(aes_block));
    if (result != cudaSuccess)
	{
		printf("\nDevice Memory Allocation Failed For cuda_aes_block_array ... Exiting Now !!!\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

    result = cudaMalloc((void **)&cuda_key, LENGTH * 15 * sizeof(BYTE));
    if (result != cudaSuccess)
	{
		printf("\nDevice Memory Allocation Failed For cuda_key ... Exiting Now !!!\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

    result = cudaMemcpy(cuda_aes_block_array, aes_block_array, block_number * sizeof(aes_block), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
	{
		printf("\nHost To Device Data Copy Failed For cuda_aes_block_array ... Exiting Now !!!\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

    result = cudaMemcpy(cuda_key, key, LENGTH * 15 * sizeof(BYTE), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
	{
		printf("\nHost To Device Data Copy Failed For cuda_key ... Exiting Now !!!\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

    StopWatchInterface* timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

    aes_encrypt <<< BlocksPerGrid, ThreadsPerBlock >>>(cuda_aes_block_array, cuda_key, expanded_key_length, block_number);

    sdkStopTimer(&timer);
	float timeToEncrypt = sdkGetTimerValue(&timer);
	printf("\nTime To Encrypt Image on %s = %.6f seconds\n", dev_prop.name, timeToEncrypt);
	sdkDeleteTimer(&timer);
	timer = NULL;

    result = cudaMemcpy(aes_block_array, cuda_aes_block_array, block_number * sizeof(aes_block), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
	{
		printf("\nDevice To Host Data Copy Failed For cuda_aes_block_array ... Exiting Now !!!\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

    for (int i = 0; i < block_number - 1; i++)
        print_file_data(aes_block_array[i].block, block_length, encFile, 1);
    print_file_data(aes_block_array[block_number - 1].block, block_length, encFile, 1);

	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

    aes_decrypt <<< BlocksPerGrid, ThreadsPerBlock >>>(cuda_aes_block_array, cuda_key, expanded_key_length, block_number);

    sdkStopTimer(&timer);
	float timeToDecrypt = sdkGetTimerValue(&timer);
	printf("\nTime To Decrypt Image on %s = %.6f seconds\n", dev_prop.name, timeToDecrypt);
	sdkDeleteTimer(&timer);
	timer = NULL;
    
    result = cudaMemcpy(aes_block_array, cuda_aes_block_array, block_number * sizeof(aes_block), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
	{
		printf("\nDevice To Host Data Copy Failed For cuda_aes_block_array ... Exiting Now !!!\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

    for (int i = 0; i < block_number - 1; i++)
        print_file_data(aes_block_array[i].block, block_length, decFile, 2);

    if (num_zero_pending == 0)
        print_file_data(aes_block_array[block_number - 1].block, block_length, decFile, 2);
    else
        print_file_data(aes_block_array[block_number - 1].block, block_length, decFile, 3);

	return SUCCESS;
}

void cleanup(void)
{
    if (cuda_key)
    {
        cudaFree(cuda_key);
        cuda_key = NULL;
    }

    if (cuda_aes_block_array)
    {
        cudaFree(cuda_aes_block_array);
        cuda_aes_block_array = NULL;
    }

	if (aes_block_array)
	{
		free(aes_block_array);
		aes_block_array = NULL;
	}

    if (decFile)
	{
		fclose(decFile);
		decFile = NULL;
	}

	if (encFile)
	{
		fclose(encFile);
		encFile = NULL;
	}

	if (key_file_stream.is_open())
		key_file_stream.close();

	if (input_file_stream.is_open())
		input_file_stream.close();
}


int main(int argc, char* argv[])
{
	// Variable Declarations
	int result;

	// Code
	print_cuda_device_properties();

	result = get_file_data(argv);
	if (result == FAILURE)
	{
		printf("Failed To Access Input Files ... Exiting Now !!!\n");
		exit(EXIT_FAILURE);
	}

	cleanup();

	return 0;
}
