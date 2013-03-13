#ifndef kernel_h_
#define kernel_h_
//#define WIN32 // wtf?

#define MAX_PW 10
#define MAX_PERMS_PER_KERNEL 50000000
//#define GETW_OPT


typedef struct
{
  unsigned char d[16];
} md5Digest;

typedef struct
{
  // w - 64 byte string 
  // state - stores the 4 ints a,b,c,d
  char w[64];
  unsigned int state[4];
  // this is 80 bytes, make it (32*3) + 1 for bank conflicts. assume 32 banks
  //char padding[17];
} md5Node;


typedef struct
{
  //unsigned char digest[16];
  md5Digest digest;
  char plaintext[MAX_PW];
} md5Plain;




typedef struct
{
  unsigned char d[16];
} md5DigestCUDA;

//typedef unsigned char md5Digest[16];

#define HEXMD5PATTERN "%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x"
#define HEXMD5(digest) \
  digest[0], \
  digest[1], \
  digest[2], \
  digest[3], \
  digest[4], \
  digest[5], \
  digest[6], \
  digest[7], \
  digest[8], \
  digest[9], \
  digest[10], \
  digest[11], \
  digest[12], \
  digest[13], \
  digest[14], \
  digest[15]

#define HEXMD5SCAN(digest) \
  &digest[0], \
  &digest[1], \
  &digest[2], \
  &digest[3], \
  &digest[4], \
  &digest[5], \
  &digest[6], \
  &digest[7], \
  &digest[8], \
  &digest[9], \
  &digest[10], \
  &digest[11], \
  &digest[12], \
  &digest[13], \
  &digest[14], \
  &digest[15]

extern "C"{
	//cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

  //void MD5StringCuda_pre(int shared_mem_block, char* charset_h, char* charset_d, md5Plain* returnMd5s_h, md5Plain* returnMd5s_d);
  void MD5StringCuda_pre(int shared_mem_block, char* &charset_h, char* &charset_d, md5Plain* &returnMd5s_h, md5Plain* &returnMd5s_d, int* &perm_init_index_d);
  bool MD5StringCuda_kernel(dim3 dimGrid, dim3 dimBlock, int shared_mem_block, int *perm_init_index_h, int *perm_init_index_d, char *charset_d, int charset_len, int pw_length, md5Plain *returnMd5s_h, md5Plain *returnMd5s_d);
	//void MD5StringCuda(dim3 dimGrid, dim3 dimBlock, char *charset, md5Digest *all_digests_h, int);

  //void cpuCheckMD5Intersection(md5Plain* returnMd5s_h, int charset_len, int pw_length, md5Digest *all_digests_h, int digests_length);
}

#endif //kernel_h_