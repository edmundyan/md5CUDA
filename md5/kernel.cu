//#include "sys/time.h"
#define WIN32 // SO STUPID WTF

#include <iostream>
#include <functional>
#include <string>

#include "time.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>
#include <time.h>

#include "kernel.h"
#include "utils_cpu.h"


/* F, G and H are basic MD5 functions: selection, majority, parity */
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z))) 

/* ROTATE_LEFT rotates x left n bits */
#define ROTATE_LEFT(x, n) (((x) << (n)) | ((x) >> (32-(n))))

/* FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4 */
/* Rotation is separate from addition to prevent recomputation */
#define FF(a, b, c, d, x, s, ac) \
  {(a) += F ((b), (c), (d)) + (x) + (unsigned int)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define GG(a, b, c, d, x, s, ac) \
  {(a) += G ((b), (c), (d)) + (x) + (unsigned int)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define HH(a, b, c, d, x, s, ac) \
  {(a) += H ((b), (c), (d)) + (x) + (unsigned int)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }
#define II(a, b, c, d, x, s, ac) \
  {(a) += I ((b), (c), (d)) + (x) + (unsigned int)(ac); \
   (a) = ROTATE_LEFT ((a), (s)); \
   (a) += (b); \
  }

// Accessor for w[16] array. Naively, this would just be w[i]; however, this
// choice leads to worst-case-scenario access pattern wrt. shared memory
// bank conflicts, as the same indices in different threads fall into the
// same bank (as the words are 16 unsigned ints long). The packing below causes the
// same indices in different threads of a warp to map to different banks. In
// testing this gave a ~40% speedup.
//
// PS: An alternative solution would be to make the w array 17 unsigned ints long
// (thus wasting a little shared memory)
//
__device__ inline unsigned int &getw(unsigned int *w, const int i)
{
	return w[(i+threadIdx.x) % 16];
}

__device__ inline unsigned int getw(const unsigned int *w, const int i)	// const- version
{
	return w[(i+threadIdx.x) % 16];
}



void inline __device__ GPUshufflegetw(unsigned int* in)
{
  unsigned int tmp[16];

  for(int i = 0; i < 16; i++)
    tmp[i] = in[i];

  for(int i = 0; i < 16; i++)
    getw(in, i) = tmp[i];
}

/* Basic MD5 step. Transform buf based on in.
 */
void inline __device__ md5_v2(const unsigned int *in, unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d)
{
  #define S11 7
  #define S12 12
  #define S13 17
  #define S14 22
  #define S21 5
  #define S22 9
  #define S23 14
  #define S24 20
  #define S31 4
  #define S32 11
  #define S33 16
  #define S34 23
  #define S41 6
  #define S42 10
  #define S43 15
  #define S44 21

	const unsigned int a0 = 0x67452301;
	const unsigned int b0 = 0xEFCDAB89;
	const unsigned int c0 = 0x98BADCFE;
	const unsigned int d0 = 0x10325476;

	//Initialize hash value for this chunk:
	a = a0;
	b = b0;
	c = c0;
	d = d0;
 
  /* Round 1 */
#ifdef GETW_OPT
  FF ( a, b, c, d, getw(in,  0), S11, 3614090360); /* 1 */
  FF ( d, a, b, c, getw(in,  1), S12, 3905402710); /* 2 */
  FF ( c, d, a, b, getw(in,  2), S13,  606105819); /* 3 */
  FF ( b, c, d, a, getw(in,  3), S14, 3250441966); /* 4 */
  FF ( a, b, c, d, getw(in,  4), S11, 4118548399); /* 5 */
  FF ( d, a, b, c, getw(in,  5), S12, 1200080426); /* 6 */
  FF ( c, d, a, b, getw(in,  6), S13, 2821735955); /* 7 */
  FF ( b, c, d, a, getw(in,  7), S14, 4249261313); /* 8 */
  FF ( a, b, c, d, getw(in,  8), S11, 1770035416); /* 9 */
  FF ( d, a, b, c, getw(in,  9), S12, 2336552879); /* 10 */
  FF ( c, d, a, b, getw(in, 10), S13, 4294925233); /* 11 */
  FF ( b, c, d, a, getw(in, 11), S14, 2304563134); /* 12 */
  FF ( a, b, c, d, getw(in, 12), S11, 1804603682); /* 13 */
  FF ( d, a, b, c, getw(in, 13), S12, 4254626195); /* 14 */
  FF ( c, d, a, b, getw(in, 14), S13, 2792965006); /* 15 */
  FF ( b, c, d, a, getw(in, 15), S14, 1236535329); /* 16 */
#else
  FF (a, b, c, d, in[ 0], S11, 0xd76aa478); /* 1 */
  FF (d, a, b, c, in[ 1], S12, 0xe8c7b756); /* 2 */
  FF (c, d, a, b, in[ 2], S13, 0x242070db); /* 3 */
  FF (b, c, d, a, in[ 3], S14, 0xc1bdceee); /* 4 */
  FF (a, b, c, d, in[ 4], S11, 0xf57c0faf); /* 5 */
  FF (d, a, b, c, in[ 5], S12, 0x4787c62a); /* 6 */
  FF (c, d, a, b, in[ 6], S13, 0xa8304613); /* 7 */
  FF (b, c, d, a, in[ 7], S14, 0xfd469501); /* 8 */
  FF (a, b, c, d, in[ 8], S11, 0x698098d8); /* 9 */
  FF (d, a, b, c, in[ 9], S12, 0x8b44f7af); /* 10 */
  FF (c, d, a, b, in[10], S13, 0xffff5bb1); /* 11 */
  FF (b, c, d, a, in[11], S14, 0x895cd7be); /* 12 */
  FF (a, b, c, d, in[12], S11, 0x6b901122); /* 13 */
  FF (d, a, b, c, in[13], S12, 0xfd987193); /* 14 */
  FF (c, d, a, b, in[14], S13, 0xa679438e); /* 15 */
  FF (b, c, d, a, in[15], S14, 0x49b40821); /* 16 */
#endif

 /* Round 2 */
#ifdef GETW_OPT
  GG ( a, b, c, d, getw(in,  1), S21, 4129170786); /* 17 */
  GG ( d, a, b, c, getw(in,  6), S22, 3225465664); /* 18 */
  GG ( c, d, a, b, getw(in, 11), S23,  643717713); /* 19 */
  GG ( b, c, d, a, getw(in,  0), S24, 3921069994); /* 20 */
  GG ( a, b, c, d, getw(in,  5), S21, 3593408605); /* 21 */
  GG ( d, a, b, c, getw(in, 10), S22,   38016083); /* 22 */
  GG ( c, d, a, b, getw(in, 15), S23, 3634488961); /* 23 */
  GG ( b, c, d, a, getw(in,  4), S24, 3889429448); /* 24 */
  GG ( a, b, c, d, getw(in,  9), S21,  568446438); /* 25 */
  GG ( d, a, b, c, getw(in, 14), S22, 3275163606); /* 26 */
  GG ( c, d, a, b, getw(in,  3), S23, 4107603335); /* 27 */
  GG ( b, c, d, a, getw(in,  8), S24, 1163531501); /* 28 */
  GG ( a, b, c, d, getw(in, 13), S21, 2850285829); /* 29 */
  GG ( d, a, b, c, getw(in,  2), S22, 4243563512); /* 30 */
  GG ( c, d, a, b, getw(in,  7), S23, 1735328473); /* 31 */
  GG ( b, c, d, a, getw(in, 12), S24, 2368359562); /* 32 */
#else
  GG (a, b, c, d, in[ 1], S21, 0xf61e2562); /* 17 */
  GG (d, a, b, c, in[ 6], S22, 0xc040b340); /* 18 */
  GG (c, d, a, b, in[11], S23, 0x265e5a51); /* 19 */
  GG (b, c, d, a, in[ 0], S24, 0xe9b6c7aa); /* 20 */
  GG (a, b, c, d, in[ 5], S21, 0xd62f105d); /* 21 */
  GG (d, a, b, c, in[10], S22,  0x2441453); /* 22 */
  GG (c, d, a, b, in[15], S23, 0xd8a1e681); /* 23 */
  GG (b, c, d, a, in[ 4], S24, 0xe7d3fbc8); /* 24 */
  GG (a, b, c, d, in[ 9], S21, 0x21e1cde6); /* 25 */
  GG (d, a, b, c, in[14], S22, 0xc33707d6); /* 26 */
  GG (c, d, a, b, in[ 3], S23, 0xf4d50d87); /* 27 */
  GG (b, c, d, a, in[ 8], S24, 0x455a14ed); /* 28 */
  GG (a, b, c, d, in[13], S21, 0xa9e3e905); /* 29 */
  GG (d, a, b, c, in[ 2], S22, 0xfcefa3f8); /* 30 */
  GG (c, d, a, b, in[ 7], S23, 0x676f02d9); /* 31 */
  GG (b, c, d, a, in[12], S24, 0x8d2a4c8a); /* 32 */
#endif

  /* Round 3 */
#ifdef GETW_OPT
  HH ( a, b, c, d, getw(in,  5), S31, 4294588738); /* 33 */
  HH ( d, a, b, c, getw(in,  8), S32, 2272392833); /* 34 */
  HH ( c, d, a, b, getw(in, 11), S33, 1839030562); /* 35 */
  HH ( b, c, d, a, getw(in, 14), S34, 4259657740); /* 36 */
  HH ( a, b, c, d, getw(in,  1), S31, 2763975236); /* 37 */
  HH ( d, a, b, c, getw(in,  4), S32, 1272893353); /* 38 */
  HH ( c, d, a, b, getw(in,  7), S33, 4139469664); /* 39 */
  HH ( b, c, d, a, getw(in, 10), S34, 3200236656); /* 40 */
  HH ( a, b, c, d, getw(in, 13), S31,  681279174); /* 41 */
  HH ( d, a, b, c, getw(in,  0), S32, 3936430074); /* 42 */
  HH ( c, d, a, b, getw(in,  3), S33, 3572445317); /* 43 */
  HH ( b, c, d, a, getw(in,  6), S34,   76029189); /* 44 */
  HH ( a, b, c, d, getw(in,  9), S31, 3654602809); /* 45 */
  HH ( d, a, b, c, getw(in, 12), S32, 3873151461); /* 46 */
  HH ( c, d, a, b, getw(in, 15), S33,  530742520); /* 47 */
  HH ( b, c, d, a, getw(in,  2), S34, 3299628645); /* 48 */
#else
  HH (a, b, c, d, in[ 5], S31, 0xfffa3942); /* 33 */
  HH (d, a, b, c, in[ 8], S32, 0x8771f681); /* 34 */
  HH (c, d, a, b, in[11], S33, 0x6d9d6122); /* 35 */
  HH (b, c, d, a, in[14], S34, 0xfde5380c); /* 36 */
  HH (a, b, c, d, in[ 1], S31, 0xa4beea44); /* 37 */
  HH (d, a, b, c, in[ 4], S32, 0x4bdecfa9); /* 38 */
  HH (c, d, a, b, in[ 7], S33, 0xf6bb4b60); /* 39 */
  HH (b, c, d, a, in[10], S34, 0xbebfbc70); /* 40 */
  HH (a, b, c, d, in[13], S31, 0x289b7ec6); /* 41 */
  HH (d, a, b, c, in[ 0], S32, 0xeaa127fa); /* 42 */
  HH (c, d, a, b, in[ 3], S33, 0xd4ef3085); /* 43 */
  HH (b, c, d, a, in[ 6], S34,  0x4881d05); /* 44 */
  HH (a, b, c, d, in[ 9], S31, 0xd9d4d039); /* 45 */
  HH (d, a, b, c, in[12], S32, 0xe6db99e5); /* 46 */
  HH (c, d, a, b, in[15], S33, 0x1fa27cf8); /* 47 */
  HH (b, c, d, a, in[ 2], S34, 0xc4ac5665); /* 48 */
#endif

  /* Round 4 */
#ifdef GETW_OPT
  II ( a, b, c, d, getw(in,  0), S41, 4096336452); /* 49 */
  II ( d, a, b, c, getw(in,  7), S42, 1126891415); /* 50 */
  II ( c, d, a, b, getw(in, 14), S43, 2878612391); /* 51 */
  II ( b, c, d, a, getw(in,  5), S44, 4237533241); /* 52 */
  II ( a, b, c, d, getw(in, 12), S41, 1700485571); /* 53 */
  II ( d, a, b, c, getw(in,  3), S42, 2399980690); /* 54 */
  II ( c, d, a, b, getw(in, 10), S43, 4293915773); /* 55 */
  II ( b, c, d, a, getw(in,  1), S44, 2240044497); /* 56 */
  II ( a, b, c, d, getw(in,  8), S41, 1873313359); /* 57 */
  II ( d, a, b, c, getw(in, 15), S42, 4264355552); /* 58 */
  II ( c, d, a, b, getw(in,  6), S43, 2734768916); /* 59 */
  II ( b, c, d, a, getw(in, 13), S44, 1309151649); /* 60 */
  II ( a, b, c, d, getw(in,  4), S41, 4149444226); /* 61 */
  II ( d, a, b, c, getw(in, 11), S42, 3174756917); /* 62 */
  II ( c, d, a, b, getw(in,  2), S43,  718787259); /* 63 */
  II ( b, c, d, a, getw(in,  9), S44, 3951481745); /* 64 */
#else
  II (a, b, c, d, in[ 0], S41, 0xf4292244); /* 49 */
  II (d, a, b, c, in[ 7], S42, 0x432aff97); /* 50 */
  II (c, d, a, b, in[14], S43, 0xab9423a7); /* 51 */
  II (b, c, d, a, in[ 5], S44, 0xfc93a039); /* 52 */
  II (a, b, c, d, in[12], S41, 0x655b59c3); /* 53 */
  II (d, a, b, c, in[ 3], S42, 0x8f0ccc92); /* 54 */
  II (c, d, a, b, in[10], S43, 0xffeff47d); /* 55 */
  II (b, c, d, a, in[ 1], S44, 0x85845dd1); /* 56 */
  II (a, b, c, d, in[ 8], S41, 0x6fa87e4f); /* 57 */
  II (d, a, b, c, in[15], S42, 0xfe2ce6e0); /* 58 */
  II (c, d, a, b, in[ 6], S43, 0xa3014314); /* 59 */
  II (b, c, d, a, in[13], S44, 0x4e0811a1); /* 60 */
  II (a, b, c, d, in[ 4], S41, 0xf7537e82); /* 61 */
  II (d, a, b, c, in[11], S42, 0xbd3af235); /* 62 */
  II (c, d, a, b, in[ 2], S43, 0x2ad7d2bb); /* 63 */
  II (b, c, d, a, in[ 9], S44, 0xeb86d391); /* 64 */
#endif

	a += a0;
	b += b0;
	c += c0;
	d += d0;
}

#define BYTETOBINARYPATTERN "%d%d%d%d%d%d%d%d"
#define BYTETOBINARY(byte)  \
  (byte & 0x80 ? 1 : 0), \
  (byte & 0x40 ? 1 : 0), \
  (byte & 0x20 ? 1 : 0), \
  (byte & 0x10 ? 1 : 0), \
  (byte & 0x08 ? 1 : 0), \
  (byte & 0x04 ? 1 : 0), \
  (byte & 0x02 ? 1 : 0), \
  (byte & 0x01 ? 1 : 0) 

/* This code has an obvious bug and another non-obvious one :) */
__device__ inline void StoI(char *string, int length) {
  for(int i = 0; i < length; i++) {
    printf(BYTETOBINARYPATTERN" ", BYTETOBINARY(string[i]));
    if ((i+1) % 8 == 0)
      printf("\n");
  }
  printf("\n");
}


__device__ inline void Decode(unsigned int *output, unsigned char *input, unsigned int len) {
  unsigned int i, j;

  for (i = 0, j = 0; j < len; i++, j += 4) {
   output[i] = ((unsigned int)input[j]) | (((unsigned int)input[j+1]) << 8) |
     (((unsigned int)input[j+2]) << 16) | (((unsigned int)input[j+3]) << 24);
  }
}

__device__ inline void Encode(unsigned char *output, unsigned int *input, unsigned int len) {
  /* Encode()
   * converts a unsigned int[4] array into a uchar[16] array
  */
  unsigned int i, j;

  for (i = 0, j = 0; j < len; i++, j += 4) {
    output[j] = (unsigned char)(input[i] & 0xff);
    output[j+1] = (unsigned char)((input[i] >> 8) & 0xff);
    output[j+2] = (unsigned char)((input[i] >> 16) & 0xff);
    output[j+3] = (unsigned char)((input[i] >> 24) & 0xff);
  }
}



__device__ bool generatePermStarting(char *c0, int *starting, int pw_length, char* charset, int charset_len, int idx) {
  // modifies c0 to be a random string perm corresponding to idx in base(charset_len)
  int i;
  int idx_new;
  int idx_tmp;
  bool reset = true;

 for(i = 0; i < 64; i++) {
    if(i >= pw_length) {

      c0[i] = 0;

    } else {
      idx_tmp = idx+starting[i];
      if(idx_tmp < charset_len)
        reset = false;


      // old
      //c0[i] = charset[(idx+starting[i]) % charset_len];
      //idx = (idx+starting[i]) / charset_len;

      // faster version
      // trying to replace mods
      // A % B = A - B * (A/B)
      idx = idx_tmp / charset_len;

      c0[i] = charset[idx_tmp - charset_len * idx];   


    }
  }
  return reset;
}

__device__ void generatePerm(char *c0, int pw_length, char *charset, int charset_len, int idx) {
  // modifies c0 to be a random string perm corresponding to idx in base(charset_len)
	unsigned int len = 0;
	char *c = c0;
  int i;
  int idx_orig = idx;

  for(i = 0; i < 64; i++) {
    if(i >= pw_length) {
      c0[i] = 0;
    } else {
      // kinda dumb once we start doing 0%26; 0/26 over and over again.. but w/e
      // this is essentially a change of base algorithm
      // converts the idx, which is in base10, to base26 if we are you a-z charset. or base62 is a-zA-Z0-9 
      c0[i] = charset[idx % charset_len];
      idx = idx / charset_len;
    }
  }

  //printf("%d = '%s'\n", idx_orig, c0);
}
__device__ inline void md5_prep(char *c0, int pw_length) {
	//unsigned int len = 0;

  char *c = c0 + pw_length;

	//while(*c) {len++; c++;}
	c[0] = 0x80;			// bit 1 after the message


  // this doesn't look right in the bit representation, but maybe that's ok.. http://nsfsecurity.pr.erau.edu/crypto/md5.html might be wrong

  //if(pw_length == len)
  //  printf("YES");

  ((unsigned int*)c0)[14] = pw_length * 8;	// message length in bits

}


//__global__ void md5_kernel(char *charset_d, int charset_len, int pw_length, unsigned char *target_digest_d, int iteration, md5Digest *all_digests_d, int digests_length, md5Plain* returnMD5s)
__global__ void md5_kernel(int *perm_init_index_d, char *charset_d, int charset_len, int pw_length, md5Plain* returnMD5s)
{
  //extern __shared__ md5Node shared_mem[];
  extern __shared__ char shared_mem[];
  char* charset_shared = shared_mem;

  // init shared memory
  for (int i = threadIdx.x; i < charset_len; i += blockDim.x) { 
    if(i < charset_len)
      charset_shared[i] = charset_d[i];
  }
  __syncthreads();

 
  //md5Node *md5Node_arr = (md5Node *) (shared_mem + charset_len + (charset_len%4));
  md5Node *md5Node_arr = (md5Node *) (shared_mem + charset_len + (charset_len & 3));

#ifdef ENABLE_MEMORY
  unsigned char digest[16]; // todo delete?
  char w[64];
#endif
  int hashes_completed;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int threads_per_kernel = gridDim.x * blockDim.x;

  // if we have iterated through all permutations already
  bool reset = false;
  //int max_iters = pow((float)charset_len, (float)pw_length);
  //int starting_idx = iteration * MAX_PERMS_PER_KERNEL;


  //for(hashes_completed = starting_idx; hashes_completed < max_iters && (hashes_completed - starting_idx) < MAX_PERMS_PER_KERNEL; hashes_completed += threads_per_kernel) {
  //  // compute a new idx depending on which iteration of the loop we're on.
  //  int overall_idx = idx + hashes_completed;

  for(hashes_completed = idx; hashes_completed < MAX_PERMS_PER_KERNEL; hashes_completed += threads_per_kernel) {
    // compute a new idx depending on which iteration of the loop we're on.
    int overall_idx = hashes_completed;

    if(generatePermStarting(md5Node_arr[threadIdx.x].w, perm_init_index_d, pw_length, charset_shared, charset_len, overall_idx))
    {
      // returned RESET.  There are no more perms
      break; // for
    } else
    {
#ifdef ENABLE_MEMORY
      // we just got a new perm, calculate the md5 for it!
      // save the original string
      for(int i = 0; i < 64; i++) {
        w[i] = md5Node_arr[threadIdx.x].w[i];
      }
#endif

      
      // prepare the md5 string (adding the '1' and the length
	    md5_prep(md5Node_arr[threadIdx.x].w, pw_length);

      // need to do a shuffle for getw...
      GPUshufflegetw((unsigned int *) &(md5Node_arr[threadIdx.x].w));
      // calculate
      md5_v2((unsigned int *) &md5Node_arr[threadIdx.x].w[0], md5Node_arr[threadIdx.x].state[0], md5Node_arr[threadIdx.x].state[1], md5Node_arr[threadIdx.x].state[2], md5Node_arr[threadIdx.x].state[3]);

#ifdef ENABLE_MEMORY
      // encode the 4-wide unsigned int array to a char array
      Encode(digest, md5Node_arr[threadIdx.x].state, 16);
#endif

#ifdef ENABLE_MEMORY
      for(int i =0; i < 16; i++)
        returnMD5s[overall_idx].digest.d[i] = digest[i];
      for(int i =0; i < MAX_PW; i++)
        returnMD5s[overall_idx].plaintext[i] = w[i];
#endif
    }

    //if(overall_idx < max_iters && (overall_idx-starting_idx) < MAX_PERMS_PER_KERNEL) {
    //  // IF we are still a valid permutation AND we have not done more than the maximum number of perms in a single kernel call

    //  // convert w[] to a permutation
    //  generatePerm(md5Node_arr[threadIdx.x].w, pw_length, charset_shared, charset_len, overall_idx);

    //  // save the original string
    //  for(int i = 0; i < 64; i++) {
    //    w[i] = md5Node_arr[threadIdx.x].w[i];
    //  }
    //  
    //  //
    //  // prepare the md5 string (adding the '1' and the length
	   // md5_prep(md5Node_arr[threadIdx.x].w);

    //  // calculate
    //  md5_v2((unsigned int *) &md5Node_arr[threadIdx.x].w[0], md5Node_arr[threadIdx.x].state[0], md5Node_arr[threadIdx.x].state[1], md5Node_arr[threadIdx.x].state[2], md5Node_arr[threadIdx.x].state[3]);

    //  // encode the 4-wide unsigned int array to a char array
    //  Encode(digest, md5Node_arr[threadIdx.x].state, 16);


    //  for(int i =0; i < 16; i++)
    //    returnMD5s[overall_idx - starting_idx].digest.d[i] = digest[i];
    //  for(int i =0; i < 10; i++)
    //    returnMD5s[overall_idx - starting_idx].plaintext[i] = w[i];

      //if(compareMD5Digest(digest, target_digest_d)) {
      //  printf("\n%d="HEXMD5PATTERN"\n", overall_idx, HEXMD5(digest));
      //}
      //if(compareMD5DigestArray(digest, all_digests_d, digests_length)) {
      //  //printf("\n%s="HEXMD5PATTERN"\n", w, HEXMD5(digest));
      //}
    //}
  } // for
}



__global__ void md5_single(char *string_d, int strlen, unsigned char *target_digest_d)
{
  // run md5 on a single string
  md5Node single_md5Node;

  printf("wetrewrfew\n");
  for(int i = 0; i < 64; i++) {
    if(i < strlen)
      single_md5Node.w[i] = string_d[i];
    else
      single_md5Node.w[i] = 0;
  }
  printf("input=%s\n",single_md5Node.w); 
  // prepare the md5 string (adding the '1' and the length
  md5_prep(single_md5Node.w, strlen);

  // calculate
  md5_v2((unsigned int *) &single_md5Node.w[0], single_md5Node.state[0], single_md5Node.state[1], single_md5Node.state[2], single_md5Node.state[3]);

  // encode the 4-wide unsigned int array to a char array
  Encode(target_digest_d, single_md5Node.state, 16);

  //// print out the hash!
  printf(""HEXMD5PATTERN"\n", HEXMD5(target_digest_d));
}

//double diffclock(clock_t clock1,clock_t clock2)
//{
//	double diffticks=clock1-clock2;
//	double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
//	return diffms;
//} 



void MD5StringCuda_pre(int shared_mem_block, char* &charset_h, char* &charset_d, md5Plain* &returnMd5s_h, md5Plain* &returnMd5s_d, int* &perm_init_index_d) {
  // Choose which GPU to run on, change this on a multi-GPU system.
  checkCudaErrors(cudaSetDevice(0));
  // every thread will just calc the same md5 hash

  // TODO RE-ENABLE
  //dim3 dimGrid(2, 1, 1);
  //dim3 dimBlock(128, 1, 1);

  // copy charset to GPU
  checkCudaErrors(cudaMalloc(&charset_d, strlen(charset_h) * sizeof(char)));
  checkCudaErrors(cudaMemcpy(charset_d, charset_h, strlen(charset_h) * sizeof(char), cudaMemcpyHostToDevice));

  // malloc the return array of hashes.
  printf("allocating %d bytes\n", MAX_PERMS_PER_KERNEL * sizeof(md5Plain));
#ifdef ENABLE_MEMORY
    checkCudaErrors(cudaMalloc(&returnMd5s_d, MAX_PERMS_PER_KERNEL * sizeof(md5Plain)));
#else
    checkCudaErrors(cudaMalloc(&returnMd5s_d, 1 * sizeof(md5Plain)));
#endif

  // malloc starting permutation array
  checkCudaErrors(cudaMalloc(&perm_init_index_d, MAX_PW * sizeof(int)));
}

bool MD5StringCuda_kernel(dim3 dimGrid, dim3 dimBlock, int shared_mem_block, int *perm_init_index_h, int *perm_init_index_d, char *charset_d, int charset_len, int pw_length, md5Plain *returnMd5s_h, md5Plain *returnMd5s_d) {
  bool error = false;
  // copy the init perm to device
  error = checkCudaErrors(cudaMemcpy(perm_init_index_d, perm_init_index_h, MAX_PW * sizeof(int), cudaMemcpyHostToDevice));
  if(error)
    return error;

  // invoke kernel
  md5_kernel<<<dimGrid, dimBlock, shared_mem_block>>>(perm_init_index_d, charset_d, charset_len, pw_length, returnMd5s_d);
  error = checkCudaErrors(cudaDeviceSynchronize());
  if(error)
    return error;

  // copy the md5s calculated back to the host
#ifdef ENABLE_MEMORY
  checkCudaErrors(cudaMemcpy(returnMd5s_h, returnMd5s_d, MAX_PERMS_PER_KERNEL * sizeof(md5Plain), cudaMemcpyDeviceToHost));
#endif
}

//void MD5StringCuda(dim3 dimGrid, dim3 dimBlock, char *charset, md5Digest *all_digests_h, int digests_length) {
//  // Choose which GPU to run on, change this on a multi-GPU system.
//  checkCudaErrors(cudaSetDevice(0));
//
//
//  // every thread will just calc the same md5 hash
//
//  // TODO RE-ENABLE
//  //dim3 dimGrid(2, 1, 1);
//  //dim3 dimBlock(128, 1, 1);
//
//  char* target_plaintext_h = "abccd";
//  char* target_plaintext_d;
//  unsigned char* target_digest_d;
//  unsigned char target_digest_h[16];
//  checkCudaErrors(cudaMalloc(&target_plaintext_d, strlen(target_plaintext_h) * sizeof(char)));
//  checkCudaErrors(cudaMemcpy(target_plaintext_d, target_plaintext_h, strlen(target_plaintext_h) * sizeof(char), cudaMemcpyHostToDevice));
//  checkCudaErrors(cudaMalloc(&target_digest_d, 16 * sizeof(char)));
//
//  // --------
//  md5_single<<<1, 1>>>(target_plaintext_d, strlen(target_plaintext_h), target_digest_d);
//  // --------
//  checkCudaErrors(cudaDeviceSynchronize());
//  checkCudaErrors(cudaMemcpy(target_digest_h, target_digest_d, 16 * sizeof(char), cudaMemcpyDeviceToHost));
//  checkCudaErrors(cudaFree(target_plaintext_d));
//  printf("Brute forcing:\n");
//  printf("Hash('%s') == "HEXMD5PATTERN"\n", target_plaintext_h, HEXMD5(target_digest_h));
//
//
//  md5Digest *all_digests_d;
//  printf("length of md5Digest = %d\n", digests_length * sizeof(md5Digest));
//  checkCudaErrors(cudaMalloc(&all_digests_d, digests_length * sizeof(md5Digest)));
//  checkCudaErrors(cudaMemcpy(all_digests_d, all_digests_h, digests_length * sizeof(md5Digest), cudaMemcpyHostToDevice));
//
//
//
//  // Launch a kernel on the GPU with one thread for each element.
//  // sharedmem is 64 bytes for each thread
//  char* charset_d;
//  checkCudaErrors(cudaMemcpy(charset_d, charset, strlen(charset) * sizeof(char), cudaMemcpyHostToDevice));
//
//  printf("strlen of chraset = %d\n", strlen(charset));
//  // We need 2x ptrs to shared_mem in the kernel. The 2nd array needs to be word-aligned, so we pad the charset
//  int charset_padding = strlen(charset) % 4;
//  int shared_mem_block = dimBlock.x * sizeof(md5Node) + (strlen(charset)+charset_padding) * sizeof(char);
//  printf("shared mem allocated = %d bytes\n", shared_mem_block);
//
//  // put charset in global mem TODO?
//  checkCudaErrors(cudaMalloc(&charset_d, strlen(charset) * sizeof(char)));
//  checkCudaErrors(cudaMemcpy(charset_d, charset, strlen(charset) * sizeof(char), cudaMemcpyHostToDevice));
//
//  // malloc the return array of hashes.
//  md5Plain* returnMd5s_d;
//  md5Plain* returnMd5s_h = new md5Plain[MAX_PERMS_PER_KERNEL];
//  checkCudaErrors(cudaMalloc(&returnMd5s_d, MAX_PERMS_PER_KERNEL * sizeof(md5Plain)));
//
//  clock_t begin=clock();
//
//  for(int pw_length = 1; pw_length < 2; pw_length++) {
//    printf("Launching kernel for permutations of length %d ", pw_length);
//    int max_iters = pow((float)strlen(charset), (float)pw_length);
//    for(int j = 0; j <= (max_iters / MAX_PERMS_PER_KERNEL); j++) {
//      printf(".");
//      md5_kernel<<<dimGrid, dimBlock, shared_mem_block>>>(charset_d, strlen(charset), pw_length, target_digest_d, j, all_digests_d, digests_length, returnMd5s_d);
//      checkCudaErrors(cudaDeviceSynchronize());
//      checkCudaErrors(cudaMemcpy(returnMd5s_h, returnMd5s_d, MAX_PERMS_PER_KERNEL * sizeof(md5Plain), cudaMemcpyDeviceToHost));
//
//      cpuCheckMD5Intersection(returnMd5s_h, strlen(charset), pw_length, all_digests_h, digests_length);
//    }
//    printf(" done\n");
//  }
//  clock_t end=clock();
//  printf("Time Elapsed: %lf secs\n", double(diffclock(end, begin)/1000.0));
//
//
//  printf("returned\n");
//  
//
//  return;
//}