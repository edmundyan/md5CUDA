//#include <stdio.h>
//#include <stdio.h>
//#include <iostream>
//#include <functional>
//#include <string>
//#include <boost/unordered_map.hpp>
//
//#include "cuda_runtime.h" //dim3
//#include "device_launch_parameters.h"
//
//#include "kernel.h"
//#include "eyanHash.h"
//#include "math.h"
//#include "utils_cpu.h"
//
//
//
//
//int hashMd5Digest(md5Digest &input, int mod){
//  int i, j;
//  unsigned int state[4];
//  for (i = 0, j = 0; j < 16; i++, j += 4) {
//   state[i] = ((unsigned int)input.d[j]) | (((unsigned int)input.d[j+1]) << 8) |
//     (((unsigned int)input.d[j+2]) << 16) | (((unsigned int)input.d[j+3]) << 24);
//  }
//
//  return (state[0] ^ state[1] ^ state[2] ^ state[3]) % mod;
//}
//
//void convertToGhettoHash(md5Digest* input, md5Digest* &hash, int len) {
//  // input - a serial array
//  // hash - ptr to hash table
//  // len - number of elements in input[]
//
//
//  int newsize = len * 2;
//  int new_loc;
//  hash = new md5Digest[newsize]();
//
//  for(int i =0; i < len; i++) {
//    new_loc = hashMd5Digest(input[i], newsize);
//    if(hash[new_loc].d[0] == 0){
//      hash[new_loc] = input[i];
//    } else {
//      printf("COLISION\n");
//    }
//  }
//
//}