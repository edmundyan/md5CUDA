#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <functional>
#include <string>
#include <time.h>
#include <boost/unordered_map.hpp>




#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include "utils_cpu.h"
#include "deviceQuery.h"
#include "string.h"
// MD5(C, Rivest)
#include "global.h"
#include "md5.h"

struct md5Digest_hash
  : std::unary_function<md5Digest, std::size_t>
{
  std::size_t operator()(md5Digest const& e) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, e.d[0]);
    return seed;
  }
};

struct md5Digest_equalto
  : std::binary_function<md5Digest, md5Digest, bool>
{
  bool operator()(md5Digest const& x, md5Digest const& y) const
  {
    for(int i = 0; i < 16; i++) {
      if(x.d[i] != y.d[i])
        return false;
    }
    return true;
  }
};
typedef boost::unordered_map<md5Digest, int, md5Digest_hash, md5Digest_equalto> map;

// --------------

// -----


int compareMD5Digest(unsigned char *digest, unsigned char *target_digest) {
  for(int i = 0; i < 16;i++) {
    if(digest[i] != target_digest[i])
      return false;
  }
  return true;
}

//cpuCheckMD5Intersection(returnMd5s_h, pw_length, all_digests_h, digests_length);
//void cpuCheckMD5Intersection(md5Plain* returnMd5s_h, int charset_len, int pw_length, md5Digest *all_digests_h, int digests_length, map all_digests_hash){
//  // loop over everything hash inside returnMd5s, and see if the hash equals any of the digests
//  // inside all_digests_h
//  
//  // the min of the two
//  int max_iters = pow((float)charset_len, (float)pw_length);
//  if(max_iters > MAX_PERMS_PER_KERNEL)
//    max_iters = MAX_PERMS_PER_KERNEL;
//
//  printf("\n");
//  for(int i = 0; i < max_iters; i++) {
//    for(int j = 0; j < digests_length; j++) {
//      if(compareMD5Digest(returnMd5s_h[i], all_digests_h[j])) {
//        printf("md5('%s') = '"HEXMD5PATTERN"'\n", returnMd5s_h[i].plaintext, HEXMD5(returnMd5s_h[i].digest));
//        break;
//      }
//    }
//  }
//
//  printf("\nthe max iters = %d\n", max_iters);
//
//}


void convertmd5HashFile() {
  char line[80];
  unsigned char *digest = new unsigned char [16];

  FILE *fp = fopen("yahoo-plaintext_linux.txt", "r");
  FILE *out = fopen("yahoo-md5.txt", "w");

  while(fgets(line, 80, fp) != NULL) {
    // remove newline char
    line[strlen(line)-1] = '\0';

    // calc the md5 digest
    MD5String(line, digest);

    for (int i = 0; i < 16; i++)
	    fprintf (out, "%02x", digest[i]);
    fprintf(out, "\n");
  }

  fclose(fp);
  fclose(out);
}

void importmd5HashFile(char* filename, md5Digest *dbArray, map &dbHashes) {
  char line[80];
  int i;

  FILE *fp = fopen(filename, "r");
  i = 0;
  while(fgets(line, 80, fp) != NULL) {
    // remove newline char
    line[strlen(line)-1] = '\0';
    sscanf(line, ""HEXMD5PATTERN, HEXMD5SCAN(dbArray[i].d));
    printf("importing "HEXMD5PATTERN"\n",  HEXMD5(dbArray[i].d));
    //md5HashTable[all_digests[i]] = 1;
    // does this create the key??
    dbHashes[dbArray[i]];

    //md5DigestSingle temp;
    //for(int j = 0; j < 16; j++)
    //  temp[j] = dbArray[i].d[j];
    //dbHashes2[temp];
    i++;
  }
  fclose(fp);
}

void printBruteForce(char* charset, int len) {
  char init[26+1] = "aaaaaaaaaaaaaaaaaaaaaaaaaa";
  char output[32][26+1];
  int base = strlen(charset);
  int id, id_tmp;
  int index;
  int perms[5];  // NEEDS TO BE output[len], but static now
  int i, j;


  for(id = 0; id < 32; id++) {
    id_tmp = id;
    for(i = 0; i < 5; i++) {
      //perms[i] = id_tmp % base;
      output[id][i] = charset[id_tmp % base];
      id_tmp = id_tmp / base;
      
    }
    output[id][i] = '\0';
  }
  //for(i = 0; i < 32; i++) {
  //  // init to 0
  //  for(j = 0; j<5; j++) {
  //    perms[j] = 0;
  //  }
  //  id = i;
  //  index = 0;

  //  // conver base
  //  while (id != 0)
  //  {
  //    perms[index] = id % base;
  //    id = id / base;
  //    index++;
  //  }
  //  for(j=0; j < 5;j++) {
  //    output[i][j] = charset[perms[j]];
  //  }
  //  output[i][j] = '\0';  // add null ending string
  //}

  for(i = 0; i < 32;i++) {
    printf("%s\n", output[i]);
  }


  printf("%s \n", init);
  printf("%s \n", init + sizeof(char));


}



void cpuIntersectDatabaseHashes(md5Plain* returnMd5s_h, map &dbHashes, int max_iters) {
  int max_mini_loops = max_iters;
  if(max_mini_loops > MAX_PERMS_PER_KERNEL)
    max_mini_loops = MAX_PERMS_PER_KERNEL;

  printf("max_mini_loops = %d\n", max_mini_loops);
  for(int k = 0; k < max_mini_loops; k++) {
    if (dbHashes.count(returnMd5s_h[k].digest)) {
      // FOUND A MATCH
      printf("FOUND A MATCH\n");
      printf("%s\n", returnMd5s_h[k].plaintext);
      printf(HEXMD5PATTERN"\n", HEXMD5(returnMd5s_h[k].digest.d));
    }
  }
}

void bruteForceLaunch(dim3 dimGrid, dim3 dimBlock, char *charset, int digests_length, md5Digest *dbArray, map &dbHashes) {
  // device ptrs
  char* charset_d;
  md5Plain* returnMd5s_d;
  md5Plain* returnMd5s_h;
  int* perm_init_index_d;
  //md5Plain* returnMd5s_h = new md5Plain[MAX_PERMS_PER_KERNEL];


  // local vars
  int perm_init_index[10] = {0}; // we are limiting pws to 10 char maximum
  int perm_init_new[10] = {0}; // we are limiting pws to 10 char maximum
  clock_t begin, begin_all, end, end_all;

  printf("strlen of chraset = %d\n", strlen(charset));
  // We need 2x ptrs to shared_mem in the kernel. The 2nd array needs to be word-aligned, so we pad the charset
  int charset_padding = strlen(charset) % 4;
  int shared_mem_block = dimBlock.x * sizeof(md5Node) + (strlen(charset)+charset_padding) * sizeof(char);
  
  // init CUDA with all the mallocs
  printf("pre-kernel inits\n");
  MD5StringCuda_pre(shared_mem_block, charset, charset_d, returnMd5s_h, returnMd5s_d, perm_init_index_d);



  begin_all=clock();
  // for i=1..7
  for(int pw_length = 6; pw_length < 7; pw_length++) {
    // zero out our init perm
    for(int i = 0; i < 10; i++) perm_init_index[i] = 0;

    printf("Launching kernel for permutations of length %d \n", pw_length);
    //int max_iters = pow((float)strlen(charset), (float)pw_length);

    while(true) {
      printf("starting perm: ");
      for(int i = 0; i < 10; i++)
        printf("%d,", perm_init_index[i]);
      printf("\n");

       //invoke kernel
      begin=clock();
      if(MD5StringCuda_kernel(dimGrid, dimBlock, shared_mem_block, perm_init_index, perm_init_index_d, charset_d, strlen(charset), pw_length, returnMd5s_h, returnMd5s_d)){
        // there was an ERROR!! ABOUT!
        printf("ABORT! ABORT!!!!!\n");
        exit(1);
      }

      end=clock();
      printf("Kernel: %lf secs\n", double(diffclock(end, begin)/1000.0));


      // update the initial permutation
      if(generatePermStartingIndicesCPU(perm_init_index, pw_length, strlen(charset), MAX_PERMS_PER_KERNEL)) {
        break;  // while
      }
    }

    //for(int j = 0; j <= (max_iters / MAX_PERMS_PER_KERNEL); j++) {

    //  printf(".");
    //  // kernel invocation
    //  clock_t begin=clock();
    //  MD5StringCuda_kernel(dimGrid, dimBlock, shared_mem_block, charset_d, strlen(charset), pw_length, j, returnMd5s_h, returnMd5s_d);
    //  clock_t end=clock();
    //  printf("Kernel: %lf secs\n", double(diffclock(end, begin)/1000.0));

    //  //begin=clock();
    //  //cpuIntersectDatabaseHashes(returnMd5s_h, dbHashes, max_iters);
    //  //end=clock();
    //  //printf("Database Cmp: %lf secs\n", double(diffclock(end, begin)/1000.0));
    //  // compare returned value to database


    //  // chckecing the ashtable, move to function
    //}
    printf(" done\n");
  }
  end_all=clock();
  printf("Time Elapsed: %lf secs\n", double(diffclock(end_all, begin_all)/1000.0));

  



  //int max_iters = pow((float)charset_len, (float)pw_length);
  //if(max_iters > MAX_PERMS_PER_KERNEL)
  //  max_iters = MAX_PERMS_PER_KERNEL;

  //int max_iters = 26;
  //for(int i = 0; i < max_iters; i++) {
  //  // what we calcled from GPU
  //  //printf("%s\n", returnMd5s_h[i].plaintext);
  //  //printf(HEXMD5PATTERN"\n", HEXMD5(returnMd5s_h[i].digest));

  //  // what our dbArray is:
  //  //printf(HEXMD5PATTERN"\n", HEXMD5(dbArray[i].d));

  //  // STUPID
  //  md5Digest temp;
  //  for(int j = 0; j < 16; j++) {
  //    temp.d[j] = returnMd5s_h[i].digest[j];
  //  }
  //  if (dbHashes.count(temp)) {
  //    // FOUND A MATCH
  //    printf("FOUND A MATCH\n");
  //    printf("%s\n", returnMd5s_h[i].plaintext);
  //    printf(HEXMD5PATTERN"\n", HEXMD5(returnMd5s_h[i].digest));
  //    /*key exist*/ 
  //  }
  //}

  printf("done\n");

}


int main()
{
  int i;
  int loop = 1;
  char charset[26+1] = "abcdefghijklmnopqrstuvwxyz";


  //int perm_init_index[10] = {0};
  //int perm_init_new[10] = {0};
  //while(true) {
  //  printf("loop %d\n", loop);
  //  // invoke kernel

  //  // update the initial permutation
  //  //void generatePermStartingIndicesCPU(int *starting, int* c0_perms, int pw_length, char *charset, int charset_len, int idx) {
  //  int pw_length = 7;

  //  for(i = 0; i < 10; i++)
  //    printf("%d,", perm_init_index[i]);
  //  printf("\n");
  //  if(generatePermStartingIndicesCPU(perm_init_index, perm_init_new, pw_length, strlen(charset), 10000000)) {
  //    printf("TRUE\n");
  //    break;  // while
  //  }

  //  printf("FALSE\n", loop);
  //  for(i = 0; i < 10;i++)
  //    perm_init_index[i] = perm_init_new[i];

  //  loop++;
  //}
  //return 0;


  //char c0[64];
  //int c0_perms[10] = {0}; // we are limiting pws to 10 char maximum
  //int c0_starting[10] = {0}; // we are limiting pws to 10 char maximum
  //int idx = 100;
  //generatePermCPU(c0, c0_perms, 3, charset, strlen(charset), idx);
  //printf("%s\n",c0);
  //for(i = 0; i < 10; i++) {
  //  printf("%d,", c0_perms[i]);
  //}
  //printf("\n");
  //printf("---\n");

  //for(i=0;i<10;i++) c0_starting[i] = c0_perms[i];

  //generatePermStartingCPU(c0_starting, c0, c0_perms, 3, charset, strlen(charset), idx);
  //printf("%s\n",c0);
  //for(int i = 0; i < 10; i++) {
  //  printf("%d,", c0_perms[i]);
  //}
  //printf("\n");
  //return 1;



  //convertmd5HashFile();

  
  int lines = 10; // TODO HACK
  char filename[80] = "md5_mini.txt";
  //int lines = 453492; // TODO HACK
  //char filename[80] = "yahoo-md5.txt";

  // stored serially in array
  md5Digest *dbArray;
  dbArray = new md5Digest[lines];

  // stored in a boost::unordered_map
  map dbHashes;
  //single_map dbHashesSingle;
  printf("importing md5 hashes...");
  importmd5HashFile(filename, dbArray, dbHashes);
  printf("DONE!\n");

  
  //printBruteForce(charset, 5);
  deviceQuery();
  // CUDA function call

  // faster shiet DELETEME
  dim3 dimGrid(256, 1, 1);
  dim3 dimBlock(256, 1, 1);
  
  
  bruteForceLaunch(dimGrid, dimBlock, charset, lines, dbArray, dbHashes);
  //MD5StringCuda(dimGrid, dimBlock, charset, all_digests, lines, all_digests_hash);
	// C++ function call
	//MD5String("password");


    return 0;
}