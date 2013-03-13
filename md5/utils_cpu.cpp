#include <stdio.h>

#include <stdio.h>
#include <iostream>
#include <functional>
#include <string>
#include <boost/unordered_map.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
//#include "deviceQuery.h"
//#include "string.h"
#include "math.h"
#include "utils_cpu.h"
//
//// MD5(C, Rivest)
//#include "global.h"
//#include "md5.h"


double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
	return diffms;
} 




void generatePermCPU(char *c0, int* c0_perms, int pw_length, char *charset, int charset_len, int idx) {
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
      c0_perms[i] = idx % charset_len;
      c0[i] = charset[c0_perms[i]];
      idx = idx / charset_len;
    }
  }
  //printf("%d = '%s'\n", idx_orig, c0);
}

void generatePermStartingCPU(int *starting, char *c0, int* c0_perms, int pw_length, char *charset, int charset_len, int idx) {
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
      c0_perms[i] = (idx+starting[i]) % charset_len;
      c0[i] = charset[c0_perms[i]];
      idx = (idx+starting[i]) / charset_len;
    }
  }
  //printf("%d = '%s'\n", idx_orig, c0);
}


bool generatePermStartingIndicesCPU(int *starting, int pw_length, int charset_len, int idx) {
  // modifies c0 to be a random string perm corresponding to idx in base(charset_len)
	unsigned int len = 0;
  int i;
  int idx_orig = idx;
  int tmp[MAX_PW] = {0};
  bool reset = true;

  for(i = 0; i < pw_length; i++) {
    if(idx+starting[i] < charset_len)
      reset = false;
    tmp[i] = (idx+starting[i]) % charset_len;
    idx = (idx+starting[i]) / charset_len;
  }

  for(i = 0; i < pw_length; i++) {
    starting[i] = tmp[i];
  }

  return reset;
}