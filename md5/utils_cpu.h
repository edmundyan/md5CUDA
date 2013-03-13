#ifndef utils_cpu_h_
#define utils_cpu_h_


#include <time.h>

//typedef struct
//{
//  unsigned char d[16];
//} md5Digest;

//typedef unsigned char md5DigestSingle[16];

double diffclock(clock_t clock1,clock_t clock2);

void generatePermCPU(char *c0, int* c0_perms, int pw_length, char *charset, int charset_len, int idx);
void generatePermStartingCPU(int *starting, char *c0, int* c0_perms, int pw_length, char *charset, int charset_len, int idx);
bool generatePermStartingIndicesCPU(int *starting, int pw_length, int charset_len, int idx);

//void cpuCheckMD5Intersection(md5Plain* returnMd5s_h, int charset_len, int pw_length, md5Digest *all_digests_h, int digests_length);
#endif // utils_cpu_h_