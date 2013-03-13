#define NUMBER_OF_ELEMENTS 5
#define BLOCK_DIM 1024
#define OFFSET 0
// When MAX_PERM = 0, means find all permutations
#define MAX_PERM 0
#define NEXT_PERM_LOOP 1

__constant__ long long arr[20][20] = { /*Not shown here to save space*/ };

// function to swap character 
// a - the character to swap with b
// b - the character to swap with a
__device__ void swap(
    char* a, 
    char* b)
{
    char tmp = *a;
    *a = *b;
    *b = tmp;
}


// function to reverse the array (sub array in array)
// first - 1st character in the array (sub-array in array)
// last - 1 character past the last character
__device__ void reverse(
    char* first, 
    char* last)
{    
    for (; first != last && first != --last; ++first)
        swap(first, last);
}


// function to find the next permutation (sub array in array)
// first - 1st character in the array (sub-array in array)
// last - 1 character past the last character
__device__ void next_permutation(
    char* first, 
    char* last)
{
    char* next = last;
    --next;
    if(first == last || first == next)
        return;

    while(true)
    {
        char* next1 = next;
        --next;
        if(*next < *next1)
        {
            char* mid = last;
            --mid;
            for(; !(*next < *mid); --mid)
                ;
            swap(next, mid);
            reverse(next1, last);
            return;
        }

        if(next == first)
        {
            reverse(first, last);
            return;
        }
    }
}    

__global__ void PermuteHybrid(char* arrDest, long long* offset, long long* Max)
{
    long long index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index >= (*Max/(NEXT_PERM_LOOP+1)))
        return;

    index *= NEXT_PERM_LOOP+1;
    long long tmpindex = index;
    
    index += *offset;
    
    char arrSrc[NUMBER_OF_ELEMENTS];
    char arrTaken[NUMBER_OF_ELEMENTS];
    for(char i=0; i<NUMBER_OF_ELEMENTS; ++i)
    {
        arrSrc[i] = i;
        arrTaken[i] = 0;
    }

    char size = NUMBER_OF_ELEMENTS;
    for(char i=NUMBER_OF_ELEMENTS-1; i>=0; --i)
    {
        for(char j=i; j>=0; --j)
        {
            if(index >= arr[i][j])
            {
                char foundcnt = 0;
                index = index - arr[i][j];
                for(char k=0;k<NUMBER_OF_ELEMENTS; ++k)
                {
                    if(arrTaken[k]==0) // not taken
                    {
                        if(foundcnt==j)
                        {
                            arrTaken[k] = 1; // set to taken
                            arrDest[ (tmpindex*NUMBER_OF_ELEMENTS) + (NUMBER_OF_ELEMENTS-size) ] = arrSrc[k];
                            break;
                        }
                        foundcnt++;
                    }
                }
                break;
            }
        }
        --size;
    }

    long long idx = tmpindex*NUMBER_OF_ELEMENTS;
    for(char a=1; a<NEXT_PERM_LOOP+1; ++a)
    {
        long long idx2 = a*NUMBER_OF_ELEMENTS;
        for(char i=0; i<NUMBER_OF_ELEMENTS; ++i)
        {
            arrDest[ idx + idx2 + i ] =
                arrDest[ idx + ((a-1)*NUMBER_OF_ELEMENTS) + i ];
        }
        next_permutation(arrDest + idx + idx2, 
            arrDest+idx + idx2 + NUMBER_OF_ELEMENTS);
    }
}

