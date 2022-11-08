#pragma once

#define __VITIS_CL__    // Comment this line out if you are not using Vitis

// Fixed Width Integer Types
typedef unsigned long uint64_t;
typedef unsigned int uint32_t;
typedef unsigned short int uint16_t;

#define VRANGE 65536

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

// 2D array in natural layout
#define ARRAY2(ptr,iB,iN,dN) ((ptr)[(iB)*(dN)+(iN)])
