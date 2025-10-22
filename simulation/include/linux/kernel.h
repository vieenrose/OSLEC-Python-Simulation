#ifndef _LINUX_KERNEL_H
#define _LINUX_KERNEL_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))
#endif

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef clamp
#define clamp(val, lo, hi) (max((lo), min((val), (hi))))
#endif

#ifndef likely
#define likely(x) (x)
#endif

#ifndef unlikely
#define unlikely(x) (x)
#endif

#ifndef __FLS_DEFINED
static inline int fls(unsigned int x)
{
	int position = 0;

	while (x != 0) {
		position++;
		x >>= 1;
	}

	return position;
}
#define __FLS_DEFINED 1
#endif

#ifndef printk
#define printk(...) fprintf(stderr, __VA_ARGS__)
#endif

#endif /* _LINUX_KERNEL_H */
