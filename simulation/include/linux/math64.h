#ifndef _LINUX_MATH64_H
#define _LINUX_MATH64_H

#include <stdint.h>

static inline uint64_t div_u64(uint64_t dividend, uint64_t divisor)
{
	return divisor ? dividend / divisor : 0;
}

static inline int64_t div_s64(int64_t dividend, int64_t divisor)
{
	return divisor ? dividend / divisor : 0;
}

#endif /* _LINUX_MATH64_H */
