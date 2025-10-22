#ifndef _LINUX_SLAB_H
#define _LINUX_SLAB_H

#include <stdlib.h>
#include <string.h>

#ifndef GFP_KERNEL
#define GFP_KERNEL 0
#endif

static inline void *kzalloc(size_t size, int flags)
{
	(void)flags;
	return calloc(1, size);
}

static inline void *kcalloc(size_t n, size_t size, int flags)
{
	(void)flags;
	return calloc(n, size);
}

static inline void kfree(const void *ptr)
{
	free((void *)ptr);
}

#endif /* _LINUX_SLAB_H */
