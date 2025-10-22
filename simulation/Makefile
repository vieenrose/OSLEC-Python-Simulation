OSLEC_ROOT ?= ..
BUILD_DIR ?= $(CURDIR)
LIB_NAME := liboslec.so

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fPIC
LDFLAGS ?= -shared
LDLIBS ?=
INCLUDES := -I$(OSLEC_ROOT) -I$(CURDIR)/include
SOURCES := $(OSLEC_ROOT)/echo.c

all: $(LIB_NAME)

$(LIB_NAME): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SOURCES) $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) $(LIB_NAME)

.PHONY: all clean
