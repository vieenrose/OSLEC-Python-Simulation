OSLEC_ROOT ?= ..
BUILD_DIR ?= $(CURDIR)
LIB_NAME := liboslec.so
ALT_LIB := liboslec_ipnlms.so

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -fPIC
LDFLAGS ?= -shared
LDLIBS ?=
INCLUDES := -I$(OSLEC_ROOT) -I$(CURDIR)/include
SOURCES := $(OSLEC_ROOT)/echo.c
IPNLMS_SOURCE := $(OSLEC_ROOT)/echo_ipnlms.c

all: $(LIB_NAME)

$(LIB_NAME): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(SOURCES) $(LDFLAGS) $(LDLIBS)

ipnlms: $(ALT_LIB)

$(ALT_LIB): $(IPNLMS_SOURCE)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(IPNLMS_SOURCE) $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) $(LIB_NAME) $(ALT_LIB)

.PHONY: all clean ipnlms
