# GPLv2 and 2C-BSD
# Copyright (c) Darko Veberic, 2014

#CPPFLAGS := -I. $(shell auger-offline-config --cppflags) -DDEBUG
CPPFLAGS := -I. $(shell auger-offline-config --cppflags)
#CXXFLAGS := -Wall -Wextra -Wpedantic -std=c++11 -ggdb3 -O0 -fno-inline -pipe
CXXFLAGS := -Wall -Wextra -ggdb3 -O2 -fPIC -pipe 
LDFLAGS := $(CXXFLAGS) $(shell auger-offline-config --ldflags)

SHELL := bash
SUFFIXES := .o .cc .cxx

EXES := $(basename $(wildcard *.cxx))
OBJS := $(patsubst %.cc, %.o, $(wildcard *.cc))

DEPS := $(patsubst %.o, %.P, $(OBJS)) $(addsuffix .P, $(EXES))

define cxx_compile_with_dependency_creation
  $(COMPILE.cc) -MD -o $@ $<
  @sed -e 's|.*:|$*.o:|' <$*.d >$*.P
  @sed -e 's/.*://' -e 's/\\$$//' <$*.d | fmt -1 | sed -e 's/^ *//' -e 's/$$/:/' >>$*.P
  @rm -f $*.d
endef

define cxx_link_rule
  $(CXX) $(CXXFLAGS) $^ $(LDFLAGS) -o $@
endef

%.o: %.cc
	$(call cxx_compile_with_dependency_creation)

%.o: %.cxx
	$(call cxx_compile_with_dependency_creation)

%: %.o
	$(call cxx_link_rule)

.PHONY: all
all: $(EXES)

read_adst: read_adst.o $(OBJS)


.PHONY: clean
clean:
	- $(RM) -f $(OBJS) $(addsuffix .o, $(EXES)) $(EXES) $(DEPS)

-include $(DEPS)
