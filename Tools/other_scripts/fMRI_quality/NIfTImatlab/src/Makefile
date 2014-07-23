#	$Id: Makefile.in,v 1.12 2009-02-26 19:18:18 valerio Exp $
# -*- Mode: Makefile -*-

#==============================================================================

# Build Variables

top_srcdir  = ..
srcdir      = .

LIBS  = -L../nifticlib/lib -lniftiio -lznz -lz

INCDIR  = -I../include -I../nifticlib/include

OS	    = LINUX
CPU	    = i686

MATLAB_FLAGS = -nodisplay -nodesktop -nojvm -nosplash
ifeq (${TMW_ROOT},)
   TMW_ROOT = $(MATDIR:/bin/=)
endif
TMWMEX = $(TMW_ROOT)/bin/mex

# Other Variables

BINDIR = ../matlab/@niftifile/private/
ifeq ($(OS),OSX)
   ifeq ($(CPU),powerpc)
	EXTENSION = mexmac
   else 
	EXTENSION = mexmaci
   endif
else
   EXTENSION = mexglx
endif

CSRC = niftiMatlabIO.cpp

OBJECTS = $(CSRC:.cpp=.o)

CFLAGS  = -I. $(INCDIR) -DHAVE_ZLIB
default: all

clean: 
	\rm -f $(OBJECTS) $(BINDIR)/niftiMatlabIO.$(EXTENSION)

debug: 
	$(TMWMEX) -ggdb $(CFLAGS) -outdir $(BINDIR) $(CSRC) $(LIBS)

all: 
	$(TMWMEX) $(CFLAGS) -outdir $(BINDIR) $(CSRC) $(LIBS)

# end of file -- Makefile --
