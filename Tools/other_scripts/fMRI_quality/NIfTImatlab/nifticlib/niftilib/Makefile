include ../Makefile

PROJNAME = niftiio


INCFLAGS = $(ZLIB_INC) $(ZNZ_INC)
LIBS = $(ZLIB_LIBS) $(ZNZ_LIBS) $(NIFTI_LIBS)

SRCS=nifti1_io.c
OBJS=nifti1_io.o
MOBJS=mjtest.o

SCRIPTS = 

depend:	
	$(RM) -f depend.mk
	$(CC) $(DEPENDFLAGS) $(INCFLAGS) $(SRCS) >> depend.mk

depend.mk:
	$(CC) $(DEPENDFLAGS) $(INCFLAGS) $(SRCS) >> depend.mk

lib: libniftiio-LINUX.a

nifti1_io.o: nifti1_io.c nifti1_io.h
	$(CC) -c $(CFLAGS) $(USEZLIB) $(INCFLAGS) $<

libniftiio-LINUX.a: $(OBJS)
	$(AR) -r libniftiio.a $(OBJS)
	$(RANLIB) libniftiio.a

libniftiio-OSX.a:
	$(RM) -f nifti1_io.o libniftiio.a
	$(MAKE) CFLAGS="$(CFLAGS) -m64 -arch i386 -arch ppc" nifti1_io.o
	mv nifti1_io.o nifti1_io_64.o
	$(MAKE) CFLAGS="$(CFLAGS) -m32 -arch i386 -arch ppc" nifti1_io.o
	mv nifti1_io.o nifti1_io_32.o
	lipo nifti1_io_*.o -create -output nifti1_io.o
	$(AR) -r libniftiio.a $(OBJS)
	$(RANLIB) libniftiio.a

include depend.mk
