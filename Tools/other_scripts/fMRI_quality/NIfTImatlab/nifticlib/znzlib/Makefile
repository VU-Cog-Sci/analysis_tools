include ../Makefile

PROJNAME = znzlib

INCFLAGS = $(ZLIB_INC)

SRCS=znzlib.c
OBJS=znzlib.o

TESTXFILES = testprog

depend:	
	$(RM) -f depend.mk
	$(CC) $(DEPENDFLAGS) $(INCFLAGS) $(SRCS) >> depend.mk

depend.mk:
	$(CC) $(DEPENDFLAGS) $(INCFLAGS) $(SRCS) >> depend.mk

lib: libznz-LINUX.a

test: $(TESTXFILES)

znzlib.o: znzlib.c znzlib.h
	$(CC) -c $(CFLAGS) $(USEZLIB) $(INCFLAGS) $<

libznz-LINUX.a: $(OBJS)
	$(AR) -r libznz.a $(OBJS)
	$(RANLIB) libznz.a

libznz-OSX.a:
	$(RM) -f znzlib.o libznz.a
	$(MAKE) CFLAGS="$(CFLAGS) -m64 -arch i386 -arch ppc" znzlib.o
	mv znzlib.o znzlib_64.o
	$(MAKE) CFLAGS="$(CFLAGS) -m32 -arch i386 -arch ppc" znzlib.o
	mv znzlib.o znzlib_32.o
	lipo znzlib_*.o -create -output znzlib.o
	$(AR) -r libznz.a znzlib.o
	$(RANLIB) libznz.a

testprog: libznz.a testprog.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o testprog testprog.c $(ZLIB_LIBS)

include depend.mk
