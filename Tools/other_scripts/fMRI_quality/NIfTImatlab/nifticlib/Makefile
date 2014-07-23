SHELL		=	csh

## Projects
NIFTI		=	niftilib
ZNZ		=	znzlib
UTILS		=	utils
UTILS_PROGS	=	nifti_tool nifti1_test
THIS_DIR	=	`basename ${PWD}`

## Note the TARFILE_NAME embeds the release version number
TARFILE_NAME	=	nifticlib-1.0.0


## Compiler  defines
cc		=	gcc
CC		=	gcc
AR		=	ar
RANLIB  = ranlib
DEPENDFLAGS	=	-MM
GNU_ANSI_FLAGS	= 	-Wall -ansi -pedantic -fPIC
ANSI_FLAGS	= 	${GNU_ANSI_FLAGS}
CFLAGS		=	$(ANSI_FLAGS)

## Command defines
## gmake does not work on MacOSX or some versions of linux MAKE  = gmake 
RM		=	rm
MV		=	mv
CP		=	cp
TAR		=	/usr/local/pkg/gnu/bin/tar


## Installation
INSTALL_BIN_DIR	=	bin
INSTALL_LIB_DIR	=	lib
INSTALL_INC_DIR	=	include


## Zlib defines
ZLIB_INC	=	-I/usr/include
ZLIB_PATH	=	-L/usr/lib
ZLIB_LIBS 	= 	$(ZLIB_PATH) -lm -lz 

#################################################################

## ZNZ defines
ZNZ_INC		=	-I../$(ZNZ)
ZNZ_PATH	=	-L../$(ZNZ)
ZNZ_LIBS	=	$(ZNZ_PATH)  -lznz
USEZLIB         =       -DHAVE_ZLIB

## NIFTI defines
NIFTI_INC	=	-I../$(NIFTI)
NIFTI_PATH	=	-L../$(NIFTI)
NIFTI_LIBS	=	$(NIFTI_PATH) -lniftiio

## Rules

.SUFFIXES: .c .o

.c.o:
	$(CC) -c $(CFLAGS) $(INCFLAGS) $<

## Targets

all:	   utils

install:   $(INSTALL_INC_DIR) $(INSTALL_LIB_DIR) $(INSTALL_BIN_DIR) znz_install nifti_install utils_install

clean:	   znz_clean nifti_clean utils_clean

znz:
	$(RM) -f $(ZNZ)/depend.mk
	(cd $(ZNZ); $(MAKE) lib;)
	@echo " ----------- $(ZNZ) build completed."
	@echo ""

nifti:	znz
	$(RM) -f $(NIFTI)/depend.mk
	(cd $(NIFTI); $(MAKE) lib;)
	@echo " ----------- $(NIFTI) build completed."
	@echo ""

utils:  nifti_install znz_install
	(cd $(UTILS); $(MAKE) all;)
	@echo Utility programs built.
	@echo ""

$(INSTALL_BIN_DIR):
	mkdir -p $@

$(INSTALL_INC_DIR):
	mkdir -p $@

$(INSTALL_LIB_DIR):
	mkdir -p $@

znz_install: $(INSTALL_INC_DIR) $(INSTALL_LIB_DIR) $(INSTALL_BIN_DIR) nifti
	($(CP) $(ZNZ)/*.a $(INSTALL_LIB_DIR); $(CP) $(ZNZ)/*.h $(INSTALL_INC_DIR);)
	$(RANLIB) $(INSTALL_LIB_DIR)/*.a
	@echo " $(ZNZ) installed."
	@echo ""

nifti_install: $(INSTALL_INC_DIR) $(INSTALL_LIB_DIR) $(INSTALL_BIN_DIR) nifti
	($(CP) $(NIFTI)/*.a $(INSTALL_LIB_DIR); $(CP) $(NIFTI)/*.h $(INSTALL_INC_DIR);)
	$(RANLIB) $(INSTALL_LIB_DIR)/*.a
	@echo " $(NIFTI) installed."
	@echo ""

utils_install: $(INSTALL_INC_DIR) $(INSTALL_LIB_DIR) $(INSTALL_BIN_DIR) utils
	(cd $(UTILS); $(CP) $(UTILS_PROGS) ../$(INSTALL_BIN_DIR);)
	@echo " $(UTILS) installed."
	@echo ""

install_clean:
	($(RM) -f $(INSTALL_INC_DIR)/* $(INSTALL_LIB_DIR)/* $(INSTALL_BIN_DIR)/*;)

znz_clean:
	(cd $(ZNZ); $(RM) -f *.o *.a core; $(RM) -f depend.mk;)
	$(RM) -f $(INSTALL_INC_DIR)/* $(INSTALL_LIB_DIR)/*

nifti_clean:
	(cd $(NIFTI); $(RM) -f *.o *.a core; $(RM) -f depend.mk;)
	$(RM) -f $(INSTALL_INC_DIR)/* $(INSTALL_LIB_DIR)/*

utils_clean:
	(cd $(UTILS); $(MAKE) clean;)
	$(RM) -f $(INSTALL_BIN_DIR)/*

help:
	@echo ""
	@echo "all:           build and install znz, nifti1 and"
	@echo "               libraries, and the utils programs"
	@echo ""
	@echo "install:       install znz, nifti1, libraries"
	@echo ""
	@echo "clean:         remove .o .a etc. files from source directories,"
	@echo "               and remove examples and utils programs"
	@echo "clean_all:     make clean and delete all installed files in bin,"
	@echo "               include and lib"
	@echo ""
	@echo "znz:              build the znz library"
	@echo "znz_install:      install the znz library"
	@echo "nifti:            build the nifti1 library"
	@echo "nifti_install:    install the nifti1 library"
	@echo "utils:            make the utils programs"
	@echo "utils_install:    install the utils programs"
	@echo ""
