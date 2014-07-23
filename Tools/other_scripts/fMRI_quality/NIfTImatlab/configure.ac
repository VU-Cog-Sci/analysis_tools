#	$Id: configure.ac,v 1.6 2008-06-25 18:03:42 valerio Exp $
AC_INIT(NIfTImatlab, 1.0)

dnl Checks the host. For now just Gnu-Linux and OSX-Darwin

AC_CANONICAL_HOST 
SYSTEM=$host
HW_CPU=$host_cpu
case $SYSTEM in
     *-linux*)
	OS_TYPE="LINUX"
	OS_DBGF="-ggdb"
	;;
     *darwin*)
	OS_TYPE="OSX"
	OS_DBGF="-g"
	;;
esac
AC_SUBST(OS_TYPE)
AC_SUBST(OS_DBGF)
AC_SUBST(HW_CPU)

dnl this is to define where matlab lives

AC_SUBST(TMW_ROOT)

dnl Checks for programs.

AC_PROG_CXX
AC_PROG_CC
AC_PROG_MAKE_SET
AC_PROG_INSTALL

dnl Checks for header files.

AC_HEADER_STDC

dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST

dnl Checks for library functions.

AC_OUTPUT(Makefile src/Makefile nifticlib/znzlib/Makefile nifticlib/niftilib/Makefile)
