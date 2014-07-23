#ifndef NIFTIMATLABIO_H_INCLUDED
#define NIFTIMATLABIO_H_INCLUDED
//  $Id: niftiMatlabIO.h,v 1.8 2008-06-25 18:03:42 valerio Exp $	
//****************************************************************************
//
// Modification History (most recent first)
// mm/dd/yy  Who  What
//
// 08/18/05  VPL  
//
//****************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mex.h"
#include "matrix.h"

#include "nifti1.h"
#include "nifti1_io.h"

enum ActionType {INIT, OPEN, CLOSE, SEEK, LOAD, READ, READCHUNK, WRITE, WRITECHUNK, CREATE, UPDATE, VERIFY, ERROR};
inline ActionType operator++(ActionType &at) { return at = (ActionType)(at + 1); }
typedef unsigned char BYTE;

//****************************************************************************
//  
// Names of the fields in nifti_image struct. For explanation, see nifti1_io.h 
//  
//****************************************************************************

char *GetString(const mxArray *par);
ActionType ParseAction(char *par);

void StoreInt(const char *name, int *store, const mxArray *strp, int numValues);
void StoreFloat(const char *name, float *store, const mxArray *strp, int numValues);
void StoreString(const char *name, char store[], const mxArray *strp, int size);
void StoreCharArray(const char *name, char **store, const mxArray *strp);
void StoreMat44(const char *name, float *store, const mxArray *strp);
void StoreExtension(nifti1_extension *store, const mxArray *strp, int numExt);
void StoreData(nifti_image *nis, const mxArray *strp);
void StoreNiftiStruct(const mxArray *strp, nifti_image *nis);

void SetInt(const char *name, int *value, mxArray *strp, int numValues);
void SetFloat(const char *name, float *value, mxArray *strp, int numValues);
void SetString(const char *name, char *store, mxArray *strp, int size);
void SetMat44(const char *name, float *value, mxArray *strp);
void SetExtension(nifti1_extension *store, mxArray *strp, int numExt);
void SetData(nifti_image *nis, mxArray **strp);
void SetNiftiStruct(mxArray *strp, nifti_image *nis);

// API functions

bool OpenFile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], ActionType act, char *errMsg);
bool CloseFile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg);
bool SeekFile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg);
bool LoadImage(int nrhs, const mxArray *prhs[], char *errMsg); 
bool ReadData(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg);
bool ReadChunk(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg);
bool InitStruct(int np, mxArray **strp, char *errMsg);
bool WriteData(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg);
bool WriteChunk(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg);
bool VerifyStructure(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg);

// Convert stored pointer to pointer we use

inline znzFile GetFilePointer(const mxArray *par)
{
	long sp = (long )mxGetScalar(par);
	znzFile fp = znzFile(sp);
	if ( fileno(fp->nzfptr) == -1 ) fp = NULL;
	return fp;
}

// Utility functions

int ComputeSkipRanges(const mxArray *rangePars, nifti_image *nis, int ranges[7][2], size_t skip[7]);
size_t ReadSkip(znzFile zfp, nifti_image *nis, int dim, size_t skip[7], int ranges[7][2],
				size_t chunk, void * &data, char *errMsg);
size_t WriteSkip(znzFile zfp, nifti_image *nis, int dim, size_t skip[7], int ranges[7][2],
				size_t chunk, void * &data, char *errMsg);
#endif

