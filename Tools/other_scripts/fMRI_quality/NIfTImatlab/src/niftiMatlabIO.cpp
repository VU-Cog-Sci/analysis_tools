//  $Id: niftiMatlabIO.cpp,v 1.6 2009-04-07 20:18:20 valerio Exp $	
//****************************************************************************
//
// This is a Matlab mex interface to nifti's official I/O functions
//
// Modification History (most recent first)
// mm/dd/yy  Who  What
//
// 01/23/08  VPL  Make file pointer persistent so we don't have to open and
//                close it all the time.
// 10/17/06  VPL  Make sure matrices are converted from Matlab (column major) to
//                C (row major) and vice-versa
// 08/18/05  VPL  
//
//****************************************************************************

#include "niftiMatlabIO.h"

//****************************************************************************
//
// Purpose: Extract string value from a matlab parameter
//   
// Parameters: const mxArray *par - parameter to be read
//   
// Returns: char *str - if NULL then error
//   
// Notes: caller must use free to release memory
//
//****************************************************************************

char *GetString(const mxArray *par)
{
	char *retValue = NULL;
	if (mxIsChar(par) )
	{
		int strLen = mxGetN(par) * mxGetM(par) + 1;
		if (strLen > 0)
		{
			retValue = static_cast<char *>(mxCalloc(strLen, sizeof(char)));
			if (retValue != NULL)
			{
				if (mxGetString(par, retValue, strLen) == 1)
				{
					mxFree(retValue);
					retValue = NULL;
				}
			}
		}
	}

	return retValue;
}

//****************************************************************************
//
// Purpose: Parse the action argument and return a an ActionType
//   
// Parameters: char *par - action name to be parsed
//   
// Returns: ActionType [INIT..ERROR]
//   
// Notes: 
//
//****************************************************************************

static const char *actionNames[]={"INIT", "OPEN", "CLOSE", "SEEK", "LOAD", "READ", "READCHUNK",
								  "WRITE", "WRITECHUNK", "CREATE", "UPDATE", "VERIFY"};

ActionType ParseAction(char *par)
{
	ActionType retValue = ERROR;

	for (retValue = INIT; retValue < ERROR; ++retValue)
		if (strcasecmp(par, actionNames[retValue]) == 0)
			break;
	
	return retValue;
}

//****************************************************************************
//
// Purpose: Store integer value from matlab structure to c struct
//   
// Parameters: const char *name    - name of field
//             int *store          - where we store it
//             const mxArray *strp - pointer to matlab structure
//             int numValues       - number of values stored
//   
// Returns: 
//   
// Notes: The values in matlab are actually doubles, cast it and then verify that
//        no decimal values where added (in which case give a warning)
//
//****************************************************************************

void StoreInt(const char *name, int *store, const mxArray *strp, int numValues)
{
	mxArray *field = mxGetField(strp, 0, name);
	int *value = static_cast<int *>(mxGetData(field));
	double iPart;
	double dPart = modf(*value, &iPart);
	if (fabs(dPart) > 0.000001)
	{
		char errMsg[256];
		sprintf(errMsg, "Variable %s is stored as an integer, value will be truncated.", name);
		mexWarnMsgIdAndTxt("niftiMatlabIO:StoreInt:trunc", errMsg);
	}

	int valIndex;
	for (valIndex = 0; valIndex < numValues; ++valIndex)
		*store++ = *value++;
}

//****************************************************************************
//
// Purpose: Store float value from matlab structure to c struct
//   
// Parameters: const char *name    - name of field
//             float *store        - where we store it
//             const mxArray *strp - pointer to matlab structure
//             int numValues       - number of values stored
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void StoreFloat(const char *name, float *store, const mxArray *strp, int numValues)
{
	mxArray *field = mxGetField(strp, 0, name);
	float *value = static_cast<float *>(mxGetData(field));
	int valIndex;
	for (valIndex = 0; valIndex < numValues; ++valIndex)
		*store++ = *value++;
}

//****************************************************************************
//
// Purpose: Store data value from matlab structure to c struct
//   
// Parameters: nifti_image *nis    - where we store it
//             const mxArray *strp - pointer to matlab structure
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void StoreData(nifti_image *nis, const mxArray *strp)
{
	int nvox = mxGetM(strp) * mxGetN(strp);

	int nbyper = nis->nbyper;
	if (nis->datatype == DT_COMPLEX) nbyper /= 2;

	nis->data = mxCalloc(nvox * nbyper, 1);
	BYTE *store = static_cast<BYTE *>(nis->data);
	BYTE *value = static_cast<BYTE *>(mxGetData(strp));
	
	int valIndex;
	for (valIndex = 0; valIndex < nvox; ++valIndex)
	{
		memcpy(store, value, nbyper);
		store += nbyper;
		value += nbyper;
	}
}

//****************************************************************************
//
// Purpose: Store char value from matlab structure to c struct
//   
// Parameters: const char *name    - name of field
//             char store[]        - where we store it
//             const mxArray *strp - pointer to matlab structure
//             int size            - maximum string size (including NULL)
//   
// Returns: 
//   
// Notes: If the string has a fixe size (size > 0) and the user increased the
//        size of the string, truncate it and give a warning.
//
//****************************************************************************

void StoreString(const char *name, char store[], const mxArray *strp, int size)
{
	mxArray *field = mxGetField(strp, 0, name);
	int strLen = mxGetN(field);
	if (strLen > size - 1)
	{
		char errMsg[256];
		sprintf(errMsg, "Variable %s can store only %d characters, string will be truncated.",
				name, size - 1);
		mexWarnMsgIdAndTxt("niftiMatlabIO:StoreString:trunc", errMsg);
	}
	mxGetString(field, store, size);
}

//****************************************************************************
//
// Purpose: Allocate space and store char value from matlab structure to c struct
//   
// Parameters: const char *name    - name of field
//             char **store        - where we store it
//             const mxArray *strp - pointer to matlab structure
//   
// Returns: 
//   
// Notes: If the string has a fixe size (size > 0) and the user increased the
//        size of the string, truncate it and give a warning.
//
//****************************************************************************

void StoreCharArray(const char *name, char **store, const mxArray *strp)
{
	mxArray *field = mxGetField(strp, 0, name);
	*store = GetString(field);
}

//****************************************************************************
//
// Purpose: Store mat44 value from matlab structure to c struct
//   
// Parameters: const char *name    - name of field
//             float *store        - where we store it
//             const mxArray *strp - pointer to matlab structure
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void StoreMat44(const char *name, float *store, const mxArray *strp)
{
	mxArray *field = mxGetField(strp, 0, name);

	if ( field != NULL)
	{
		double *value = static_cast<double *>(mxGetData(field));
		int rowIndex, colIndex;
		for (rowIndex = 0; rowIndex < 4; ++rowIndex)
			for (colIndex = 0; colIndex < 4; ++colIndex)
				store[rowIndex + colIndex*4] = static_cast<float>(value[colIndex + rowIndex*4]);
	}
	
}

//****************************************************************************
//
// Purpose: Store nifti1_extension value from matlab structure to c struct
//   
// Parameters: nifti1_extension *store - where we store it
//             const mxArray *strp     - pointer to matlab structure
//             int numExt              - number of extensions
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void StoreExtension(nifti1_extension *store, const mxArray *strp, int numExt)
{
	mxArray *structField = mxGetField(strp, 0, "ext_list");
	int currExt;
	for (currExt = 0; currExt < numExt; ++currExt)
	{
		mxArray *structEntry;

		structEntry = mxGetField(structField, currExt, "esize");
		store[currExt].esize = *((int *)mxGetData(structEntry));
		
		structEntry = mxGetField(structField, currExt, "ecode");
		store[currExt].ecode = *((int *)mxGetData(structEntry));

		structEntry = mxGetField(structField, currExt, "edata");
		mxGetString(structEntry, store[currExt].edata, mxGetN(structEntry));
	}
}

//****************************************************************************
//
// Purpose: Store the info from the matlab structure in the c struct
//   
// Parameters: const mxArray *strp - matlab storage of structure
//             nifti_image *nis    - where we store the info
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void StoreNiftiStruct(const mxArray *strp, nifti_image *nis)
{
	StoreInt("ndim", &(nis->ndim), strp, 1);
	StoreInt("nx", &(nis->nx), strp, 1);
	StoreInt("ny", &(nis->ny), strp, 1);
	StoreInt("nz", &(nis->nz), strp, 1);
	StoreInt("nt", &(nis->nt), strp, 1);
	StoreInt("nu", &(nis->nu), strp, 1);
	StoreInt("nv", &(nis->nv), strp, 1);
	StoreInt("nw", &(nis->nw), strp, 1);
	StoreInt("dim", nis->dim, strp, 8);
	
	int tempVal;
	StoreInt("nvox", &tempVal, strp, 1);
	nis->nvox = size_t(tempVal);

	StoreInt("nbyper", &(nis->nbyper), strp, 1);
	StoreInt("datatype", &(nis->datatype), strp, 1);
	StoreInt("qform_code", &(nis->qform_code), strp, 1);
	StoreInt("sform_code", &(nis->sform_code), strp, 1);
	StoreInt("freq_dim", &(nis->freq_dim), strp, 1);
	StoreInt("phase_dim", &(nis->phase_dim), strp, 1);
	StoreInt("slice_dim", &(nis->slice_dim), strp, 1);
	StoreInt("slice_code", &(nis->slice_code), strp, 1);
	StoreInt("slice_start", &(nis->slice_start), strp, 1);
	StoreInt("slice_end", &(nis->slice_end), strp, 1);
	StoreInt("xyz_units", &(nis->xyz_units), strp, 1);
	StoreInt("time_units", &(nis->time_units), strp, 1);
	StoreInt("nifti_type", &(nis->nifti_type), strp, 1);
	StoreInt("intent_code", &(nis->intent_code), strp, 1);
	StoreInt("iname_offset", &(nis->iname_offset), strp, 1);
	StoreInt("swapsize", &(nis->swapsize), strp, 1);
	StoreInt("byteorder", &(nis->byteorder), strp, 1);
	StoreInt("num_ext", &(nis->num_ext), strp, 1);

	StoreFloat("dx", &(nis->dx), strp, 1);
	StoreFloat("dy", &(nis->dy), strp, 1);
	StoreFloat("dz", &(nis->dz), strp, 1);
	StoreFloat("dt", &(nis->dt), strp, 1);
	StoreFloat("du", &(nis->du), strp, 1);
	StoreFloat("dv", &(nis->dv), strp, 1);
	StoreFloat("dw", &(nis->dw), strp, 1);
	StoreFloat("scl_slope", &(nis->scl_slope), strp, 1);
	StoreFloat("scl_inter", &(nis->scl_inter), strp, 1);
	StoreFloat("cal_min", &(nis->cal_min), strp, 1);
	StoreFloat("cal_max", &(nis->cal_max), strp, 1);
	StoreFloat("slice_duration", &(nis->slice_duration), strp, 1);
	StoreFloat("quatern_b", &(nis->quatern_b), strp, 1);
	StoreFloat("quatern_c", &(nis->quatern_c), strp, 1);
	StoreFloat("quatern_d", &(nis->quatern_d), strp, 1);
	StoreFloat("qoffset_x", &(nis->qoffset_x), strp, 1);
	StoreFloat("qoffset_y", &(nis->qoffset_y), strp, 1);
	StoreFloat("qoffset_z", &(nis->qoffset_z), strp, 1);
	StoreFloat("qfac", &(nis->qfac), strp, 1);
	StoreFloat("toffset", &(nis->toffset), strp, 1);
	StoreFloat("intent_p1", &(nis->intent_p1), strp, 1);
	StoreFloat("intent_p2", &(nis->intent_p2), strp, 1);
	StoreFloat("intent_p3", &(nis->intent_p3), strp, 1);
	StoreFloat("pixdim", nis->pixdim, strp, 8);

	StoreString("intent_name", nis->intent_name, strp, 16);
	StoreString("descrip", nis->descrip, strp, 80);
	StoreString("aux_file", nis->aux_file, strp, 24);
	StoreCharArray("fname", &(nis->fname), strp);
	StoreCharArray("iname", &(nis->iname), strp);

	StoreMat44("qto_xyz", (float *)nis->qto_xyz.m, strp);
	StoreMat44("qto_ijk", (float *)nis->qto_ijk.m, strp);
	StoreMat44("sto_xyz", (float *)nis->sto_xyz.m, strp);
	StoreMat44("sto_ijk", (float *)nis->sto_ijk.m, strp);
	
	StoreExtension(nis->ext_list, strp, nis->num_ext);
}

//****************************************************************************
//
// Purpose: Get the int value from the c struct and store it in the matlab
//          structure. If the field doesn't exist, create it.
//   
// Parameters: const char *name    - name of the field
//             int *value          - value(s) to be stored
//             const mxArray *strp - pointer to matlab structure
//             int numValues       - number of values stored
//   
// Returns: 
//   
// Notes: In matlab, the values are actually integers
//
//****************************************************************************

void SetInt(const char *name, int *value, mxArray *strp, int numValues)
{
	mxArray *fieldValue = mxCreateNumericMatrix(1, numValues, mxINT32_CLASS, mxREAL);
	int fieldNumber = mxGetFieldNumber(strp, name);
	if (fieldNumber == -1)
	{
		fieldNumber = mxAddField(strp, name);
		char errMsg[256];
		sprintf(errMsg, "Unable to add %s field", name);
		mexErrMsgTxt(errMsg);
	}

	if (value != NULL)
	{
		int *dataPtr = static_cast<int *>(mxGetData(fieldValue));
		int valInd;
		for (valInd = 0; valInd < numValues; ++valInd)
			*dataPtr++ = *value++;
	}

	mxSetFieldByNumber(strp, 0, fieldNumber, fieldValue);
}

//****************************************************************************
//
// Purpose: Get the float values from the c struct and store it in the matlab
//          structure. If the field doesn't exist, create it.
//   
// Parameters: const char *name    - name of the field
//             float *value        - value(s) to be stored
//             const mxArray *strp - pointer to matlab structure
//             int numValues       - number of values stored
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void SetFloat(const char *name, float *value, mxArray *strp, int numValues)
{
	mxArray *fieldValue = mxCreateNumericMatrix(1, numValues, mxSINGLE_CLASS, mxREAL);
	int fieldNumber = mxGetFieldNumber(strp, name);
	if (fieldNumber == -1)
	{
		fieldNumber = mxAddField(strp, name);
		char errMsg[256];
		sprintf(errMsg, "Unable to add %s field", name);
		mexErrMsgTxt(errMsg);
	}
		
	if (value != NULL)
	{
		float *dataPtr = static_cast<float *>(mxGetData(fieldValue));
		int valInd;
		for (valInd = 0; valInd < numValues; ++valInd)
			*dataPtr++ = *value++;
	}

	mxSetFieldByNumber(strp, 0, fieldNumber, fieldValue);
}

//****************************************************************************
//
// Purpose: Get the data from the c struct and store it in the matlab
//          structure. If the field doesn't exist, create it.
//   
// Parameters: nifti_image *nis - structure with all the info
//             mxArray **strp    - pointer to matlab structure
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void SetData(nifti_image *nis, mxArray **strp)
{
	char errMsg[256];
	mxClassID classID;
	mxComplexity complexFlag = mxREAL;
	switch ( nis->datatype )
	{
		case DT_RGB:					// Same as DT_RGB24
		case DT_FLOAT128:
		case DT_COMPLEX128:
		case DT_COMPLEX256:
		case DT_INT64:
		case DT_UINT64:
			sprintf(errMsg, "Unsupported data type");
			mexErrMsgTxt(errMsg);
			break;
			
		case DT_BINARY:
			classID = mxLOGICAL_CLASS;
			break;
			
		case DT_UNSIGNED_CHAR:          // Same as DT_UINT8
			classID = mxUINT8_CLASS;
			break;
			
		case DT_INT8:
			classID = mxINT8_CLASS;
			break;
			
		case DT_SIGNED_SHORT:			// Same as DT_INT16
			classID = mxINT16_CLASS;
			break;
			
		case DT_UINT16:
			classID = mxUINT16_CLASS;
			break;
			
		case DT_SIGNED_INT:				// Same as DT_INT32
			classID = mxINT32_CLASS;
			break;
			
		case DT_UINT32:
			classID = mxUINT32_CLASS;
			break;

		case DT_FLOAT:					// Same as DT_FLOAT32
			classID = mxSINGLE_CLASS;
			break;
			
		case DT_COMPLEX:				// Same as DT_COMPLEX64
			complexFlag = mxCOMPLEX;
			classID = mxSINGLE_CLASS;
			break;
			
		case DT_DOUBLE:					// Same as DT_FLOAT64
			classID = mxDOUBLE_CLASS;
			break;
	}

	int nvox = nis->data == NULL ? 0 : nis->nvox;
	// begin SJI
	// bug in the intel beta of matlab.  can't properly handle complex data
	// we will read twice as many singles and let the matlab wrapper handle it.
	if (nis->datatype == DT_COMPLEX)  {
	  *strp = mxCreateNumericMatrix(1, 2*nvox, classID, mxREAL);
	}
	else {
	  *strp = mxCreateNumericMatrix(1, nvox, classID, complexFlag);
	}
	// end SJI
	if (nis->data != NULL)
	{
		if (nis->datatype == DT_COMPLEX)
		{
           		// begin SJI
		        // void *realPtr = mxGetPr(*strp);
		        // void *imagPtr = mxGetPi(*strp);
			BYTE *dataPtr = static_cast<BYTE *>(mxGetData(*strp));
			BYTE *store = static_cast<BYTE *>(nis->data);

			int valInd;
			for (valInd = 0; valInd < nvox; ++valInd)
			{
			//	memcpy(realPtr, store, nis->nbyper/2);
			//	store += nis->nbyper/2;
			//	realPtr += nis->nbyper/2;
			//	memcpy(imagPtr, store, nis->nbyper/2);
			//	store += nis->nbyper/2;
			//	imagPtr += nis->nbyper/2;
				memcpy(dataPtr, store, nis->nbyper);
				store += nis->nbyper;
				dataPtr += nis->nbyper;
			}
			// end SJI
		}
		else
		{
			BYTE *dataPtr = static_cast<BYTE *>(mxGetData(*strp));
			BYTE *store = static_cast<BYTE *>(nis->data);
	
			int valInd;
			for (valInd = 0; valInd < nvox; ++valInd)
			{
				memcpy(dataPtr, store, nis->nbyper);
				store += nis->nbyper;
				dataPtr += nis->nbyper;
			}
		}
	}
}

//****************************************************************************
//
// Purpose: Get the string values from the c struct and store it in the matlab
//          structure. If the field doesn't exist, create it.
//   
// Parameters: const char *name    - name of the field
//             char *value         - string to be stored
//             const mxArray *strp - pointer to matlab structure
//             int size            - maximum size of string
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void SetString(const char *name, char *value, mxArray *strp, int size)
{
	int fieldNumber = mxGetFieldNumber(strp, name);
	if (fieldNumber == -1)
	{
		fieldNumber = mxAddField(strp, name);
		char errMsg[256];
		sprintf(errMsg, "Unable to add %d field", name);
		mexErrMsgTxt(errMsg);
	}

	mxArray *fieldValue = NULL;
	if (value == NULL)
	{
		int dims[2];
		dims[0] = 1;
		dims[1] = size;
		
		fieldValue = mxCreateCharArray(2, dims);
	}
	else
	{
		fieldValue = mxCreateString(value);
	}

	mxSetFieldByNumber(strp, 0, fieldNumber, fieldValue);
}

//****************************************************************************
//
// Purpose: Get the mat44 values from the c struct and store it in the matlab
//          structure. If the field doesn't exist, create it.
//   
// Parameters: const char *name    - name of the field
//             float *value        - values to be stored
//             const mxArray *strp - pointer to matlab structure
//   
// Returns: 
//   
// Notes: The mat44 is a 4x4 matrix of floats
//
//****************************************************************************

void SetMat44(const char *name, float *value, mxArray *strp)
{
	mxArray *fieldValue = mxCreateNumericMatrix(4, 4, mxDOUBLE_CLASS, mxREAL);
	int fieldNumber = mxGetFieldNumber(strp, name);
	if (fieldNumber == -1)
	{
		fieldNumber = mxAddField(strp, name);
		char errMsg[256];
		sprintf(errMsg, "Unable to add %d field", name);
		mexErrMsgTxt(errMsg);
	}
		
	if (value != NULL)
	{
		double *dataPtr = static_cast<double *>(mxGetData(fieldValue));
		int rowIndex, colIndex;
		for (rowIndex = 0; rowIndex < 4; ++rowIndex)
			for (colIndex = 0; colIndex < 4; ++colIndex)
				dataPtr[colIndex + rowIndex*4] = value[rowIndex + colIndex*4];
	}

	mxSetFieldByNumber(strp, 0, fieldNumber, fieldValue);
}

//****************************************************************************
//
// Purpose: Get the nifti1_extension values from the c struct and store it in
//          the matlab structure. If the field doesn't exist, create it.
//   
// Parameters: nifti1_extension *value - values to be stored
//             mxArray *strp           - pointer to matlab structure
//             int numExt              - number of extensions to store
//   
// Returns: 
//   
// Notes: The mat44 is a 4x4 matrix of floats
//
//****************************************************************************

void SetExtension(nifti1_extension *value, mxArray *strp, int numExt)
{
	int fieldNumber = mxGetFieldNumber(strp, "ext_list");
	if (fieldNumber == -1)
	{
		fieldNumber = mxAddField(strp, "ext_list");
		char errMsg[256];
		sprintf(errMsg, "Unable to add ext_list field");
		mexErrMsgTxt(errMsg);
	}

	const char *extNames[] = {"esize", "ecode", "edata"};
	int dims[] = {1, 1};
	dims[1] = numExt > 0 ? numExt : 1;
	
	mxArray *structField = mxCreateStructArray(2, dims, 3, extNames);
	if ( value == NULL )
	{
		mxSetField(structField, 0, "esize", mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL));
		mxSetField(structField, 0, "ecode", mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL));
		int strDims[] = {1, 0};
		mxSetField(structField, 0, "edata", mxCreateCharArray(2, dims));
	}
	else
	{
		int currExt;
		int strDims[] = {1, 0};
		for (currExt = 0; currExt < numExt; ++currExt)
		{
			int *dataPtr = NULL;
			
			mxArray *sEntry = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
			dataPtr = static_cast<int *>(mxGetData(sEntry));
			*dataPtr = value[currExt].esize;
			mxSetField(structField, currExt, "esize", sEntry);

			mxArray *cEntry = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
			dataPtr = static_cast<int *>(mxGetData(cEntry));
			*dataPtr = value[currExt].ecode;
			mxSetField(structField, currExt, "ecode", cEntry);
			
			mxArray *dEntry = mxCreateString(value[currExt].edata);
			mxSetField(structField, currExt, "edata", dEntry);
		}
	}
			   
	mxSetFieldByNumber(strp, 0, fieldNumber, structField);
}

//****************************************************************************
//
// Purpose: Store the info from the c struct in the matlab structure
//   
// Parameters: mxArray *strp    - matlab storage of structure
//             nifti_image *nis - the c struct
//   
// Returns: 
//   
// Notes: Fields will be added if needed. If c struct is NULL, just create
//        fields with proper types.
//
//****************************************************************************

void SetNiftiStruct(mxArray *strp, nifti_image *nis)
{
	if (nis != NULL)
	{
		SetInt("ndim", &(nis->ndim), strp, 1);
		SetInt("nx", &(nis->nx), strp, 1);
		SetInt("ny", &(nis->ny), strp, 1);
		SetInt("nz", &(nis->nz), strp, 1);
		SetInt("nt", &(nis->nt), strp, 1);
		SetInt("nu", &(nis->nu), strp, 1);
		SetInt("nv", &(nis->nv), strp, 1);
		SetInt("nw", &(nis->nw), strp, 1);
		SetInt("dim", nis->dim, strp, 8);

		int tempVal = int(nis->nvox);
		SetInt("nvox", &tempVal, strp, 1);
		
		SetInt("nbyper", &(nis->nbyper), strp, 1);
		SetInt("datatype", &(nis->datatype), strp, 1);
		SetInt("qform_code", &(nis->qform_code), strp, 1);
		SetInt("sform_code", &(nis->sform_code), strp, 1);
		SetInt("freq_dim", &(nis->freq_dim), strp, 1);
		SetInt("phase_dim", &(nis->phase_dim), strp, 1);
		SetInt("slice_dim", &(nis->slice_dim), strp, 1);
		SetInt("slice_code", &(nis->slice_code), strp, 1);
		SetInt("slice_start", &(nis->slice_start), strp, 1);
		SetInt("slice_end", &(nis->slice_end), strp, 1);
		SetInt("xyz_units", &(nis->xyz_units), strp, 1);
		SetInt("time_units", &(nis->time_units), strp, 1);
		SetInt("nifti_type", &(nis->nifti_type), strp, 1);
		SetInt("intent_code", &(nis->intent_code), strp, 1);
		SetInt("iname_offset", &(nis->iname_offset), strp, 1);
		SetInt("swapsize", &(nis->swapsize), strp, 1);
		SetInt("byteorder", &(nis->byteorder), strp, 1);
		SetInt("num_ext", &(nis->num_ext), strp, 1);

		SetFloat("dx", &(nis->dx), strp, 1);
		SetFloat("dy", &(nis->dy), strp, 1);
		SetFloat("dz", &(nis->dz), strp, 1);
		SetFloat("dt", &(nis->dt), strp, 1);
		SetFloat("du", &(nis->du), strp, 1);
		SetFloat("dv", &(nis->dv), strp, 1);
		SetFloat("dw", &(nis->dw), strp, 1);
		SetFloat("scl_slope", &(nis->scl_slope), strp, 1);
		SetFloat("scl_inter", &(nis->scl_inter), strp, 1);
		SetFloat("cal_min", &(nis->cal_min), strp, 1);
		SetFloat("cal_max", &(nis->cal_max), strp, 1);
		SetFloat("slice_duration", &(nis->slice_duration), strp, 1);
		SetFloat("quatern_b", &(nis->quatern_b), strp, 1);
		SetFloat("quatern_c", &(nis->quatern_c), strp, 1);
		SetFloat("quatern_d", &(nis->quatern_d), strp, 1);
		SetFloat("qoffset_x", &(nis->qoffset_x), strp, 1);
		SetFloat("qoffset_y", &(nis->qoffset_y), strp, 1);
		SetFloat("qoffset_z", &(nis->qoffset_z), strp, 1);
		SetFloat("qfac", &(nis->qfac), strp, 1);
		SetFloat("toffset", &(nis->toffset), strp, 1);
		SetFloat("intent_p1", &(nis->intent_p1), strp, 1);
		SetFloat("intent_p2", &(nis->intent_p2), strp, 1);
		SetFloat("intent_p3", &(nis->intent_p3), strp, 1);
		SetFloat("pixdim", nis->pixdim, strp, 8);

		SetString("intent_name", nis->intent_name, strp, 15);
		SetString("descrip", nis->descrip, strp, 79);
		SetString("aux_file", nis->aux_file, strp, 23);
		SetString("fname", nis->fname, strp, 0);
		SetString("iname", nis->iname, strp, 0);

		SetMat44("qto_xyz", (float *)nis->qto_xyz.m, strp);
		SetMat44("qto_ijk", (float *)nis->qto_ijk.m, strp);
		SetMat44("sto_xyz", (float *)nis->sto_xyz.m, strp);
		SetMat44("sto_ijk", (float *)nis->sto_ijk.m, strp);

		SetExtension(nis->ext_list, strp, nis->num_ext);
	}
	else
	{
		SetInt("ndim", NULL, strp, 1);
		SetInt("nx", NULL, strp, 1);
		SetInt("ny", NULL, strp, 1);
		SetInt("nz", NULL, strp, 1);
		SetInt("nt", NULL, strp, 1);
		SetInt("nu", NULL, strp, 1);
		SetInt("nv", NULL, strp, 1);
		SetInt("nw", NULL, strp, 1);
		SetInt("dim", NULL, strp, 8);
		SetInt("nvox", NULL, strp, 1);
		SetInt("nbyper", NULL, strp, 1);
		SetInt("datatype", NULL, strp, 1);
		SetInt("qform_code", NULL, strp, 1);
		SetInt("sform_code", NULL, strp, 1);
		SetInt("freq_dim", NULL, strp, 1);
		SetInt("phase_dim", NULL, strp, 1);
		SetInt("slice_dim", NULL, strp, 1);
		SetInt("slice_code", NULL, strp, 1);
		SetInt("slice_start", NULL, strp, 1);
		SetInt("slice_end", NULL, strp, 1);
		SetInt("xyz_units", NULL, strp, 1);
		SetInt("time_units", NULL, strp, 1);
		SetInt("nifti_type", NULL, strp, 1);
		SetInt("intent_code", NULL, strp, 1);
		SetInt("iname_offset", NULL, strp, 1);
		SetInt("swapsize", NULL, strp, 1);
		SetInt("byteorder", NULL, strp, 1);
		SetInt("num_ext", NULL, strp, 1);

		SetFloat("dx", NULL, strp, 1);
		SetFloat("dy", NULL, strp, 1);
		SetFloat("dz", NULL, strp, 1);
		SetFloat("dt", NULL, strp, 1);
		SetFloat("du", NULL, strp, 1);
		SetFloat("dv", NULL, strp, 1);
		SetFloat("dw", NULL, strp, 1);
		SetFloat("scl_slope", NULL, strp, 1);
		SetFloat("scl_inter", NULL, strp, 1);
		SetFloat("cal_min", NULL, strp, 1);
		SetFloat("cal_max", NULL, strp, 1);
		SetFloat("slice_duration", NULL, strp, 1);
		SetFloat("quatern_b", NULL, strp, 1);
		SetFloat("quatern_c", NULL, strp, 1);
		SetFloat("quatern_d", NULL, strp, 1);
		SetFloat("qoffset_x", NULL, strp, 1);
		SetFloat("qoffset_y", NULL, strp, 1);
		SetFloat("qoffset_z", NULL, strp, 1);
		SetFloat("qfac", NULL, strp, 1);
		SetFloat("toffset", NULL, strp, 1);
		SetFloat("intent_p1", NULL, strp, 1);
		SetFloat("intent_p2", NULL, strp, 1);
		SetFloat("intent_p3", NULL, strp, 1);
		SetFloat("pixdim", NULL, strp, 8);

		SetString("intent_name", NULL, strp, 15);
		SetString("descrip", NULL, strp, 79);
		SetString("aux_file", NULL, strp, 23);
		SetString("fname", NULL, strp, 0);
		SetString("iname", NULL, strp, 0);

		SetMat44("qto_xyz", NULL, strp);
		SetMat44("qto_ijk", NULL, strp);
		SetMat44("sto_xyz", NULL, strp);
		SetMat44("sto_ijk", NULL, strp);

		SetExtension(NULL, strp, 1);
	}
}

//****************************************************************************
//
// Purpose: The hook from Matlab
//   
// Parameters: int nlhs          - number of left-hand arguments (1 or 0)
//             mxArray *plhs[]   - array of left-hand arguments
//             int nrhs          - number of right-hand arguments (> 0)
//   
// Returns: 
//   
// Notes: 
//
//****************************************************************************

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	// Sanity checks: only one return parameter, at last one input parameter
	// first input parameter must be a string

	if (nrhs < 1)
		mexErrMsgTxt("Not enough input arguments.");
	if (!mxIsChar(prhs[0]) )
		mexErrMsgTxt("First input argument must specify action.");
		
	char errMsg[256];
	bool error = false;

	// Read first input parameter and figure out the action

	char *actPar = GetString(prhs[0]);
	if (actPar == NULL)
		mexErrMsgTxt("Error reading first (action) input argument.");

	ActionType act = ParseAction(actPar);
	switch (act)
	{
		case INIT:
			error = InitStruct(nlhs, plhs, errMsg);
			break;
			
		case OPEN:
		case CREATE:
		case UPDATE:
			error = OpenFile(nlhs, plhs, nrhs, prhs, act, errMsg);
			break;

		case CLOSE:
			error = CloseFile(nlhs, plhs, nrhs, prhs, errMsg);
			break;
			
		case SEEK:
			error = SeekFile(nlhs, plhs, nrhs, prhs, errMsg);
			break;

		case LOAD:
			error = LoadImage(nrhs, prhs, errMsg);
			break;
			
		case READ:
			error = ReadData(nlhs, plhs, nrhs, prhs, errMsg);
			break;

		case READCHUNK:
			error = ReadChunk(nlhs, plhs, nrhs, prhs, errMsg);
			break;

		case WRITE:
			error = WriteData(nlhs, plhs, nrhs, prhs, errMsg);
			break;

		case WRITECHUNK:
			error = WriteChunk(nlhs, plhs, nrhs, prhs, errMsg);
			break;
			
		case VERIFY:
			error = VerifyStructure(nlhs, plhs, nrhs, prhs, errMsg);
			break;
			
		case ERROR:
		default:
			sprintf(errMsg, "Invalid action '%s'", actPar);
			error = true;
			break;
	}

	mxFree(actPar);
	if (error)
		mexErrMsgTxt(errMsg);
		
}

//****************************************************************************
//
// Purpose: Verify that structure is correct
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: 
//
//****************************************************************************

bool VerifyStructure(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;

	nifti_image *nis = static_cast<nifti_image *>(mxMalloc(sizeof(nifti_image)));
	memset(nis, 0, sizeof(nifti_image));
	
	StoreNiftiStruct(prhs[1], nis);
	nifti_1_header nhdr;
	nhdr = nifti_convert_nim2nhdr(nis);
	
	// The "nifti_hdr_looks_good" function does not handle ANALYZE 7.5 headers
	// so skip it in this case

	if ( nis->nifti_type == NIFTI_FTYPE_ANALYZE )
		plhs[0] = mxCreateDoubleScalar(1);
	else
		plhs[0] = mxCreateDoubleScalar(nifti_hdr_looks_good(&nhdr));
	
	mxFree(nis);

	return retValue;
}

//****************************************************************************
//
// Purpose: Open file for and return value
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             ActionType            - READ/WRITE/UPDATE
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: If there are no values in the nifti_image structure on the right hand
// side they are read from the file.
//
//****************************************************************************

bool OpenFile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], ActionType act, char *errMsg)
{
	bool retValue = false;
	
	// We'll check the types of the parameters

	if ( (nrhs != 2) || (nlhs != 2) || !mxIsStruct(prhs[1]) )
	{
		retValue = true;
		sprintf(errMsg, "\nusage: \t[<file pos>, <nifti_image struct>] = "
				"niftiMatlabIO('[OPEN|CREATE|UPDATE]', <nifti_image struct>)");
	}
	else
	{
		nifti_image *nis = static_cast<nifti_image *>(mxMalloc(sizeof(nifti_image)));
		memset(nis, 0, sizeof(nifti_image));
		StoreNiftiStruct(prhs[1], nis);

		// Open file and .... depends

		znzFile zfp = NULL;
		mexMakeMemoryPersistent((void *)zfp);

		switch ( act )
		{
			case OPEN:
			{
				char opts[] = "r";
				zfp = nifti_image_open(nis->fname, opts, &nis);
				
				// Fix bug in nifti_image_read where pixdim[0] doesn't get set
				if ( nis != NULL )
				{
					if ( nis->qfac >= 0 )
						nis->pixdim[0] = 1;
					else
						nis->pixdim[0] = -1;
				}
				break;
			}

			case CREATE:
				nis->byteorder = nifti_short_order();
				zfp = nifti_image_write_hdr_img(nis, 2, "w");
				break;

			case UPDATE:
			{
				char opts[] = "r";
				zfp = nifti_image_open(nis->fname, opts, &nis);
				break;
			}
		}
		
		if ( zfp == NULL || nis == NULL || nis->iname == NULL || nis->nbyper <= 0 || nis->nvox <= 0 )
		{
			retValue = true;
			sprintf(errMsg, "OpenFile: unable to open file %s", nis->fname);
			zfp = 0;
		}

		plhs[0] = mxCreateDoubleScalar((long )zfp);
		plhs[1] = mxDuplicateArray(prhs[1]);
		SetNiftiStruct(plhs[1], nis);
    }
		
	return retValue;
}

//****************************************************************************
//
// Purpose: Move within the file
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: 
//
//****************************************************************************

bool SeekFile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;
	
	// We'll check the types of the parameters

	if ( (nrhs != 4) || (nlhs != 1) )
	{
		retValue = true;
		sprintf(errMsg, "\nusage: \t<file pos> = niftiMatlabIO('SEEK', <offset>, <origin>, <file id>)");
	}
	else
	{
		long offset = static_cast<long>(mxGetScalar(prhs[1]));
		int whence = static_cast<int>(mxGetScalar(prhs[2]));

		// Get file identifier
		
		znzFile zfp = GetFilePointer(prhs[3]);

		if ( zfp == NULL )
		{
			sprintf(errMsg, "SeekFile: file not open.");
			return true;
		}
		
		if ( znzseek(zfp, offset, whence) != 0 )
		{
			retValue = true;
			sprintf(errMsg, "SeekFile: seek error in file");
			plhs[0] = mxCreateDoubleScalar(0);
		}
		else
		{
			plhs[0] = mxCreateDoubleScalar(znztell(zfp));
		}
	}
		
	return retValue;
}

//****************************************************************************
//
// Purpose: Close the file
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: 
//
//****************************************************************************

bool CloseFile(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;
	
	// We'll check the types of the parameters

	if ( nrhs != 2 )
	{
		retValue = true;
		sprintf(errMsg, "\nusage: \tniftiMatlabIO('CLOSE', <file id>)");
	}
	else
	{
		znzFile zfp = GetFilePointer(prhs[1]);
		if ( zfp == NULL )
		{
			sprintf(errMsg, "CloseFile: file not open.");
			return true;
		}
		
		if ( znzclose(zfp) )
		{
			retValue = true;
			sprintf(errMsg, "CloseFile: error closing file");
		}
	}
	
	return retValue;
}

//****************************************************************************
//
// Purpose: Load image using info stored in the nifti_image structure
//   
// Parameters: int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: 
//
//****************************************************************************

bool LoadImage(int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;
	
	// We'll check the types of the parameters, but we have to hope that the
	// nifti_image struct was initialized correctly before.

	if ((nrhs < 2) || !mxIsStruct(prhs[1]))
	{
		retValue = true;
		sprintf(errMsg, "\nusage: niftiMatlabIO('LOAD', <nifti_image struct>)\n"
				"\t<nifti_image struct> = pre-initialized nifti_image structure");
	}
	else
	{
		nifti_image *nis = static_cast<nifti_image *>(mxMalloc(sizeof(nifti_image)));
		memset(nis, 0, sizeof(nifti_image));

		StoreNiftiStruct(prhs[1], nis);

		if (nifti_image_load(nis) != 0)
		{
			retValue = true;
			sprintf(errMsg, "LoadImage: unable to load data from %s", nis->fname);
		}
		else
		{
			SetNiftiStruct((mxArray *)prhs[1], nis);
		}

		mxFree(nis->data);
		mxFree((void *)nis);
	}
	
	return retValue;
}

//****************************************************************************
//
// Purpose: Read next "chunk" of data from file and return it
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: 
//
//****************************************************************************

bool ReadData(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;

	// We'll check the types of the parameters, but we have to hope that the
	// nifti_image struct was initialized correctly before.

	if ((nrhs != 4) || (nlhs != 4) || !mxIsStruct(prhs[3]))
	{
		retValue = true;
		sprintf(errMsg, "\nusage: [<bytes read>, <data store>, <file position>, <nifti_image struct>]"
				" = niftiMatlabIO('READ', <file id>, <num bytes>, <nifti_image struct>)");
	}
	else
	{
		znzFile zfp = GetFilePointer(prhs[1]);
		if ( zfp == NULL )
		{
			sprintf(errMsg, "ReadData: file not open.");
			return true;
		}
		
		size_t ntot = static_cast<size_t>(mxGetScalar(prhs[2]));
		nifti_image *nis = static_cast<nifti_image *>(mxMalloc(sizeof(nifti_image)));
		memset(nis, 0, sizeof(nifti_image));
		StoreNiftiStruct(prhs[3], nis);

		nis->data = mxCalloc(ntot, 1);

		size_t bytesRead = nifti_read_buffer(zfp, nis->data, ntot, nis);
		if ( bytesRead != ntot)
		{
			char errMsg[256];
			sprintf(errMsg, "ReadData: read only %d bytes.", bytesRead);
			mexWarnMsgIdAndTxt("niftiMatlabIO:ReadData", errMsg);
		}

		if ( bytesRead > 0 )
			plhs[0] = mxCreateDoubleScalar(bytesRead);
		else
			plhs[0] = mxCreateDoubleScalar(0);
		
		nis->nvox = bytesRead / nis->nbyper;

		SetData(nis, &plhs[1]);

		plhs[2] = mxCreateDoubleScalar(znztell(zfp));
		plhs[3] = mxDuplicateArray(prhs[3]);
		SetNiftiStruct(plhs[3], nis);

		mxFree(nis->data);
	}
	
	return retValue;
}

//****************************************************************************
//
// Purpose: Read "chunk" of data from file and return it
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: 
//
//****************************************************************************

bool ReadChunk(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;
	
	// We'll check the types of the parameters, but we have to hope that the
	// nifti_image struct was initialized correctly before.

	if ((nlhs != 4) || (nrhs != 4) || !mxIsStruct(prhs[2]))
	{
		retValue = true;
		sprintf(errMsg, "\nusage: [<bytes read>, <data store>, <file position>, <nifti_image struct>]"
				" = niftiMatlabIO('READCHUNK', <file id>, <nifti_image struct>, <varargin>)");
	}
	else
	{
		znzFile zfp = GetFilePointer(prhs[1]);
		if ( zfp == NULL )
		{
			sprintf(errMsg, "ReadChunk: file not open.");
			return true;
		}
		
		nifti_image *nis = static_cast<nifti_image *>(mxMalloc(sizeof(nifti_image)));
		memset(nis, 0, sizeof(nifti_image));
		StoreNiftiStruct(prhs[2], nis);

		// Unpack the dimensions

		int ranges[7][2];
		size_t skip[7];
		int numPix = ComputeSkipRanges(prhs[3], nis, ranges, skip);
		size_t numBytes = numPix * nis->nbyper;
		nis->data = mxMalloc(numBytes);
		memset(nis->data, 0, numBytes);

		if ( nis->nifti_type == 1 )
			if ( znzseek(zfp, nis->iname_offset, SEEK_SET ) != 0 )
			{
				sprintf(errMsg, "ReadChunk: seek error in file %s", nis->iname);
				return true;
			}

		// Compute number of consecutive bytes you can read

		const size_t chunk = size_t((ranges[0][1] - ranges[0][0] + 1) * nis->nbyper);

		void *data = nis->data;
		size_t bytesRead = ReadSkip(zfp, nis, (nis->ndim - 1), skip, ranges, chunk, data, errMsg);

		if ( bytesRead == 0 )
		{
			plhs[0] = mxCreateDoubleScalar(0);
			plhs[1] = mxCreateDoubleScalar(nis->iname_offset);
			plhs[3] = mxDuplicateArray(prhs[2]);
			nis->nvox = 0;
			SetNiftiStruct(plhs[3], nis);
			mxFree(nis->data);
			return true;
		}
		
		plhs[0] = mxCreateDoubleScalar(bytesRead);
		
		nis->nvox = bytesRead / nis->nbyper;

		SetData(nis, &plhs[1]);

		size_t pos = znztell(zfp);
		plhs[2] = mxCreateDoubleScalar(pos);
		plhs[3] = mxDuplicateArray(prhs[2]);
		SetNiftiStruct(plhs[3], nis);

		mxFree(nis->data);
	}
	
	return retValue;
}

//****************************************************************************
//
// Purpose: Write "chunk" of data from file and return it
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes: Get data from variable passed in, don't use the pointer in the
//        data structure.
//
//****************************************************************************

bool WriteChunk(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;
	
	// We'll check the types of the parameters, but we have to hope that the
	// nifti_image struct was initialized correctly before.

	if ((nrhs != 5) || !mxIsStruct(prhs[2]))
	{
		retValue = true;
		sprintf(errMsg, "\nusage: [<bytes read>, <file position>]"
				" = niftiMatlabIO('WRITECHUNK', <file id>, <nifti_image struct>, <data>, <varargin>)");
	}
	else
	{
		znzFile zfp = GetFilePointer(prhs[1]);
		if ( zfp == NULL )
		{
			sprintf(errMsg, "WriteChunk: file not open.");
			return true;
		}
		
		nifti_image *nis = static_cast<nifti_image *>(mxMalloc(sizeof(nifti_image)));
		memset(nis, 0, sizeof(nifti_image));
		StoreNiftiStruct(prhs[2], nis);

		StoreData(nis, prhs[3]);
		
		// Unpack the dimensions

		int ranges[7][2];
		size_t skip[7];
		int numPix = ComputeSkipRanges(prhs[4], nis, ranges, skip);
		size_t numBytes = numPix * nis->nbyper;

		// Skip space of header, if needed

		if ( nis->nifti_type == 1 )
			if ( znzseek(zfp, nis->iname_offset, SEEK_SET ) != 0 )
			{
				sprintf(errMsg, "WriteChunk: seek error in file %s", nis->iname);
				return true;
			}

		// Compute number of consecutive bytes you can write

		const size_t chunk = size_t((ranges[0][1] - ranges[0][0] + 1) * nis->nbyper);

		void *data = nis->data;
		size_t bytesWritten = WriteSkip(zfp, nis, (nis->ndim - 1), skip, ranges, chunk, data, errMsg);

		if ( bytesWritten == 0 )
		{
			plhs[0] = mxCreateDoubleScalar(0);
			plhs[1] = mxCreateDoubleScalar(nis->iname_offset);
			return true;
		}
		
		plhs[0] = mxCreateDoubleScalar(bytesWritten / nis->nbyper);
		size_t pos = znztell(zfp);
		plhs[1] = mxCreateDoubleScalar(pos);
	}
	
	return retValue;
}

//****************************************************************************
//
// Purpose: This writes the image data or the header to file.
//   
// Parameters: int nlhs              - number of left-hand arguments
//             mxArray *plhs[]       - left-hand arguments
//	           int nrhs              - number of right-hand arguments
//             const mxArray *prhs[] - right-hand arguments
//             char *errMsg          - possible error message
//   
// Returns: If true, then error
//   
// Notes:
//
//
//****************************************************************************

bool WriteData(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[], char *errMsg)
{
	bool retValue = false;
	
	if ((nrhs != 5) || (nlhs != 2))
	{
		retValue = true;
		sprintf(errMsg, "\nusage: \t[size, <file position>]"
				"= niftiMatlabIO('write', <file id>, <nifti_image structure>, <data>, <num bytes>)");
	}
	else
	{
		znzFile zfp = GetFilePointer(prhs[1]);
		if ( zfp == NULL )
		{
			sprintf(errMsg, "WriteData: file not open.");
			return true;
		}
		
		nifti_image *nis = static_cast<nifti_image *>(mxMalloc(sizeof(nifti_image)));
		memset(nis, 0, sizeof(nifti_image));
		StoreNiftiStruct(prhs[2], nis);

		size_t bytesWritten = 0;
		size_t numBytes = static_cast<size_t>(mxGetScalar(prhs[4]));
		StoreData(nis, prhs[3]);
		bytesWritten = nifti_write_buffer(zfp, nis->data, numBytes);
		if ( bytesWritten != numBytes )
		{
			char errMsg[256];
			sprintf(errMsg, "WriteData: wrote only %d bytes.", bytesWritten);
			mexWarnMsgIdAndTxt("niftiMatlabIO:WriteData", errMsg);
		}
		
		plhs[0] = mxCreateDoubleScalar(bytesWritten / nis->nbyper);
		plhs[1] = mxCreateDoubleScalar(znztell(zfp));
			
		mxFree((void *)nis);
	}
	
	return retValue;
}

//****************************************************************************
//
// Purpose: Initialize the nifti structure
//   
// Parameters: int np         - number of elements in strp
//             mxArray **strp - parameter containing the structure
//             char *errMsg   - possible error message
//   
// Returns: If true, then error
//   
// Notes: For the meaning of the field names, see nifti1_io.h
//
//****************************************************************************

bool InitStruct(int np, mxArray **strp, char *errMsg)
{
	bool retValue = false;

	if (np < 1)
	{
		sprintf(errMsg, "InitStruct: I need a place to store the structure !");
		retValue = true;
	}
	else
	{
		int dims[2] = {1, 1};

#define NUMBER_OF_FIELDS (sizeof(niftiFieldNames)/sizeof(*niftiFieldNames))
		const char *niftiFieldNames[] = {"ndim", "nx", "ny", "nz", "nt", "nu", "nv", "nw",
										 "dim", "nvox", "nbyper", "datatype",
										 "dx", "dy", "dz", "dt", "du", "dv", "dw",
										 "pixdim", "scl_slope", "scl_inter",
										 "cal_min", "cal_max", "qform_code", "sform_code",
										 "freq_dim", "phase_dim", "slice_dim", "slice_code",
										 "slice_start", "slice_end", "slice_duration",
										 "quatern_b", "quatern_c", "quatern_d", "qoffset_x",
										 "qoffset_y", "qoffset_z", "qfac", "qto_xyz",
										 "qto_ijk", "sto_xyz", "sto_ijk", "toffset",
										 "xyz_units", "time_units", "nifti_type", "intent_code",
										 "intent_p1", "intent_p2", "intent_p3", "intent_name",
										 "descrip", "aux_file", "fname", "iname", "iname_offset",
										 "swapsize", "byteorder", "data", "num_ext", "ext_list"};

		*strp = mxCreateStructArray(2, dims, NUMBER_OF_FIELDS, niftiFieldNames);
		
		if (*strp == NULL)
		{
			sprintf(errMsg, "InistStruct: Error creating structure");
			retValue = true;
		}
		else
		{
			SetNiftiStruct(*strp, NULL);
		}
	
	}
	
	return retValue;
}

//****************************************************************************
//
// Purpose: Loop trough dimensions recursively, skipping needed space and
// reading desired data.
//   
// Parameters: znzFile zfp        - pointer to file
//             nifti_image *nis   - pointer to data structure
//             int dim            - current dimension
//             size_t skip[7]     - initial skip for all dimensions
//             int ranges[7][2]   - index range for all dimensions
//             size_t chunk       - how much data to read at a time
//             void * &data       - pointer to where the data goes
//             char *errMsg       - error message string
//   
// Returns: either bytes read, or 0 to indicate error
//   
// Notes: 
//
//****************************************************************************

size_t ReadSkip(znzFile zfp, nifti_image *nis, int dim, size_t skip[7],
				int ranges[7][2], size_t chunk, void * &data, char *errMsg)
	
{
	size_t bytesRead = 0;
	size_t endSkip = 0;
	
	// Skip space, if needed
	
	if ( skip[dim] > 0 )
		if ( znzseek(zfp, skip[dim], SEEK_CUR ) != 0 )
		{
			sprintf(errMsg, "ReadSkip: seek error in file %s", nis->iname);
			return 0;
		}

	// If this is innermost dimension, read and return, otherwise proceed down

	if ( dim > 0 )
	{
		for (int pind = ranges[dim][0]; pind <= ranges[dim][1]; ++pind)
		{
			size_t thisRead = ReadSkip(zfp, nis, dim-1, skip, ranges, chunk, data, errMsg);
			if ( thisRead == 0 )
			{
				sprintf(errMsg, "ReadSkip: read error in file %s", nis->iname);
				return 0;
			}
			bytesRead += thisRead;
		}

		// Skip to the end of current dimension

		endSkip = nis->dim[dim + 1] - ranges[dim][1];
		if ( endSkip > 0 )
		{
			for ( int lowerDim = 1; lowerDim <= dim; ++lowerDim )
				endSkip *= nis->dim[lowerDim];
			endSkip *= nis->nbyper;
			if ( znzseek(zfp, endSkip, SEEK_CUR ) != 0 )
			{
				sprintf(errMsg, "ReadSkip: seek error in file %s", nis->iname);
				return 0;
			}
		}

		return bytesRead;
	}
	
	if ( nifti_read_buffer(zfp, data, chunk, nis) != chunk )
	{
		sprintf(errMsg, "ReadSkip: something terribly wrong happened in reading data from %s.\nGiving up.", nis->fname);
		return true;
	}

	data = (void *)(size_t(data) + chunk);

	// Skip to the next "line"

	endSkip = nis->dim[1] - ranges[0][1];
	if ( endSkip > 0 )
	{
		endSkip *= nis->nbyper;
		if ( znzseek(zfp, endSkip, SEEK_CUR ) != 0 )
		{
			sprintf(errMsg, "ReadSkip: seek error in file %s", nis->iname);
			return 0;
		}
	}

	return chunk;	
}

//****************************************************************************
//
// Purpose: Loop trough dimensions recursively, skipping needed space and
// writing desired data.
//   
// Parameters: znzFile zfp        - pointer to file
//             nifti_image *nis   - pointer to data structure
//             int dim            - current dimension
//             size_t skip[7]     - initial skip for all dimensions
//             int ranges[7][2]   - index range for all dimensions
//             size_t chunk       - how much data to read at a time
//             void * &data       - pointer to where the data goes
//             char *errMsg       - error message string
//   
// Returns: either bytes written, or 0 to indicate error
//   
// Notes: 
//
//****************************************************************************

size_t WriteSkip(znzFile zfp, nifti_image *nis, int dim, size_t skip[7],
				int ranges[7][2], size_t chunk, void * &data, char *errMsg)
	
{
	size_t bytesWritten = 0;
	size_t endSkip = 0;
	
	// Skip space, if needed
	
	if ( skip[dim] > 0 )
		if ( znzseek(zfp, skip[dim], SEEK_CUR ) != 0 )
		{
			sprintf(errMsg, "WriteSkip: seek error in file %s", nis->iname);
			return 0;
		}

	// If this is innermost dimension, read and return, otherwise proceed down

	if ( dim > 0 )
	{
		for (int pind = ranges[dim][0]; pind <= ranges[dim][1]; ++pind)
		{
			size_t thisWrite = WriteSkip(zfp, nis, dim-1, skip, ranges, chunk, data, errMsg);
			if ( thisWrite == 0 )
			{
				sprintf(errMsg, "WriteSkip: write error in file %s", nis->iname);
				return 0;
			}
			bytesWritten += thisWrite;
		}

		// Skip to the end of current dimension

		endSkip = nis->dim[dim + 1] - ranges[dim][1];
		if ( endSkip > 0 )
		{
			for ( int lowerDim = 1; lowerDim <= dim; ++lowerDim )
				endSkip *= nis->dim[lowerDim];
			endSkip *= nis->nbyper;
			if ( znzseek(zfp, endSkip, SEEK_CUR ) != 0 )
			{
				sprintf(errMsg, "WriteSkip: seek error in file %s", nis->iname);
				return 0;
			}
		}

		return bytesWritten;
	}
	
	if ( nifti_write_buffer(zfp, data, chunk) != chunk )
	{
		sprintf(errMsg, "WriteSkip: something terribly wrong happened in writing data to %s.\nGiving up.", nis->fname);
		return true;
	}

	data = (void *)(size_t(data) + chunk);

	// Skip to the next "line"

	endSkip = nis->dim[1] - ranges[0][1];
	if ( endSkip > 0 )
	{
		endSkip *= nis->nbyper;
		if ( znzseek(zfp, endSkip, SEEK_CUR ) != 0 )
		{
			sprintf(errMsg, "WriteSkip: seek error in file %s", nis->iname);
			return 0;
		}
	}

	return chunk;	
}

//****************************************************************************
//
// Purpose: Compute the ranges and amount to skip for each
//   
// Parameters: const mxArray *rangePars - passed from matlab
//             nifti_image *nis         - header info
//             int ranges[7][2]         - computed ranges
//             size_t skip[7]           - computed bytes to skip in each dimensin
//   
// Returns: int numPix - total number of pixels
//   
// Notes: 
//
//****************************************************************************

int ComputeSkipRanges(const mxArray *rangePars, nifti_image *nis, int ranges[7][2], size_t skip[7])
{
	int numPix = 1;
	size_t totalSize = 0;
		
	int numDims = int(mxGetNumberOfElements(rangePars));
	for (int dim = 0; dim < numDims; ++dim)
	{
		const mxArray *element = mxGetCell(rangePars, dim);
		long value[2];
		mwSize numElements = mxGetNumberOfElements(element);

		if ( numElements == 0 )
		{
			value[0] = 1;
			value[1] = nis->dim[dim + 1];
		}
		else
		{
			switch ( mxGetClassID(element) )
			{
				case mxINT32_CLASS:
				{
					int *lp = (int *)mxGetPr(element);
					value[0] = lp[0];
					if ( numElements == 1 )
						value[1] = value[0];
					else
						value[1] = lp[1];
					break;
				}
				
				case mxSINGLE_CLASS:
				{
					float *fp = (float *)mxGetPr(element);
					value[0] = long(fp[0]);
					if ( numElements == 1 )
						value[1] = value[0];
					else
						value[1] = long(fp[1]);
					break;
				}
				
				case mxDOUBLE_CLASS:
				{
					double *dp = mxGetPr(element);
					value[0] = long(dp[0]);
					if ( numElements == 1 )
						value[1] = value[0];
					else
						value[1] = long(dp[1]);
					break;
				}
			}
		}

		numPix *= value[1] - value[0] + 1;
			
		ranges[dim][0] = value[0];
		ranges[dim][1] = value[1];
			
		if ( dim == 0 )
		{
			skip[dim] = (ranges[0][0] - 1) * nis->nbyper;
			totalSize = nis->dim[1] * nis->nbyper;
		}
		else
		{
			skip[dim] = (ranges[dim][0] - 1) * totalSize;
			totalSize *= nis->dim[dim + 1];
		}
	}

	// Load remaining values from header

	for (int dim = numDims; dim < nis->ndim; ++dim)
	{
		ranges[dim][0] = 1;
		ranges[dim][1] = nis->dim[dim + 1];

		numPix *= nis->dim[dim + 1];
			
		skip[dim] = (ranges[dim][0] - 1) * totalSize;
		totalSize *= nis->dim[dim + 1];
	}
		
	return numPix;
}
