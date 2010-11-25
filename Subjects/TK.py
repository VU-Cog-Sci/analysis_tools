#!/usr/bin/env python

import datetime
from Subject import *

# subject information
initials='TK'
firstName = 'tomas'
standardFSID = 'TK_091009tk'
birthdate = datetime.date( 1978, 10, 04 )
labelFolderOfPreference = '7T_Retino_labels'

presentSubject = subject( initials, firstName, birthdate, standardFSID )

