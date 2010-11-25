#!/usr/bin/env python

import datetime
from Subject import *

# subject information
initials='JB'
firstName = 'Jessica'
standardFSID = 'JB_051110_12'
birthdate = datetime.date( 1989, 12, 4 )
labelFolderOfPreference = ''

presentSubject = Subject( initials, firstName, birthdate, standardFSID )

