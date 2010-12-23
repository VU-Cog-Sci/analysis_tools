#!/usr/bin/env python

import datetime
from Subject import *

# subject information
initials='JB'
firstName = 'Jan'
standardFSID = 'JB_090206jb'
birthdate = datetime.date( 1977, 12, 24 )
labelFolderOfPreference = ''

presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )

