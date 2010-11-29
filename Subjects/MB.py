#!/usr/bin/env python

import datetime
from Subject import *

# subject information
initials='MB'
firstName = 'Mark'
standardFSID = 'MB_221110_12'
birthdate = datetime.date( 1988, 6, 9 )
labelFolderOfPreference = ''

presentSubject = Subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )

