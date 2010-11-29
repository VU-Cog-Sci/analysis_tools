#!/usr/bin/env python

import datetime
from Subject import *

# subject information
initials='DB'
firstName = 'd'
standardFSID = 'DB_290910db'
birthdate = datetime.date( 2010, 9, 29 )
labelFolderOfPreference = ''

presentSubject = subject( initials, firstName, birthdate, standardFSID, labelFolderOfPreference )

