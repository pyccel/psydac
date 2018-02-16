# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os


__author__="ARA"

# TODO: reimplement __eq__ and __ne__

DEBUG = False

def _singleton(cls):
    """
    A Class representing a singleton. Python does not offer this pattern.
    """
    instances = {}
    def getinstance():
        if cls not in instances:
            instances[cls] = cls() # Line 5
        return instances[cls]
    return getinstance

@_singleton
class common_obj(object):
    """
    A Class that represents the Fortran module as a unique object. This is why
    it is implemented as a singleton.
    """

    def __init__(self):
        """
        Creates the Fortran module
        """
        try:
            from clapp.core.django import m_django_core as _core
            self._core = _core
            self.initialize()
        except:
            raise ValueError("Error while importing pyfem. CLAPP will stop immediatly")

    def initialize(self):
        """
        Creates a dictionary for available ids.
        """
        # ...
        self._avail_ids = {}
        # ...

        # ... PLAF
        for name in ["matrix", "vector", "linear_solver", "ddm_parameters", "linear_operator"]:
            self._avail_ids[name] = list(range(1,self.n_max_core_array+1))[::-1]
        # ...

        # ... SPL
        for name in ["mapping"]:
            self._avail_ids[name] = list(range(1,self.n_max_core_array+1))[::-1]
        # ...

        # ... DISCO
        for name in ["field", "space", "derivatives", "collocation", "multilevels"]:
            self._avail_ids[name] = list(range(1,self.n_max_core_array+1))[::-1]
        # ...

        # ... FEMA
        for name in ["context", "assembler", "output", "coordinates"]:
            self._avail_ids[name] = list(range(1,self.n_max_core_array+1))[::-1]
        # ...

    @property
    def core(self):
        """
        Returns the Fortran module
        """
        return self._core

    @property
    def n_max_core_array(self):
        """
        Returns the maximum size of the internal array, each contains a list of
        Django objects.
        """
        return self.core.get_n_max_core_array()

    @property
    def avail_ids(self):
        """
        Returns the dictionary of available ids
        """
        return self._avail_ids

    def newID(self, name, verbose=False):
        """
        Returns a new ID for the object name

        name: str
          a Class name (usually a subdirectory from a given library) from Django

        verbose: bool
          allows prints if True.
        """
        if len(self.avail_ids[name]) == 0:
            print("Cannot allocate new fortran objects.")
            raise ValueError("You can overcome this limitation, by recompiling the python package with a bigger n_max_core_array value")
        _id = self.avail_ids[name].pop()
        if verbose:
            print(("[+] create "+name+" with ID: "+ str(_id)))
        return _id

    def freeID(self, name, ID, verbose=False):
        """
        Makes the ID available again

        name: str
          a Class name (usually a subdirectory from a given library) from Django

        ID: int
          ID of the object to be freed

        verbose: bool
          allows prints if True.
        """
        self._avail_ids[name].append(ID)
        L = np.asarray(self.avail_ids[name])
        L.sort()
        self._avail_ids[name] = list(L)[::-1]
        if verbose:
            print(("[-] free "+name+" with ID: "+ str(ID)))
#            print (">>> avail "+name+" ids: ", self.avail_ids[name])

class Basic(object):
    """
    Class representing the Parent of all classes within clapp, and coming from
    Fortran.
    """
    def __init__(self, *args, **kwargs):
        """
        Creates a new instance
        """
        self._com = common_obj()
        self._id = None

        self.infoData = {} # dictionary for info data. used for printing

    @property
    def com(self):
        """
        Returns the Fortran module
        """
        return self._com

    @property
    def id(self):
        """
        Returns the current object id.
        """
        return self._id

#    def writeInfoDataXML(self, doc, rootElt):
#
#        # Create the main <card> element
#        CURRENT_TAG = self.__class__.__name__
#        maincard = doc.createElement(CURRENT_TAG)
#        rootElt.appendChild(maincard)
#
#        for d in self.infoData:
#            TAG = d ; txt = self.infoData[d]
#            curElt = doc.createElement(TAG)
#            curText = doc.createTextNode(txt)
#            curElt.appendChild(curText)
#            maincard.appendChild(curElt)

    def __str__(self):
        """
        prints the current object
        """
        line = ""
        for d in self.infoData:
            line += str(d + " : " + str(self.infoData[d]))
            line += "\n"
        return line

    def __eq__(self, other):
        """
        Allows comparison of objects
        """
        return self.id == other.id

    def __ne__(self, other):
        """
        Allows comparison of objects
        """
        return self.id != other.id
