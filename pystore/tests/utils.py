#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for testing pystore package.
"""


import attr


@attr.s(kw_only=True)
class ClassToBeWritten:
    """
    This class is used only for testing purposes.
    """

    a = attr.ib(type=float)
    b = attr.ib(type=float)

    def __eq__(self, other: "ClassToBeWritten"):
        if other.a == self.a and other.b == self.b:
            return True
        else:
            return False
