"""
This module tests the `phishbench` module
"""
import unittest
import phishbench


class PhishBenchTest(unittest.TestCase):

    def test_no_config(self):
        self.assertRaises(TypeError, phishbench.initialize, None)

    def test_no_output(self):
        self.assertRaises(TypeError, phishbench.initialize, "Test.ini", None)

    def test_missing_config(self):
        self.assertRaises(FileNotFoundError, phishbench.initialize, "ABCFNF.ini")
