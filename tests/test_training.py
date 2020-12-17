import unittest
from torchani import training



class TestTrainingUtils(unittest.TestCase):
    # Test that checks that the friendly constructor
    # reproduces the values from ANI1x with the correct parameters
    def testInsertInKey(self):
        d = {'A':1234, 'B':134, 'C': { 'D': 1378374, 'E': 23434}}
        d_ex = {'A':1234, 'B':134, 'C': { 'D': None, 'E': 23434}}
        training.insert_in_key(d, 'D', None)
        self.assertTrue(d_ex == d)
