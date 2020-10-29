import pandas as pd 
import numpy as np 

import argparse
import os
import sys


parser = argparse.ArgumentParser("CondorJob",
                                 "")

FUNCTION_MAP = {'top20' : my_top20_func,
                'listapps' : my_listapps_func }

parser.add_argument('command', choices=FUNCTION_MAP.keys())

args = parser.parse_args()

func = FUNCTION_MAP[args.command]
func()                                 

