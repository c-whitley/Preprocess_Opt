import pandas as pd 
import numpy as np

import os
import argparse
import sys


arg_parser = argparse.ArgumentParser()


function_map = {''}


class Pipeline_Job():

    def __init__(self, func, whole_data, job_name):

        self.job_name = job_name
        self.function = func
        self.initialised = False
        self.whole_data = whole_data

        print(f"Job created: {self.job_name}")

        # If no job name is given, just use the time
        if self.job_name == None:
            self.job_name = str(datetime.datetime.now())[:-7]

        # Name of the directory for the job
        self.file_name = os.path.join(os.getcwd(), self.job_name)
        
        if not os.path.exists(self.file_name):
            os.mkdir(self.file_name)
        
        print(f"Directory: {self.file_name}")


    def prepare(self, data, n_pipe = 1): #n_pipe is the number of pipelines per pickle.

        if not os.path.exists(os.path.join(self.file_name, "input")):
            os.mkdir(os.path.join(self.file_name, "input"))


        for i, address in tqdm(enumerate(self.whole_data)):

            pipe = pp.Pipeline(address)
            
            pipe_filename = os.path.join(self.file_name, "input", "input{}".format(i))

            dill.dump(pipe, open(pipe_filename, 'wb'), protocol = 4)

            if i > 10:
               break

        #data.to_pickle(os.path.join(self.file_name, "input", "data"), protocol = 4)
        #dill.dump(data.values, open(os.path.join(self.file_name, "input", "data"),'wb'))
        #ind = data.index.names
        #data.reset_index().to_json(os.path.join(self.file_name, "input", "data.json"), orient = 'split')
        #pickle.dump(ind, open(os.path.join(self.file_name, "input", "ind"), "wb"), protocol = 4)



        os.system("sh zip_bash.sh {}".format(self.job_name))
        i += 1

        self.n_jobs = i
        self.initialised = True