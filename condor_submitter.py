import sys
sys.path.append("/condor_data/sgbellis/modules")
from Preprocess_Opt import preprocessing_pipeline as pp 
import pandas as pd 
import pickle
import os
import sys
from tqdm.notebook import tqdm
import dill


class Condor_Job_Pipeline: 

    def __init__(self, name, iterable, function): 

        self.name = name
        self.iterable = iterable
        self.function = function
        #self.main = main
        #self.dependencies = dependencies

        self.jobdir = os.path.join("/condor_data", "sgbellis", "jobs", name)

        if not(os.path.exists(self.jobdir)):
            os.mkdir(self.jobdir)
            print("directory created at {}".format(self.jobdir))
        else: 
            print("caution: directory already exists")
    
    def prepare(self, X, n = None):

         pipe = pp.Pipeline_Opt()

         for i, address in enumerate(self.iterable):

             pipe.make_pipeline(address)

             with open(os.path.join(self.jobdir, "pipe{}".format(i)), "wb") as f:

                 dill.dump(pipe, f, protocol=4)
             if n is not(None):
                if i == n: 
                    break
         
         self.n_jobs = i + 1
         ind = X.index.names
         pickle.dump(ind, open(os.path.join(self.jobdir, "ind"), "wb"), protocol = 4)
         X.reset_index().to_json(os.path.join(self.jobdir, "data.json"), orient = "split")

         os.system("sh dependency_zipper.sh {}/".format(self.jobdir))
         os.system("cp ./{} {}/".format(self.function, os.path.join(self.jobdir)))
         self.submission_file()

    def submission_file(self):

        with open(os.path.join(self.jobdir, "condor_scorer"), "w") as file:

            file.write(f"python_script = {self.function}\n")
            file.write(f"python_version = python_3.7.4\n")
            file.write('input_files = data.json, ind, dependencies.zip \n')
            file.write("indexed_input_files = pipe\n")
            file.write("indexed_output_files = output\n")
            file.write("indexed_stdout = stdout.txt\n")
            file.write("indexed_stderr = stderr.txt\n")
            file.write("log = log.txt\n")
            file.write(f"total_jobs = {self.n_jobs}\n")






        


