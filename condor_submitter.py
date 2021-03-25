import sys
sys.path.append("/condor_data/sgbellis/modules")
#from Preprocess_Opt import preprocessing_pipeline as pp 
import pandas as pd 
import numpy as np
import pickle
import os
import sys
from tqdm.notebook import tqdm
import dill
import glob
import matplotlib.pyplot as plt

import shutil


class Condor_Job_Pipeline: 

    def __init__(self, name, iterable, function, dependencies, jobdir=None):
        """Class to submit a condor pipeline 

        Args:
            name (str): Name of the function to be used on the condor node.
            iterable (iterable): Iterable containing constituent part to be sent to each node invdividually.
            function ([type]): Python file containing the function.
            dependencies ([type]): List of directories containing dependencies for the job.
            jobdir ([type], optional): Optionally nominate a directory for the job to go to. Defaults to None.
        """     

        self.name = name
        self.iterable = iterable
        self.function = function
        self.dependencies = dependencies
        self.jobdir = jobdir

        self.user = input("Username: ")

        if self.jobdir == None:
            # If no job directory is nominated just infer one from the job name.
            self.jobdir = os.path.join("/condor_data", self.user, "jobs", name)

        if not(os.path.exists(self.jobdir)):
            os.mkdir(self.jobdir)
            print("directory created at {}".format(self.jobdir))

        else: 
            print("caution: directory already exists")
        print(os.getcwd())
    def prepare(self, X, n = None):

        for i, address in enumerate(self.iterable):

            with open(os.path.join(self.jobdir, "pipe{}".format(i)), "wb") as f:
                pickle.dump(address, f, protocol=4)

            if n is not(None):
                if i == n: 
                    break
        
        self.n_jobs = i + 1
        #ind = X.index.names
        #pickle.dump(ind, open(os.path.join(self.jobdir, "ind"), "wb"), protocol = 4)
        #X.reset_index().to_json(os.path.join(self.jobdir, "data.json"), orient = "split")
        
        # Make temporary directory
        os.mkdir("Temp")

        # Make the submission file
        self.submission_file()

        for dependency in self.dependencies:
            os.system(f'cp -R {dependency} {self.jobdir}/Temp')              
            print(f'Copied {dependency} to {self.jobdir}/Temp')

        # Copy data
        X.to_hdf(os.path.join(self.jobdir, "data.h5"), mode = "w", key = "df")
        #os.system("sh dependency_zipper.sh {}/".format(self.jobdir))
        # Copy the main function file to the job directory
        os.system(f"cp ./{self.function} {self.jobdir}/")

        # Turn temp directory into SFX
        os.system(f'7z a -sfx7zWindows.sfx {self.jobdir}/dependencies.exe {self.jobdir}/Temp/')
        # Link exe file with zip
        os.system(f"ln -s {self.jobdir}/dependencies.exe {self.jobdir}/dependencies.zip")
        # Remove temp directory
        shutil.rmtree("Temp")

    
    def submit(self):

        os.chdir(self.jobdir)
        os.system("python_submit condor_scorer -N")

        # Edit .bat file to run self-extracting directory
        with open("condor_scorer.bat","r+") as f:

            lines = f.readlines()
            lines[33] = "dependencies.zip\n"
            f.writelines(lines)
        
        #os.system("condor_submit condor_scorer.sub")
        stdout=os.popen("condor_submit condor_scorer.sub").read()        
        print(stdout)

    def submission_file(self):

        with open(os.path.join(self.jobdir, "condor_scorer"), "w") as file:
            file.write(f"python_script = {self.function}\n")
            file.write(f"python_version = python_3.7.4\n")
            file.write('input_files = data.h5, dependencies.exe\n')
            file.write("indexed_input_files = pipe\n")
            file.write("indexed_output_files = output\n")
            file.write("indexed_stdout = stdout.txt\n")
            file.write("indexed_stderr = stderr.txt\n")
            file.write("log = log.txt\n")
            file.write(f"total_jobs = {str(self.n_jobs)}\n")

    def retrieve(self): 

        self.results = []

        for file in tqdm(glob.glob(self.jobdir + "/output*")): 
            try:
                temp = dill.load(open(file, "rb"))
                self.results.append(temp)
            except:
                print("{} is empty!".format(file))

    def result_plotter(self, **kwargs):
        weights = kwargs.get("weights", [2,1])
        opt_array = np.zeros((2, len(self.results)))
        for i,results in enumerate(self.results): 
           
            index = results.opt_func.best_index_
            opt_array[0, i] = results.opt_func.cv_results_['mean_test_score'][index]
            opt_array[1, i] = results.opt_func.cv_results_['std_test_score'][index]
            #opt_array[2, i] = (weights[0]*opt_array[0,i] + weights[1]*opt_array[1, i])/(weights[0]+weights[1])
        weighted_score = np.divide((weights[0]*opt_array[0,:] + weights[1]*opt_array[1, :]),(weights[0]+weights[1]))
        
        fig1,ax1 = plt.subplots()
        fig2,ax2 = plt.subplots()
        
        ax1.plot(np.arange(1, len(self.results) + 1), weighted_score)
        ax1.set_xlabel("iteration")
        ax1.set_ylabel("Score")
        ax1.set_title(self.name)

        ax2.errorbar(np.arange(1, len(self.results) + 1), opt_array[0,:], opt_array[1,:])
        ax2.set_xlabel("iteration")
        ax2.set_ylabel("Weighted Score")
        ax2.set_title(self.name)

        self.best_job = self.results[np.argsort(weighted_score)[::-1]].opt_func
        #print(self.best_job.best_estimator_)

class Condor_Job_Mie:

    def __init__(self, name, n_jobs, function):

        self.name = name
        self.jobdir = os.path.join("/condor_data", "sgbellis", "jobs", name)
        self.n_jobs = n_jobs
        self.function = function

        if not(os.path.exists(self.jobdir)):
            os.mkdir(self.jobdir)
            print("directory created at {}".format(self.jobdir))
        else: 
            print("caution: directory already exists")
        print(os.getcwd())

    def prepare(self, X):

        #splitting dataframe into n_jobs datasets
        self.X = X
        reference = self.X.mean(axis = 0)
        sub_df = np.array_split(self.X, self.n_jobs)
        self.IndexLabels = X.index

        for i,df in enumerate(tqdm(sub_df)):

            #with os.path.join(self.jobdir, "input{}".format(i)) as f:

            df.reset_index(drop = True).to_json(os.path.join(self.jobdir, "input{}".format(i)), orient = "split")
            #print("written file {}".format(i))

        pickle.dump(reference.values, open(os.path.join(self.jobdir, "reference"), "wb"), protocol=4)

        self.n_jobs = i+1
        os.chdir("/condor_data/sgbellis/dysplasia")
        os.system("sh dependency_zipper.sh {}/".format(self.jobdir))
        os.system("cp ./{} {}/".format(self.function, os.path.join(self.jobdir)))
        self.submission_file()
    def submit(self):


        os.chdir(self.jobdir)
        print("working in directory ", os.getcwd())
        os.system("ln -s dependencies.exe dependencies.zip")  
        os.system("python_submit mie_condor -N")
        f = open("mie_condor.bat","r")
        lines = f.readlines()
        lines[33] = "dependencies.zip\n"
        f = open("mie_condor.bat", "w")
        f.writelines(lines)
        f.close()        
        
    def submission_file(self):

        with open(os.path.join(self.jobdir, "mie_condor"), "w") as file:

            file.write(f"python_script = {self.function}\n")
            file.write(f"python_version = python_3.7.4\n")
            #file.write('input_files = data.h5, dependencies.zip \n')
            file.write('input_files = reference, dependencies.zip \n')

            file.write("indexed_input_files = input\n")
            file.write("indexed_output_files = output\n")
            file.write("indexed_stdout = stdout.txt\n")
            file.write("indexed_stderr = stderr.txt\n")
            file.write("log = log.txt\n")
            file.write(f"total_jobs = {self.n_jobs}\n")

    def retrieve(self):
        self.X_corr = pd.read_json(os.path.join(self.jobdir, "output0"), orient = "split")
        for i in tqdm(range(1,self.n_jobs)):

            temp = pd.read_json(os.path.join(self.jobdir, "output{}".format(i)), orient = "split")
            self.X_corr = self.X_corr.append(temp)

        return self.X_corr

            







        


