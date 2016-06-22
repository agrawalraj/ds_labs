
"""
Autograder for lab3_ex1.py. If you have already attempted this lab, we 
only record the max score in the progress_dict. 
 
"""
#Maybe have a make file that will run all solution scripts - sets up everything 

import numpy as np
import os

def printSummary(is_right):
	if is_right:
	    print('You got 100% . Good job!')
	    return 100
	else:
		print('Look over your code again.')
		return 0

def fileExists(path):
    return os.path.isfile(path) 

def ranScript(did_run, lab_num, ex_num):
    lab_str = 'lab' + str(lab_num) + '_' + str(ex_num)  
    if did_run == False:
        print('No saved file exists: Try running: Python ' + lab_str 'in the' +
             'lab ' + str(lab_num) + ' folder.')

def updateDictionary(key, score):
    if progress_dict.contains(key):
        progress_dict[key] = max(progress_dict[key], score)
    else:
        progress_dict[key] = score

def runScript(path):
    os.system(path)


def deleteFiles(path_lst):
    for path in path_lst:
        os.remove(path)

if __name__ == '__main__':

	user_path = '../temp_saved_work/lab3_ex1_user.npy'
    did_run = fileExists(user_path)
    ranScript(did_run, 3, 1)

    if did_run:
        user_response = np.load(user_path)  
    
    solution_path = '../temp_saved_work/lab3_ex1_sol.npy'
    did_run_sol = fileExists(solution_path)
    
    if did_run_sol == False:
        runScript('../solutions/lab3_ex1_sol.py')
    
    solutions = np.load(solution_path) #if this error need to call solutions script 

    work_progress_path = '../user_progress/completed_work'
    progress_dict = np.load(work_progress_path)

    is_right = np.array_equal(user_response, solutions)
    score = printSummary(is_right)

    updateDictionary('lab3_ex1', score)
