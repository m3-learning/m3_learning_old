"""
Created on Sun Jan 24 16:34:00 2021
@author: Alibek Kaliyev
"""

import h5py

# define a small function called 'print_tree' to look at the folder tree structure
def print_tree(parent):
    print(parent.name)
    if isinstance(parent, h5py.Group):
        for child in parent:
            print_tree(parent[child])


def make_group(base, group):
    """Utility function to add a group onto a h5_file, adds the dependency to not return and error if it already exists.

    Args:
        base (h5_file): base h5 file to add new group
        group (string): name of the 
    """
    try: 
        base.create_group(group)
    except:
        print('could not add group - it might already exist')
    
def make_dataset(base, dataset):
    try: 
        base.create_group(dataset)
    except:
        print('could not add dataset - it might already exist')
        
#TODO Add a utility to check that a dataset exists or not.