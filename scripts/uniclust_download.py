# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import requests
import tarfile 
import argparse
import os 
import sys 

args_parser = argparse.ArgumentParser(description='Get the path to adopt_esm_msa folder that contains also the uniclust dataset')

# Add the arguments
args_parser.add_argument('adopt_esm_msa_path', 
                        action='store',
                        type=str 
                        )
                        
# Execute the parse_args() method
args = args_parser.parse_args()
adopt_esm_msa_path = args.adopt_esm_msa_path


# Function that makes sure that Uniclust exists in the right location 
def get_uniclust(main_folder):

    # Declare the folder for the uniclust dataset
    uniclust_path = f'{adopt_esm_msa_path}/databases'

    go_ahead = 1 

    # Saving Uniclust locally 
    url = 'http://gwdu111.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz'
    filename = os.path.basename(url)
    
    # Check if the tarfile exists already in the main folder and the databases is not empty 
    if os.path.isfile(f'{main_folder}/{filename}') and len(os.listdir(uniclust_path))!=0:
        go_ahead=0
        print('Uniclust installation found.')
        return go_ahead
    else:
        # Download Uniclust
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        
        print(f'Downloading Uniclust into {uniclust_path}...')
        file.extractall(path=uniclust_path)
        go_ahead = 0

    return go_ahead

def main():

    go_ahead = get_uniclust(adopt_esm_msa_path)
    return go_ahead 

if __name__ == "__main__":
    sys.stdout.write(str(main()) + '\n')

