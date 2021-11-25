#!/bin/bash

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Create the folder structure used for esm-msa
# --------------------------------------------
DIRECTORY=$1
SUB_DIR_UNICLUST="$1"
SUB_DIR_UNICLUST+="/databases"

SUB_DIR_MSAS="$1"
SUB_DIR_MSAS+="/msas"

SUB_DIR_FASTAS="$1"
SUB_DIR_FASTAS+="/msa_fastas"

SUB_DIR_REPRS="$1"
SUB_DIR_REPRS+="/esm_msa_reprs"

if [[ -d "${DIRECTORY}" && ! -L "${DIRECTORY}" ]] ; then
    	echo "Directory found"

	if [[ -d "${SUB_DIR_UNICLUST}" && ! -L "${SUB_DIR_UNICLUST}" ]] ; then
		echo "$SUB_DIR_UNICLUST was found"

	else
	       mkdir -p "$SUB_DIR_UNICLUST"
	       echo "$SUB_DIR_UNICLUST was created"
	fi 

	if [[ -d "${SUB_DIR_MSAS}" && ! -L "${SUB_DIR_MSAS}" ]] ; then
                echo "$SUB_DIR_MSAS was found"

	else
               mkdir -p "$SUB_DIR_MSAS"
	       echo "$SUB_DIR_MSAS was created"
	fi
	
	if [[ -d "${SUB_DIR_FASTAS}" && ! -L "${SUB_DIR_FASTAS}" ]] ; then
                echo "$SUB_DIR_FASTAS was found"
        else
               mkdir -p "$SUB_DIR_FASTAS"     
	       echo "$SUB_DIR_FASTAS was created"
	fi

	if [[ -d "${SUB_DIR_REPRS}" && ! -L "${SUB_DIR_REPRS}" ]] ; then
                echo "$SUB_DIR_REPRS was found"
        else
               mkdir -p "$SUB_DIR_REPRS"
	       echo "$SUB_DIR_REPRS was created"
        fi
else
	mkdir -p "$SUB_DIR_UNICLUST"
	echo "$SUB_DIR_UNICLUST was created"
	mkdir -p "$SUB_DIR_MSAS"
	echo "$SUB_DIR_MSAS was created"
	mkdir -p "$SUB_DIR_FASTAS"
	echo "$SUB_DIR_FASTAS was created"
	mkdir -p "$SUB_DIR_REPRS"
	echo "$SUB_DIR_REPRS was created"
fi


# Run the uniclust_download script and provided it run successfully, i.e. returned 0, create the docker image
# -----------------------------------------------------------------------------------------------------------

var=$(python3 ../scripts/uniclust_download.py "$DIRECTORY")
char=${var: -1}

if [ "$char" = '0' ]; then 	

        docker pull ghcr.io/peptoneinc/msa-gen-adopt:latest
        docker run -it -d --name=msa-gen-adopt-beta -v "$DIRECTORY":/work ghcr.io/peptoneinc/msa-gen-adopt:latest
        echo "Docker container is up and running. Go ahead..."  
else
	
	echo "Issues were found in the Uniclust dataset. Please empty the folder $SUB_DIR_UNICLUST and re-run this routine."
fi

