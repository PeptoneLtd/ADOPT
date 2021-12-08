#!/bin/bash

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# First - Check whether the container exists and running
# ------------------------------------------------------
result=$( docker ps -q -f name=msa-gen-adopt )
n_cores = $( grep -c ^processor /proc/cpuinfo )

if [[ -n "$result" ]]; then
	echo "Container exists"
	# Second - Copy the fasta file to the right folder
	docker cp "$1" msa-gen-adopt:/work/msa_fastas
	
	path_with_filename="$1"
	file_name="${path_with_filename##*/}"
	
	docker_fasta_path="/work/msa_fastas/"
	docker_fasta_path+="$file_name"

	# Third - Get the msas
	docker exec -it msa-gen-adopt \
        	bash -c "ffindex_from_fasta _fas.ff{data,index} $docker_fasta_path
      		hhblits_omp -i _fas -oa3m /work/res_a3m -d /work/databases/UniRef30_2020_06 -cpu $n_cores -n 3 -e 1e-3
        	ffindex_unpack /work/res_a3m.ff{data,index} /work/msas/ 
			rm _fas.ff{data,index} 
			rm /work/res_a3m.ff{data,index} "
		
		
        
else
	echo "No such container. Please create the container first."
fi

