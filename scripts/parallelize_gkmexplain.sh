#!/usr/bin/env bash 

#This script was written by Anna Shcherbina
#It takes the four arguments listed below:
inputf=$1  ## input fasta file 
numchunks=$2 ## number of parallel chunks to split input fasta file into 
numthreads=$3 ## number of chunks to process in parallel with gkmexplain 
modelpath=$4 ## path to model to use with gkmexplain. 

#split $inputf fasta file into $numchunks fasta files 
split -t'>' -n $numchunks -d $inputf $inputf-

#run gkmexplain in parallel with $numthreads chunks processed in parallel using the model stored at $modelpath.
ls $inputf-* | xargs -I{} -n1 -P $numthreads  gkmexplain {} $modelpath {}.gkmexplain

#aggregate the results
for chunk in *.gkmexplain
do
    cat $chunk >> $inputf.gkmexplain.aggregated.txt
done
