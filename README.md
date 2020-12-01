## lsgkm-SVR: Extension of lsgkm to regression, with explanations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4300866.svg)](https://doi.org/10.5281/zenodo.4300866)

gkm-SVM, a sequence-based method for predicting regulatory DNA elements, is a useful tool for studying gene regulatory mechanisms. `LS-GKM`, offers much better scalability and provides further advanced gapped *k*-mer based kernel functions. This repository extends the lsgkm functionality further to support regression. It builds off of kundajelab/lsgkm (which has gkmexplain functionality for interpreting the predictions), which in turn builds off of Dongwon-Lee/lsgkm (the original lsgkm repository).

I (Avanti Shrikumar) created a separate repository (rather than forking Dongwon-Lee/lsgkm) so that users can post github issues to this repository (it is not possible to post github issues to forked repositories).

The API for this tool is backwards-compatible with the original lsgkm implementation. See [this Colab notebook](https://colab.research.google.com/github/kundajelab/rlsgkm/blob/master/examples/Demo_Regression_on_Simulated_Data.ipynb) for an example tutorial.
### Citation

*Please cite this repository if you use the **regression functionality** in your research:*

* https://doi.org/10.5281/zenodo.4300866

*Please cite the following paper if you use gkmexplain in your research:*

* Shrikumar, A.\*†, Prakash, E.†, Kundaje, A*. GkmExplain: fast and accurate interpretation of nonlinear gapped k-mer SVMs. Bioinformatics, Volume 35, Issue 14, July 2019, Pages i173–i182 (2019). doi:10.1093/bioinformatics/btz322 *\* Co-corresponding authors* *† Co-first authors*. Please refer to https://github.com/kundajelab/gkmexplain for code to replicate the analysis in the paper.

*Please cite the following papers if you use LS-GKM in your research:*

* Ghandi, M.†, Lee, D.†, Mohammad-Noori, M. & Beer, M. A. Enhanced Regulatory Sequence Prediction Using Gapped k-mer Features. PLoS Comput Biol 10, e1003711 (2014). doi:10.1371/journal.pcbi.1003711 *† Co-first authors*

* Lee, D. LS-GKM: A new gkm-SVM for large-scale Datasets. Bioinformatics btw142 (2016). doi:10.1093/bioinformatics/btw142


### Installation

After downloading and extracting the source codes, type:

    $ cd src
    $ make 

If successful, You should be able to find the following executables in the current (src) directory:

    gkmtrain
    gkmpredict
    gkmexplain

`make install` will simply copy these executables to the `../bin` direcory


### Tutorial

We introduce the users to the basic workflow of `LS-GKM`.  Please refer to help messages 
for more detailed information of each program.  You can access to it by running the programs 
without any argument/parameter.
  

#### Training of LS-GKM

You train a SVM classifier/regressor using `gkmtrain`.

If doing classification, it takes three arguments; 
positive sequence file, negative sequence file, and prefix of output.

If doing regression, it again takes three arguments, but they are
the sequences file, the labels, and the prefix of the output


    Usage: gkmtrain [options] <file1> <file2> <outprefix>

     train gkm-SVM using libSVM

    Arguments:
     file1: if classification: positive sequence file (FASTA format). If regression: all sequences file (FASTA format)
     file2: if classification: negative sequence file (FASTA format). If regression: corresponding labels (one per line)
     outprefix: prefix of output file(s) <outprefix>.model.txt or
                <outprefix>.cvpred.txt

    Options:
     -y <0 ~ 4>   set svm type (default: 0 C_SVC)
                    0 -- C_SVC
                    1 -- NU_SVC (untested)
                    2 -- ONE_CLASS (untested)
                    3 -- EPSILON_SVR
                    4 -- NU_SVR (untested)
     -t <0 ~ 6>   set kernel function (default: 4 wgkm)
                  NOTE: RBF kernels (3, 5 and 6) work best with -c 10 -g 2
                    0 -- gapped-kmer
                    1 -- estimated l-mer with full filter
                    2 -- estimated l-mer with truncated filter (gkm)
                    3 -- (truncated l-mer) gkm + RBF (gkmrbf)
                    4 -- (truncated l-mer) gkm + center weighted (wgkm)
                         [weight = max(M, floor(M*exp(-ln(2)*D/H)+1))]
                    5 -- (truncated l-mer) gkm + center weighted + RBF (wgkmrbf)
                    6 -- gapped-kmer + RBF
     -l <int>     set word length, 3<=l<=12 (default: 11)
     -k <int>     set number of informative column, k<=l (default: 7)
     -d <int>     set maximum number of mismatches to consider, d<=4 (default: 3)
     -g <float>   set gamma for RBF kernel. -t 3 or 5 or 6 only (default: 1.0)
     -M <int>     set the initial value (M) of the exponential decay function
                  for wgkm-kernels. max=255, -t 4 or 5 only (default: 50)
     -H <float>   set the half-life parameter (H) that is the distance (D) required
                  to fall to half of its initial value in the exponential decay
                  function for wgkm-kernels. -t 4 or 5 only (default: 50)
     -c <float>   set the regularization parameter SVM-C (default: 1.0)
     -e <float>   set the precision parameter epsilon (default: 0.001)
     -p <float>   set the SVR epsilon (default: 0.1)
     -w <float>   set the parameter SVM-C to w*C for the positive set (default: 1.0)
     -m <float>   set cache memory size in MB (default: 100.0)
                  NOTE: Large cache signifcantly reduces runtime. >4Gb is recommended
     -s           if set, use the shrinking heuristics
     -x <int>     set N-fold cross validation mode (default: no cross validation)
     -i <int>     run i-th cross validation only 1<=i<=ncv (default: all)
     -r <int>     set random seed for shuffling in cross validation mode (default: 1)
     -v <0 ~ 4>   set the level of verbosity (default: 2)
                    0 -- error msgs only (ERROR)
                    1 -- warning msgs (WARN)
                    2 -- progress msgs at coarse-grained level (INFO)
                    3 -- progress msgs at fine-grained level (DEBUG)
                    4 -- progress msgs at finer-grained level (TRACE)
    -T <1|4|16>   set the number of threads for parallel calculation, 1, 4, or 16
                     (default: 1)


First try to train a model using simple test files. Type the following command in `tests/` directory:

    $ ../bin/gkmtrain wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.tr.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.neg.tr.fa test_gkmtrain

It will generate `test_gkmtrain.model.txt`, which will then be used for scoring of 
any DNA sequences as described below.  This result should be the same as `wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.model.txt`

You can also perform cross-validation (CV) analysis with `-x <N>` option. For example,
the following command will perform 5-fold CV. 

    $ ../bin/gkmtrain -x 5 wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.tr.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.neg.tr.fa test_gkmtrain

The result will be stored in `test_gkmtrain.cvpred.txt`, and this should be the same as 
`wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.cvpred.txt`

Please note that it will run SVM training *N* times, which can take time if training 
sets are large.  In this case, you can perform CV analysis on a specific set 
by using `-i <I>` option for parallel runnings. The output will be `<outprefix>.cvpred.<I>.txt`

The format of the cvpred file is as follows:
  
    [sequenceid] [SVM score] [label] [CV-set]
    ...


#### Scoring DNA sequence using gkm-SVM

You use `gkmpredict` to score any set of sequences.

    Usage: gkmpredict [options] <test_seqfile> <model_file> <output_file>

     score test sequences using trained gkm-SVM

    Arguments:
     test_seqfile: sequence file for test (fasta format)
     model_file: output of gkmtrain
     output_file: name of output file

    Options:
     -v <0|1|2|3|4>  set the level of verbosity (default: 2)
                       0 -- error msgs only (ERROR)
                       1 -- warning msgs (WARN)
                       2 -- progress msgs at coarse-grained level (INFO)
                       3 -- progress msgs at fine-grained level (DEBUG)
                       4 -- progress msgs at finer-grained level (TRACE)
    -T <1|4|16>      set the number of threads for parallel calculation, 1, 4, or 16
                     (default: 1)

Here, you will try to score the positive and the negative test sequences. Type:

    $ ../bin/gkmpredict wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.test.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.model.txt test_gkmpredict.txt
    $ ../bin/gkmpredict wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.neg.test.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.model.txt test_gkmpredict.neg.txt

#### Evaluating prediction quality 

You may evaluate the model prediction quality as follows:

    $ python scripts/lsgkm_eval.py -p test_gkmpredict.txt -n test_gkmpredict.neg.txt  

This will output metrics such as accuracy (at threshold 0), AUROC and AUPRC. The threshold for accuracy can be changed with the `-t` flag.

If you wish to generate a simple GC% baseline predictions, you can run:

    $ python scripts/gc_predictor.py -fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.test.fa -o test_gc_content.txt 
    $ python scripts/gc_predictor.py -fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.neg.test.fa -o test_gc_content.neg.txt

To compute evaluation metrics on the GC% predictions, you may use the same commands as above.

#### Explaining predictions with gkmexplain

You use `gkmexplain` to explain the predictions on sequences

    Usage: gkmexplain [options] <test_seqfile> <model_file> <output_file>

     explain prediction on test sequences using trained gkm-SVM

    Arguments:
     test_seqfile: sequence file for test (fasta format)
     model_file: output of gkmtrain
     output_file: name of output file

    Options:
     -v <0|1|2|3|4>  set the level of verbosity (default: 2)
                       0 -- error msgs only (ERROR)
                       1 -- warning msgs (WARN)
                       2 -- progress msgs at coarse-grained level (INFO)
                       3 -- progress msgs at fine-grained level (DEBUG)
                       4 -- progress msgs at finer-grained level (TRACE)
     -m <0|1>  set the explanation mode (default: 0)
                       0 -- importance scores
                       1 -- hypothetical importance scores (considering lmers with d mismatches)
                       2 -- hypothetical importance scores (considering d+1 mismatches)
                       3 -- perturbation effect estimation (considering lmers with d mismatches)
                       4 -- perturbation effect estimation (considering d+1 mismatches)
                       5 -- score perturbations for only the central position in the region

Here, you will explain the predictions on the positive sequences. Type:

    $ ../bin/gkmexplain wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.test.fa wgEncodeSydhTfbsGm12878Nfe2hStdAlnRep0.model.txt test_gkmexplain.txt

The output file will have three columns corresponding to "region id", "score from gkmpredict", and "explanation". The "explanation" is stored in a format consisting of sets of four values separated by semilcolons. Each group corresponds to a position in the original input sequence. The four values within a group correspond to the bases ACGT. If using the explanation mode of 0 (i.e. "importance scores"), all but one of the bases will have a score of 0; the base with a nonzero score corresponds to the base that was present in the original input sequence. Hypothetical importance scores (generated by modes 1 and 2) produce estimates of what the importance score would be if a different base were present in the underlying sequence; when looking at hypothetical importance scores, multiple bases at a given position will likely have nonzero scores.

Please see the colab example notebooks at https://github.com/kundajelab/gkmexplain for code demonstrating how to visualize the resulting scores.

Although I (Av Shrikumar) haven't implemented support for multiple threads, the process of generating explanations can be parallelized by sequence; see [here](scripts/parallelize_gkmexplain.sh) for a script to do that parallelization, written by Anna Shcherbina.



#### Generating weight files for deltaSVM

You need to generate all possible non-redundant *k*-mers using the Python script
`scripts/nrkmers.py`.  Then, you score them using `gkmpredict` as described above. 
The output of `lgkmpredict` can be directly used by the deltaSVM script `deltasvm.pl`
available from our deltasvm website.

#### Contact info

Please email Dongwon Lee (dwlee AT jhu DOT edu) if you have any questions about lsgkm (excluding regression functionality).
Please contact Avanti Shrikumar (avanti dot shrikumar at gmail) and Anshul Kundaje (anshul@kundaje.net) if you have questions about gkmexplain or the regression functionality.
