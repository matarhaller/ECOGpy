setwd('/Users/matar/MATLAB/')
library(R.matlab)

subj = "ST27"
task = "DecisionVis"

DATADIR = paste('/Users/matar/cluster/MATLAB/DATA/Avgusta/data/', subj, sep = "")  

data = readMat(paste(DATADIR, '/', task, '_datasummary.mat', sep = ""))
attach(data) #srate, dur.elecs, peaks, latencies, latencies.norm, means, pksums, sums, rows

hist(latencies.norm[9,])

library(e1071)                    # load e1071 

duration = faithful$eruptions     # eruption durations 

skewness(duration) 