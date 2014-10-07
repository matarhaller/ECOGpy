setwd('/home/knight/matar/PYTHON/ECOGpy/')
#library(psych)
library(proxy)
#source("R/EFA_Comparison.R")
library(reshape2)
#library(ggplot2)
#library(cluster)
#library(lattice)

params = c('maxes_rel', 'lats_pro','medians','stds')

DATADIR = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/Stats/outliers/for_PCA/' #csv files with values for each subj/task with outlier rejection for PCA (all elecs have same number of trials, each feature has different number of trials)

d = read.table('test.subjects',sep = '_')

for (i in 1:nrow(d) ) {
  for (p in params) {
    saveDir = paste('/home/knight/matar/MATLAB/DATA/Avgusta/PCA/Stats/PCA_', p, '/', sep = "")
    
    subj = d$V1[i]
    task = d$V2[i]
        
    data = read.csv(paste(DATADIR, subj, '_', task, '_', p, '.csv', sep = ""), header = T, stringsAsFactors = F)
    data = data.matrix(data)
    
    #implement kaiser rule
    fit = prcomp(data, scale=TRUE, retx = FALSE) #uses correlation matrix, no rotation
    eigvalues = fit$sdev^2 #(sdev aka signular values, works bc sum of eigvalues = sum total of the variance)
    num_pcs = sum(eigvalues>1)
    
    #write loadings csv
    filename = paste(saveDir, subj, '_', task, '_loadings.csv',sep = "")
    write.table(fit$rotation[,1:num_pcs],file = filename, sep =",", row.names=TRUE, col.names = NA)
    filename = paste(saveDir, subj, '_', task, '_summary.csv',sep = "")
    write.table(summary(fit)$importance[,1:num_pcs],file = filename, sep =",", row.names = TRUE, col.names = NA)
  
  }
}