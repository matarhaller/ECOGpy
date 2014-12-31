setwd('/home/knight/matar/PYTHON/ECOGpy/')
library(proxy)
library(reshape2)
library(psych)

params = c('maxes_rel','medians','stds', 'means', 'maxes')
#params = c('means')

DATADIR = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/Stats/outliers/for_PCA/unsmoothed/' #csv files with values for each subj/task with outlier rejection for PCA (all elecs have same number of trials, each feature has different number of trials)

d = read.table('test.subjects',sep = '_')

for (i in 1:nrow(d)) {
  for (p in params) {
    saveDir = paste('/home/knight/matar/MATLAB/DATA/Avgusta/PCA/Stats/Networks/unsmoothed/PCA_', p, '/', sep = "")
    
    subj = d$V1[i]
    task = d$V2[i]
        
    data = read.csv(paste(DATADIR, subj, '_', task, '_', p, '.csv', sep = ""), header = T, stringsAsFactors = F)
    data = data.matrix(data)
    
    #implement kaiser rule
    #fit = prcomp(data, scale=TRUE, retx = FALSE) #uses correlation matrix, no rotation
    #eigvalues = fit$sdev^2 #(sdev aka signular values, works bc sum of eigvalues = sum total of the variance)
    #num_pcs = sum(eigvalues>1)
    
    num_pcs = dim(data)[2] #start with num pcs == num elecs
    fit = principal(data, nfactors = num_pcs, rotate='none') #calculate PCA without rotation
    num_pcs = sum(fit$values>1) #kaiser rule
    fit = principal(data, nfactors = num_pcs, rotate = 'none', scores = TRUE)
    
    #write loadings, scores, summary csv
    filename = paste(saveDir, subj, '_', task, '_loadings.csv',sep = "")
    write.table(fit$loadings,file = filename, sep =",", row.names=TRUE, col.names = NA)
    filename = paste(saveDir, subj, '_', task, '_summary.txt',sep = "")
    capture.output(print(fit), file = filename)
    filename = paste(saveDir, subj, '_', task, '_scores.csv', sep = "")
    write.table(fit$scores, file = filename, sep= ",", row.names = FALSE)
  }
}