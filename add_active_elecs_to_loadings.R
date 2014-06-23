#add active_elecs column to each _loadings.csv file

setwd('/home/knight/matar/PYTHON/ECOGpy/')
library(psych)
library(R.matlab)
library(proxy)
source("EFA_Comparison.R")
library(reshape2)
library(ggplot2)
library(cluster)

DATADIR = '/home/knight/matar/MATLAB/DATA/Avgusta/Subjs/'
saveDir = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/plots/'

d = read.table('test.subjects',sep = '_')
df = data.frame()

for (i in 1:nrow(d) ) {
  
  subj = d$V1[i]
  task = d$V2[i]
  
  data = readMat(paste(DATADIR, subj, '/', task, '/HG_elecMTX_mean.mat', sep = ""))
  active_elecs = as.vector(data$active.elecs)
  
  #write loadings csv
  filename = paste(saveDir, subj, '_', task, '_loadings.csv',sep = "")
  df_loadings = read.table(filename, header = TRUE, sep = ',')
  
  df_loadings = cbind(active_elecs, df_loadings)
  
  write.table(df_loadings,file = filename, sep =",", row.names=FALSE)
  
}