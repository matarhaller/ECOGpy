setwd('/home/knight/matar/PYTHON/ECOGpy/')
library(psych)
library(proxy)
source("R/EFA_Comparison.R")
library(reshape2)
library(ggplot2)
library(cluster)

param = 'maxes' #or lats_pro or means

DATADIR = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/ShadePlots_hclust/elecs/significance_windows/csv_files/' #csv files with values for each subj/task 
saveDir = paste('/home/knight/matar/MATLAB/DATA/Avgusta/PCA/Stats/PCA_', param, '/', sep = "")

d = read.table('test.subjects',sep = '_')

for (i in 13:nrow(d) ) {
#for (i in c(36,37,38,39,40,41)) {  
  subj = d$V1[i]
  task = d$V2[i]
  
  data = read.csv(paste(DATADIR, subj, '_', task, '_', param, '.csv', sep = ""), header = T, stringsAsFactors = F)
  data = data.matrix(data)
  
  #drop rows with NA values
  #data = data[complete.cases(data), ]
    
  efa = EFA.Comp.Data(Data=data, F.Max=10, Graph=T) #calculate number of components
  filename = paste(saveDir, subj, "_", task, '_efa.png',sep = "")
  dev.copy(png,filename)
  dev.off()
           
  fit = principal(data, nfactors = efa, rotate='none') #calculate PCA without rotation
  
  #write loadings csv
  df_loadings = cbind(data.frame(unclass(fit$loadings)), data.frame(fit$communality))
  df_SSloadings = data.frame(fit$values[1:efa])
  filename = paste(saveDir, subj, '_', task, '_loadings.csv',sep = "")
  write.table(df_loadings,file = filename, sep =",", row.names=TRUE, col.names = NA)
  filename = paste(saveDir, subj, '_', task, '_SSloadings.csv',sep = "")
  write.table(df_SSloadings,file = filename, sep =",", row.names=FALSE)

}
