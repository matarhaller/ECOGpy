#setwd('/Users/matar/Dropbox/PCA_elecs/')
setwd('/home/knight/matar/PYTHON/ECOGpy/')
library(psych)
library(R.matlab)
library(proxy)
source("EFA_Comparison.R")
library(reshape2)
library(ggplot2)
library(cluster)

DATADIR = '/home/knight/matar/MATLAB/DATA/Avgusta/Subjs/'
saveDir = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/plots_sim/'

d = read.table('/home/knight/matar/MATLAB/DATA/Avgusta/PCA/test.subjects',sep = '_') 
df = data.frame()

hsim = 0.5

for (i in 1:nrow(d) ) {
  
  subj = d$V1[i]
  task = d$V2[i]
  
  data = readMat(paste(DATADIR, subj, '/', task, '/HG_elecMTX_mean.mat', sep = ""))
  active_elecs = as.vector(data$active.elecs)
  data = t(data$data)
  
  #Determine number of PCs (using EFA)
  efa = EFA.Comp.Data(Data=data, F.Max=10, Graph=T) #calculate number of components
  filename = paste(saveDir, subj, "_", task, '_efa.png',sep = "")
  dev.copy(png,filename)
  dev.off()
  
  #PCA
  fit = principal(data, nfactors = efa, rotate='varimax') #calculate PCA with rotation
  
  #write loadings csv
  df_loadings = cbind(data.frame(active_elecs), data.frame(unclass(fit$loadings)), data.frame(fit$communality))
  df_SSloadings = data.frame(fit$values[1:efa])
  filename = paste(saveDir, subj, '_', task, '_loadings.csv',sep = "")
  write.table(df_loadings,file = filename, sep =",", row.names=FALSE)
  filename = paste(saveDir, subj, '_', task, '_SSloadings.csv',sep = "")
  write.table(df_SSloadings,file = filename, sep =",", row.names=FALSE)  
  
  #heirarchical clustering
  distance = dist(fit$loadings, method = 'correlation') #calculate distance between weights per elec
  clusters<-hclust(distance, method = 'complete') #calculate hierarchical clusters
  groups <- cutree(clusters, h=hsim)  #group clusters based on specific similarity
  nclust = max(groups)
  
  #add groups to dataframe
  df = rbind(df, data.frame(subj, task, active_elecs, groups))
  
  #plot tree
  a = plot(clusters,labels = as.character(active_elecs), main=paste(subj,task, '-', 'height:', as.character(hsim), sep = ' '))
  if (nclust==1) nclust<-2 #so can plot 1 big box if only have 1 cluster
  b = rect.hclust(clusters, k = nclust, cluster = groups, border="blue") 
  filename = paste(saveDir, subj, "_", task, '_hclust_sim.png',sep = "")
  dev.copy(png,filename)
  dev.off()
  
  #plot components
#   scores = as.data.frame(cbind(fit$scores, 1:nrow(fit$scores)))
#   timename = tail(colnames(scores),n=1)
#   names(scores)[names(scores)==timename] <- "time"
#   scores.long = melt(scores, id.vars = "time")
#   p <- ggplot(data=scores.long, aes(x = time, y = value)) + geom_line(aes(colour=variable),size=1.5) + xlab('time')+ scale_colour_hue(l=40) + theme_bw(base_size = 12) + theme_classic()+ scale_size_discrete()
#   filename = paste(saveDir, subj, "_", task, '_PCA.png',sep = "")
#   ggsave(file = filename, plot = p, width = 10, height = 4)
#   dev.off()
}

filename = paste(saveDir, 'groupidx_sim.csv',sep = "")
write.table(df,file = filename, sep =",", row.names=FALSE, col.names = c('subj','task','active_elec', 'groups'))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       