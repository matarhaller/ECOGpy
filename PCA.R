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

#for (i in 1:nrow(d) ) {
for (i in c(46)) {  
  subj = d$V1[i]
  task = d$V2[i]
  
  data = readMat(paste(DATADIR, subj, '/', task, '/HG_elecMTX_mean.mat', sep = ""))
  active_elecs = as.vector(data$active.elecs)
  data = t(data$data)
  
  efa = EFA.Comp.Data(Data=data, F.Max=10, Graph=T) #calculate number of components
  filename = paste(saveDir, subj, "_", task, '_efa.png',sep = "")
  dev.copy(png,filename)
  dev.off()
           
  fit = principal(data, nfactors = efa, rotate='varimax') #calculate PCA with rotation
  
  #write loadings csv
  df_loadings = cbind(data.frame(unclass(fit$loadings)), data.frame(fit$communality))
  df_SSloadings = data.frame(fit$values[1:efa])
  filename = paste(saveDir, subj, '_', task, '_loadings.csv',sep = "")
  write.table(df_loadings,file = filename, sep =",", row.names=FALSE)
  filename = paste(saveDir, subj, '_', task, '_SSloadings.csv',sep = "")
  write.table(df_SSloadings,file = filename, sep =",", row.names=FALSE)
  
  #distance = dist(fit$loadings, method = 'correlation') #calculate distance between weights per elec
  #clusters<-hclust(distance, method = 'complete') #calculate hierarchical clusters
  #groups <- cutree(clusters, k=efa)  #group clusters into number of components
  #df = rbind(df, data.frame(subj, task, active_elecs, groups)) #add groups to dataframe

  #plot tree
  #a = plot(clusters,labels = as.character(active_elecs))
  #b = rect.hclust(clusters, k = efa, cluster = groups, border="red") 
  #filename = paste(saveDir, subj, "_", task, '_hclust.png',sep = "")
  #dev.copy(png,filename)
  #dev.off()
    
  #plot components
  scores = as.data.frame(cbind(fit$scores, 1:nrow(fit$scores)))
  timename = tail(colnames(scores),n=1)
  names(scores)[names(scores)==timename] <- "time"
  scores.long = melt(scores, id.vars = "time")
  p <- ggplot(data=scores.long, aes(x = time, y = value)) + geom_line(aes(colour=variable),size=1.5) + xlab('time')+ scale_colour_hue(l=40) + theme_bw(base_size = 12) + theme_classic()+ scale_size_discrete()
  filename = paste(saveDir, subj, "_", task, '_PCA.png',sep = "")
  ggsave(file = filename, plot = p, width = 10, height = 4)

}

#filename = paste(saveDir, 'groupidx.csv',sep = "")
#write.table(df,file = filename, sep =",", row.names=FALSE, col.names = c('subj','task','active elec', 'group'))
