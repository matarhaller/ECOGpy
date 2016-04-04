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
saveDir = '/home/knight/matar/MATLAB/DATA/Avgusta/PCA/plots_gap/'

d = read.table('/home/knight/matar/MATLAB/Avgusta/PCA/test.subjects',sep = '_') 
df = data.frame()

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
  df_loadings = cbind(data.frame(unclass(fit$loadings)), data.frame(fit$communality))
  df_SSloadings = data.frame(fit$values[1:efa])
  filename = paste(saveDir, subj, '_', task, '_loadings.csv',sep = "")
  write.table(df_loadings,file = filename, sep =",", row.names=FALSE)
  filename = paste(saveDir, subj, '_', task, '_SSloadings.csv',sep = "")
  write.table(df_SSloadings,file = filename, sep =",", row.names=FALSE)
  
  #determine number of clusters (using gap statistic)
  myclust<-function(x,k) list(cluster = cutree(hclust(dist(x, method = 'correlation'), method = 'complete'), k=k))
  kmax = min(length(active_elecs)-1,15) #maximum number of clusters to try
  result = clusGap(fit$loadings, FUN = myclust, K.max = kmax, B = 500, verbose = 1)
  
  gap = result$Tab[,"gap"]
  nclust = maxSE(gap, result$Tab[,'SE.sim'], method = 'firstSEmax') #default method 
  nclust2 = maxSE(gap, result$Tab[,'SE.sim'], method = 'globalSEmax')
  
  #plot gap statistic
  plot(result, lwd = 2, main=paste('gap statistic, niter = ', result$B))
  lines(rep(nclust, 2), range(gap), col = "blue", lty = 6, lwd = 2)
  lines(rep(nclust2, 2), range(gap), col = "green", lty = 4, lwd = 2)
  legend('bottomright',c(paste('firstSEmax: n=', nclust), paste('globalSEmax: n=', nclust2)), fill = c('blue','green'))
  filename = paste(saveDir, subj, "_", task, "_cgap.png", sep = "")
  dev.copy(png, filename)
  dev.off()
  
  #heirarchical clustering
  distance = dist(fit$loadings, method = 'correlation') #calculate distance between weights per elec
  clusters<-hclust(distance, method = 'complete') #calculate hierarchical clusters
  groups <- cutree(clusters, k=nclust)  #group clusters into number of components
  groups2 <-cutree(clusters, k=nclust2)
  
  #add groups to dataframe
  df = rbind(df, data.frame(subj, task, active_elecs, groups, groups2))
  
  #plot tree
  a = plot(clusters,labels = as.character(active_elecs), main=paste(subj,task,'firstSEmax', sep = ' '))
  if (nclust==1) nclust<-2 #so can plot 1 big box if only have 1 cluster
  b = rect.hclust(clusters, k = nclust, cluster = groups, border="blue") 
  filename = paste(saveDir, subj, "_", task, '_hclust_local.png',sep = "")
  dev.copy(png,filename)
  dev.off()
  
  a = plot(clusters,labels = as.character(active_elecs), main=paste(subj, task,'globalSEmax',sep = ' '))
  if (nclust2==1) nclust2<-2 #so can plot 1 big box if only have 1 cluster
  b = rect.hclust(clusters, k = nclust2, cluster = groups2, border="green") 
  filename = paste(saveDir, subj, "_", task, '_hclust_global.png',sep = "")
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

filename = paste(saveDir, 'groupidx_gap.csv',sep = "")
write.table(df,file = filename, sep =",", row.names=FALSE, col.names = c('subj','task','active_elec', 'group_firstSEmax', 'group_globalSEmax'))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       