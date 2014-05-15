library(ggplot2)

setwd('~/Desktop/')
my_data = './roi_onsets_allactiveelecs.csv'
roi_dur_onsets = read.csv(my_data, sep=',', header=TRUE, skip = 2)
roi_dur_onsets$is.duration<-factor(roi_dur_onsets$is.duration)

tmp<-subset(roi_dur_onsets, ROI == 'MFG' | ROI == 'Broca' | ROI == 'Pre_centrl') #keep ROIs of interest
tmp<-subset(tmp, is.duration == 1)
tmp$ROI<-factor(tmp$ROI)
plot(tmp$ROI, tmp$HGonset.mean)

tmp$difficulty = NA
tmp[which(tmp$task == 'EmoRep' | tmp$task == 'FaceGen'),]$difficulty = 'easy'
tmp[which(tmp$task == 'SelfAud' | tmp$task == 'EmoGen'),]$difficulty = 'hard'

roi_info = tmp[which(tmp$difficulty == 'easy' | tmp$difficulty == 'hard'),]
roi_info$difficulty <- factor(roi_info$difficulty)

plot(HGonset.mean ~ difficulty*ROI, data = roi_info)

aov_result <- aov(HGonset.mean ~ difficulty*ROI, data=roi_info)

#roi_summary<-summarySE(roi_info,measurevar="HGonset.mean",groupvars=c("ROI","difficulty"))
roi_summary <- summarySEwithin(roi_info, measurevar="HGonset.mean", withinvars=c("difficulty", "ROI"),conf.interval=.95)

#cbPalette <- c("#D55E00","#0072B2")
cbPalette <- c("#ef8a62","#0072B2")
bp4<-ggplot(roi_summary,aes(x=ROI,y=HGonset.mean, group = difficulty, shape=difficulty, color = difficulty)) + geom_line(size=2) + geom_errorbar(aes(ymin=HGonset.mean-se,ymax=HGonset.mean+se), colour="black",width=.1, size = 1)  + geom_point(size=10) + theme_bw() + theme(legend.justification=c(1,0),legend.position="right") + theme(legend.title = element_blank(), legend.background=element_rect(fill="white",size=.5)) + scale_fill_manual(values=cbPalette)+ scale_colour_manual(values=cbPalette)

#bp4<-ggplot(roi_summary,aes(x=HGonset.mean, y = ROI, group = difficulty, shape=difficulty, color = difficulty)) + geom_line(size=2) +geom_point(size=10)+ theme_bw() + theme(legend.justification=c(1,0),legend.position="right") + theme(legend.title = element_blank(), legend.background=element_rect(fill="white",size=.5)) + scale_fill_manual(values=cbPalette)+ scale_colour_manual(values=cbPalette) + scale_x_continuous('onsets') + scale_y_discrete('ROI')

