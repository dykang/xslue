#install.packages(c("ggpubr", "tidyverse", "Hmisc", "corrplot"))
#devtools::install_url("https://github.com/wilkelab/cowplot/archive/0.6.3.zip")
#library(ggpubr)
#library(tidyverse)
library(Hmisc)
library(corrplot)
library("PerformanceAnalytics")
library(RColorBrewer) # for convenience of heatmap colors, it reflects your mood sometimes


exit <- function() {
  .Internal(.invokeRestart(list(NULL, NULL), NULL))
}


#cond='_coi=short_onlypastel_limit=False'


#cond_list <- c("_limit=False", "_coi=full_limit=False", "_coi=short_limit=False")
cond_list <- c("_coi=short_nopastel_limit=False", "_coi=short_onlypastel_limit=False")

#cond_list <- c("_limit=1000", "_coi=full_limit=1000", "_coi=short_limit=1000")
#cond_list <- c("_coi=short_nopastel_limit=1000", "_coi=short_onlypastel_limit=1000")


for (cond in cond_list){
    print("Start Correlation Analysis using R")
    print(cond)
    my_data <- read.csv(file=paste0("./sample/samples",cond,'.csv'), header=TRUE, sep=",")
    head(data)

    #my_data <- mtcars[, c(1,3,4,5,6,7)]
    png(paste0("output/pairplot",cond,".png"), width = 2000, height = 2000)
    chart.Correlation(my_data, histogram = TRUE, pch = 19)
    dev.off()
    print("PairPlot Done")
}
exit()

for (cond in cond_list){
    print("Start Correlation Analysis using R")
    print(cond)
    my_data <- read.csv(file=paste0("./sample/samples",cond,'.csv'), header=TRUE, sep=",")
    head(data)
    #my_data$cyl <- factor(my_data$cyl)
    #str(my_data)
    M<-cor(my_data)
    print("Corr Done")

    #exit()
    assign("w",2000)
    assign("h",2000)
    assign("rf",0.5) # font ratio 0.3

    assign("tlcex",2.5) # font: 3.5 for nopastel & onlypastel, otherwise 2.5

    assign("method", "color") #"circle"
    #col<- colorRampPalette(c("red", "yellow", "green"))(100)
    col <- brewer.pal(n = 50, name = "RdYlBu")

    cex.before <- par("cex")
    par(cex = rf)
    png(paste0("output/corr_circle_upper_hclust",cond,'.png'), width = w, height = h)
    #corrplot(M, method="circle", type="upper", order = "hclust", col=col, addrect = 10, tl.col = "black", tl.srt = 75, tl.cex = 1.0/par("cex"), cl.cex = 1.0/par("cex"), cl.ratio = 0.05, cl.align = "r")
    corrplot(M, method=method, type="upper", order = "hclust", col=col, addrect = 50, tl.col = "black", tl.srt = 75, tl.cex = tlcex, cl.cex = 1.0, cl.ratio = 0.05, cl.align = "r")
    dev.off()
    par(cex = cex.before)


    cex.before <- par("cex")
    par(cex = rf)
    png(paste0("output/corr_circle_upper",cond,".png"), width = w, height = h)
    #corrplot(M, method="circle", type="upper", col=col, tl.col = "black", tl.srt = 75, tl.cex = 1/par("cex"), cl.cex = 1/par("cex"), cl.ratio = 0.05, cl.align = "r")
    corrplot(M, method=method, type="upper", col=col, tl.col = "black", tl.srt = 75, tl.cex = tlcex, cl.cex = 1.0, cl.ratio = 0.05, cl.align = "r")

    dev.off()
    par(cex = cex.before)
    print("Plot Done")

    # Mark the insignificant coefficients according to the specified p-value significance level
    cex.before <- par("cex")
    par(cex = rf)
    png(paste0("output/corr_circle_upper_coeff_0.95_0.05",cond,".png"), width = w, height = h)
    #cor_5 <- rcorr(as.matrix(my_data))
    ##M <- cor_5$r
    #p_mat <- cor_5$P
    res1 <- cor.mtest(my_data, conf.level = .95)
    res2 <- cor.mtest(my_data, conf.level = .99)
    #corrplot(M, type = "upper", method="circle", col=col, p.mat = res1$p, sig.level = 0.05,  insig = "p-value",tl.col = "black", tl.srt = 75, tl.cex = 1/par("cex"), cl.cex = 1/par("cex"), cl.ratio = 0.05, cl.align = "r")
    # insig = "p-value",
    corrplot(M, type = "upper", method=method, col=col, p.mat = res1$p, sig.level = 0.05,  tl.col = "black", tl.srt = 75, tl.cex = tlcex, cl.cex = 1.0, cl.ratio = 0.05, cl.align = "r")
    dev.off()
    par(cex = cex.before)
    print("Plot_coeff Done")
}



