setwd("D:/Estudos/FGV/TCC/Rio2016");
library(ggplot2);
library(gmodels);

#################################
# An�lises do Top-4
#################################

# Carga dos Dados
Top4.Tiros = read.csv("Rio2016_Resultados_Top4_Tiros.csv", sep=";", dec=".");
Top4.Series = read.csv("Rio2016_Resultados_Top4_Series.csv", sep=";", dec=".");

summary(Top4.Tiros[-1]);
var(Top4.Tiros[-1]);
boxplot(Top4.Tiros[-1]);

summary(Top4.Series[-1]);
var(Top4.Series[-1]);
boxplot(Top4.Series[-1]);

#################################
# An�lises do Round
#################################

# Dados
Dados = read.csv("RoundOlimpico_Rio2016_Individual.csv", sep=";", dec=",");

# N�mero de Amarelos
Dados$Amarelos = Dados$Dez + Dados$Nove;

# An�lise Explorat�ria
hist(Dados$RoundTotal);
boxplot(Dados$RoundTotal);
hist(Dados$Dez);
boxplot(Dados$Dez);
hist(Dados$Nove);
boxplot(Dados$Nove);
hist(Dados$Amarelos);
boxplot(Dados$Amarelos);
hist(Dados$WorldRanking);
boxplot(Dados$WorldRanking);
table(Dados$Finalista);


#################################
# Padroniza��o de Vari�veis
#################################
Dados$Round1_S = scale(Dados$Round1);
Dados$Round2_S = scale(Dados$Round2);
Dados$RoundTotal_S = scale(Dados$RoundTotal);
Dados$Dez_S = scale(Dados$Dez);
Dados$X_S = scale(Dados$X);
Dados$Nove_S = scale(Dados$Nove);
Dados$WorldRanking_S = scale(Dados$WorldRanking);
Dados$PontosCombates_S = scale(Dados$PontosCombates);
Dados$Amarelos_S = scale(Dados$Amarelos);
Dados$Classe = "N�o Finalista";
Dados$Classe[which(Dados$Finalista==1)] = "Finalista";
Dados$Classe = factor(Dados$Classe);

#################################
# An�lise de Cluster
#################################

Cluster.Dados = Dados[, c(19,20,23,24,25)];

# An�lise com 4 Grupos
Cluster.Modelo = kmeans(Cluster.Dados,4);
Dados$Cluster.4 = Cluster.Modelo$cluster;
CrossTable(Dados$Cluster.4, Dados$Classe, chisq=FALSE);
aggregate(cbind(Dados$RoundTotal, Dados$Dez, Dados$WorldRanking, Dados$PontosCombates, Dados$Amarelos), list(Dados$Cluster.4), mean);

# An�lise com 5 Grupos
Cluster.Modelo = kmeans(Cluster.Dados,5);
Dados$Cluster.5 = Cluster.Modelo$cluster;
CrossTable(Dados$Cluster.5, Dados$Classe, chisq=FALSE);
aggregate(cbind(Dados$RoundTotal, Dados$Dez, Dados$WorldRanking, Dados$PontosCombates, Dados$Amarelos), list(Dados$Cluster.5), mean);

#################################
# Previs�o de Finalistas com KNN
#################################
library(class);

Knn.Base = Dados[, c(19,20,23,24,25)];
Knn.SorteioTeste = sample(Dados$Rank, 16, replace=TRUE);
Knn.BaseTeste = Knn.Base[Knn.SorteioTeste,];
Knn.ClassesTeste = Dados$Classe[Knn.SorteioTeste];
Knn.BaseTreino = Knn.Base[-Knn.SorteioTeste,];
Knn.ClassesTreino = Dados$Classe[-Knn.SorteioTeste];

Knn.Pred = knn(train=Knn.BaseTreino, test=Knn.BaseTeste, cl=Knn.ClassesTreino, k=8, prob=TRUE);
CrossTable(Knn.ClassesTeste, Knn.Pred, prop.chisq=FALSE);


################################################
# Previs�o De Finalistas com Regress�o Log�stica
################################################

# RegLog.Modelo = glm(Finalista ~ WorldRanking + PontosCombates + RoundTotal + Amarelos, family=binomial(logit), data=Dados);
RegLog.Modelo = glm(Finalista ~ WorldRanking_S + PontosCombates_S + RoundTotal_S + Amarelos_S, family=binomial(logit), data=Dados);
RegLog.Modelo;

# Predi��o
RegLog.Prob = predict.glm(RegLog.Modelo,type="response");
Dados = cbind(Dados, RegLog.Prob);

# Matriz de Confus�o
PontoDeCorte = 0.9;
Dados$Previsao = "N�o Finalista";
Dados$Previsao[which(Dados$RegLog.Prob>=PontoDeCorte)] = "Finalista";
CrossTable(Dados$Classe, Dados$Previsao, prop.chisq=FALSE);

library(caret);
library(e1071);
confusionMatrix(Dados$Previsao, Dados$Classe, positive="Finalista");

# Curva ROC da Predi��o
library(ROCR);
ROC.Pred = prediction(RegLog.Prob, Dados$Classe);
ROC.Perf = performance(ROC.Pred, measure="tpr", x.measure="fpr");
plot(ROC.Perf, main="Curva ROC", col="blue", lwd=2);
abline(a=0, b=1, lwd=2, lty=2, col="red");

# Grava��o dos resultados
write.csv2(Dados, "RoundOlimpico_Rio2016_Individual_Probs2.csv", sep=";", dec=",");


##########################################
# Previs�o de Finalistas com Decision Tree
##########################################
install.packages("C50");
library(C50);

Tree.Modelo = C5.0(Knn.BaseTreino, Knn.ClassesTreino, trials=1);

# Visualiza��o da �rvore
summary(Tree.Modelo);

# Predi��es (Classes)
Tree.Predicao.Classes = predict(Tree.Modelo, Knn.BaseTeste, type="class");
# Predi��es (Scoring)
Tree.Predicao.Scoring = predict(Tree.Modelo, Knn.BaseTeste, type="prob");

# Resultado (Matriz de Confus�o)
CrossTable(Knn.ClassesTeste, Tree.Predicao, prop.chisq=FALSE);


###################
# Limpeza dos Dados
###################
rm(list=ls());