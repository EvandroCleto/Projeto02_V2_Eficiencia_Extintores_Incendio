# Data Science Academy - Formação Cientista de Dados

# Autor: Evandro Eulálio Cleto

# Data Início:      14/04/2023
# Data Finalização: 09/05/2023
#________________________________________________________________________________________________________________________________________________________________
# Big Data Analytics com R e Microsoft Azure Machine Learning Versão 3.0
# Projeto com Feedback 2
# Machine Learning na Segurança do Trabalho Prevendo a Eficiência de Extintores de Incêndio
#________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________
# Objetivo deste projeto = Revendo a Eficiência de Extintores de Incêndio

# Tipo de Aprendizado a ser utilizado = Supervisionado para Classificação

# Variável target = STATUS

# Métrica a ser alcançada = 80%

# Métricas a serem avaliadas:
# 1) Acurácia;
# 2) Precisão(Previsão de Falso positivo é mais prejudicial que falsos negativos);
# Neste problema de negócio, os falsos positivos irão prever o "extinction state" quando na verdade houve o "non-extinction state" 
# esta situação é extremamente perigosa em casos de incêndio.
# 3) AUC.
#_________________________________________________________________________________________________________________________________________________________________
#_________________________________________________________________________________________________________________________________________________________________
# Algorítimos de Regressão a serem utilizados no projeto para criação dos modelos e avaliação
### 1) Regressão Logistica(Benchmark);
### 2) KNN(2 versões);
### 3) Naive Bayes(2 versões);
### 4) SVM(2 versões).

#------------------------------------------------------------- Dicionário de Dados -----------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|Data properties and descriptions for liquid fuels				
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FEATURES	|MIN/MAX VALUES	            |UNIT	|DESCRIPTIONS	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|SIZE	    |7, 12, 14, 16, 20	        |cm		|Recorded as 7 cm=1, 12 cm=2, 14 cm=3, 16 cm=4, 20 cm=5	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FUEL	    |Gasoline, Kerosene, Thinner|	 	  |	Fuel type	- 
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|DISTANCE	|10 - 190				          	|cm		|
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|DESIBEL 	|72 - 113				          	|dB		|
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|AIRFLOW	  |0 - 17	    		        		|m/s	|	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FREQUENCY	|1-75	    		           		|Hz		|
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|STATUS	  |0, 1				             		|	  	| 0 indicates the non-extinction state, 1 indicates the extinction state	
#---------------------------------------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|Data properties and descriptions for LPG				
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FEATURES	|MIN/MAX VALUES	                              |UNIT	|DESCRIPTIONS	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|SIZE	  	|Half throttle setting, Full throttle setting |		|Reocerded as Half throttle setting=6, Full throttle setting=7	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FUEL	  	|LPG		                                      |   |      Fuel type	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|DISTANCE	|10 - 190								                  	  |cm	|	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|DESIBEL 	|72 - 113					                  				  |dB	|	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|AIRFLOW	  |0 - 17	    						                		  |m/s|	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FREQUENCY	|1-75	    								                    |Hz	|	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|STATUS	  |0, 1										                      |		|0 indicates the non-extinction state, 1 indicates the extinction state	
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FUEL_gasoline|0, 1										                  |		|Variável Dummy	gerada a partir de FUEL
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FUEL_kerosene|0, 1										                  |		|Variável Dummy	gerada a partir de FUEL
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FUEL_lpg     |0, 1										                  |		|Variável Dummy	gerada a partir de FUEL
#---------------------------------------------------------------------------------------------------------------------------------------------------#
#|FUEL_thinner |0, 1										                  |		|Variável Dummy	gerada a partir de FUEL
#---------------------------------------------------------------------------------------------------------------------------------------------------#

# Configuração do diretório de trabalho,
setwd("D:/Cursos/Curso_FCD/1-BigDataRAzure/Arquivos/Cap22/Projeto02_V_3")
getwd()
# Importa Bibliotecas.
library(readxl)# Biblioteca para importar arquivo em Excel.
library(tidyverse)
library('lattice')
library(writexl)# Pacote para exportação de dataframes em Excel
library(car)# Pacote para geração de qqPlots
library(e1071) # Pacote para analisar Assimetria e Curtose
library(caTools) # Pacote para divisão de dados em treino e teste
library(caret)
library(ROCR)
library(pROC)# Pacote para visualizar, suavizar e comparar as características de curvas ROC
library(multiROC)# Pacote para calcular métricas(Specificity, Sensitivity and AUC) de Classificações Multi-Classes
library(ggplot2)
search()

# 1- Carga de dados:
dados_original <- read_excel('Acoustic_Extinguisher_Fire_Dataset.xlsx', sheet = 'A_E_Fire_Dataset')

# 2- Análise Exploratória:

# Visualiza Data Frame criado na importação.
View(dados_original)

# Exibe a dimensão dos dados.
dim(dados_original)

# Exibe resumo da estrutura dos Dados.
str(dados_original)

# Exibe as Classes.
class(dados_original)

# Exibe o sumários dos dados.
summary(dados_original)

# Checa dados missing(NAs):
colSums(is.na(dados_original)) 

# Não há dados missing porém a varível AIRFLOW possui valor mínimo 0 e deverá ser avaliada, pois presume-se que com este valor mínimo não existiu
# fluxo de ar e desta forma não extinguiu o fogo.

# Verifico quantos registros possuem AIRFLOW igual a 0:
dados_original %>% filter(AIRFLOW == 0) %>% count()
# Existem 1632 registros

# Agora verifica quantos registros possuem AIRFLOW igual a 0 e que extinguiram o incendio(variável target STATUS == 1)
dados_original %>% filter(AIRFLOW == 0 & STATUS == 1) %>% count()
# Verificado que existiram 18 registros onde houve extinsão do fogo, porém o AIRFLOW foi 0.
# Desta forma vou optar em manter a variável  AIRFLOW igual a 0.
# Fica como ponto de atenção e caso os modelos apresentem baixa performance, estudarei a remoção de observações nesta situação ou alguma outra técnica 
# de imputação para substituir valores = 0.  

# Verificando balanceamento da variável STATUS(Target):
# 0 indicates the non-extinction state, 1 indicates the extinction state.
STATUS_table <- table(as.character(dados_original$STATUS))
STATUS_table <- round((prop.table(STATUS_table)* 100),2)
print(STATUS_table)

# Proporção da variável STATUS(Target):
# 0(non-extinction state) = 50.22%
# 1(extinction state)     = 49.78%
# Verificado que não será necessário aplicar técnica de oversampling na variável STATUS pois as 2 classes estão em proporções semelhentes. 

# Realiza o encoding da variável Fuel para análise de correlação e cria a variável Fuel_FAC:
dados_original$FUEL_FAC <- factor(dados_original$FUEL, labels = c('1','2','3','4'))

# Transforma a variável FUEL_FAC para numérico para análise de correlação:
dados_original$FUEL_FAC <- as.numeric(ordered(dados_original$FUEL_FAC, levels = c('1','2','3','4')))

# Análise de Correlação:
cols <- c("SIZE" ,"DISTANCE", "DESIBEL", "AIRFLOW",
          "FREQUENCY", "FUEL_FAC","STATUS")

# Métodos de Correlação a serem utilizados:
# Pearson - coeficiente usado para medir o grau de relacionamento entre duas variáveis com relação linear.
# Spearman - Teste não paramétrico, para medir o grau de relacionamento entre duas variaveis.

# Vetor com os métodos de correlação:
metodos <- c("pearson", "spearman")

# Aplicando os métodos de correlação com a função cor():
cors <- lapply(metodos, function(method)
  (cor(dados_original[, cols], method = method)))

head(cors)

# Preprando o plot:
require(lattice)
plot.cors <- function(x, labs){
  diag(x) <- 0.0 
  plot( levelplot(x, 
                  main = paste("Plot de Correlação usando Método", labs),
                  scales = list(x = list(rot = 90), cex = 1.0)) )
}

# Mapa de Correlação:
Map(plot.cors, cors, metodos)

# Na análise de correlação(Em ambos os métodos) cheguei as  seguintes conclusões:
# 1- A varíavel DISTANCE tem uma média correlação negativa(-0.6440506) com a variável Target(STATUS);  
# 2- A varíavel AIRFLOW  tem uma alta correlação positiva(0.7585152) com a variável Target(STATUS);  
# 3- A varíavel FUEL_FAC tem uma baixa correlação negativa(-0.02427262) com a variável Target(STATUS) e será retirada do modelo;
# 4- As varíaveis DESIBEL e FREQUENCY tem um média correlação positiva entre elas(0.6649270), mas vou manter ambas as variáveis no dataset, pois acredito 
# acredito que não exista multicolinearidade entre elas;
# 5- As varíaveis DISTANCE e AIRFLOW  tem um média correlação negativa entre elas(-0.6893409), mas vou manter ambas as variáveis no dataset, pois acredito 
# acredito que não exista multicolinearidade entre elas.

# 3- Pré Processamento:

# Retira a variável FUEL e FUEL_FAC  do dataset:
dados_original$FUEL <- NULL
dados_original$FUEL_FAC  <- NULL

# Visualiza Data Frame Pre Processado:
View(dados_original)

# Exibe resumo da estrutura dos Dados:
str(dados_original)

# Exibe o sumários dos dados:
summary(dados_original)

# Gero novo dataframe para seguir com o projeto:
dados_processados <- dados_original

# Visualiza novo Data Frame:
View(dados_processados)

# Salvo novo Data Frame:
# write_xlsx(dados_processados, "dados_processados.xlsx")

# Carrega novo Data Frame:
# dados_processados <- read_excel('dados_processados.xlsx', sheet = 'Sheet1')

#  4- Analise Estatistica:
# Exibe os sumários dos dados para comparação com os Boxplots.
summary(dados_processados)

# Criação dos Boxplots das variáveis para análise estátistica:

# Lista com o nome das variáveis para geração dos Boxplots e Histogramas.
lista_variaveis <- c('DISTANCE','DESIBEL','AIRFLOW','FREQUENCY')

# Função para gerar os Boxplots:
gerar_boxplots <- function(dataframe, lista_variaveis){
  par(mfrow = c(2, 2))
  for(variavel in lista_variaveis){
    if(variavel %in% colnames(dataframe)){
      boxplot(dataframe[[variavel]], main = variavel)
    }
    else{
      cat(paste("Variável", variavel, "não encontrada no dataframe.\n"))
    }
  }
}

# Execução da Função para gerar os Boxplots:
gerar_boxplots(dados_processados, lista_variaveis)

# Na análise dos boxplots não foram encontrados outliers.

#Função para gerar os Histogramas:
gerar_histogramas <- function(dataframe, lista_variaveis){
  par(mfrow = c(2, 2))
  for(variavel in lista_variaveis){
    if(variavel %in% colnames(dataframe)){
      hist(dataframe[[variavel]], main = variavel, xlab = variavel, breaks = 6)
    }
    else{
      cat(paste("Variável", variavel, "não encontrada no dataframe.\n"))
    }
  }
}

# Execução da Função para gerar os Histogramas:
gerar_histogramas(dados_processados, lista_variaveis)

# Na análise dos histogramas verificou-se que as variáveis não seguem uma distribuição normal e deverão ser padronizadas.
# Antes porém farei uma análise da Assimetria e Curtose das variávies.

# Análise da Variável DISTANCE:
sk <- skewness(dados_processados$DISTANCE)
print(sk)
#sk = 0 -> Os dados são simétricos, logo seguem uma Distribuição Normal.
ck <- kurtosis(dados_processados$DISTANCE)
print(ck)
#ck = -1.206872. Coeficiente de Curtose negativo(Platicúrtica).
# A Variável DISTANCE deverá ser normalizada.

# Análise da Variável DESIBEL:
sk <- skewness(dados_processados$DESIBEL)
print(sk)
#sk = -0.1790166 -> Os dados possuem assimetria negativa, logo não seguem uma Distribuição Normal.
ck <- kurtosis(dados_processados$DESIBEL)
print(ck)
#ck = -0.9012049. Coeficiente de Curtose negativo(Platicúrtica).
# A Variável DESIBEL deverá ser normalizada.

# Análise da Variável AIRFLOW:
sk <- skewness(dados_processados$AIRFLOW)
print(sk)
#sk = 0.2443264 -> Os dados possuem assimetria positiva, logo não seguem uma Distribuição Normal.
ck <- kurtosis(dados_processados$AIRFLOW)
print(ck)
#ck = -1.19817. Coeficiente de Curtose negativo(Platicúrtica).
# A Variável AIRFLOW deverá ser normalizada.

#Análise da Variável FREQUENCY:
sk <- skewness(dados_processados$FREQUENCY)
print(sk)
#sk = 0.4348175 -> Os dados possuem assimetria positiva, logo não seguem uma Distribuição Normal.
ck <- kurtosis(dados_processados$FREQUENCY)
print(ck)
#ck = -0.9012049. Coeficiente de Curtose negativo(Platicúrtica).
# A Variável FREQUENCY deverá ser normalizada.

# 5- Engenharia de Atributos:
# Crio um Data Frame em separado para as variáveis normalizadas para não duplicar estas variáveis no dataset dados_processados.
dados_scaled <- dados_processados[, lista_variaveis]
View(dados_scaled)

# Padronização do z-score Usando a função scale():
dados_scaled <- as.data.frame(scale(dados_scaled))
View(dados_scaled)

# Função para transformação das variáveis em fatores:
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

# Variáveis do tipo fator:
categorical.vars <- c('SIZE', 'STATUS')

# Executa transformação em fator:
dados_processados <- to.factors(df = dados_processados, variables = categorical.vars)
str(dados_processados)

# Retiro as colunas que foram Normalizadas do dataframe dados_processados:
dados_processados$DISTANCE<-NULL
dados_processados$DESIBEL<-NULL
dados_processados$AIRFLOW<-NULL
dados_processados$FREQUENCY<-NULL

# Adiciona as demais colunas do dataset dados_processados ao dataset dados_padronizados e cria um novo Data Frame:
dados_padronizados <- cbind(dados_scaled, dados_processados)
View(dados_padronizados)
str(dados_padronizados)

# Movendo a variável STATUS(target) para a 1ª posição do dataset:
dados_padronizados <- dados_padronizados %>% select(STATUS, everything())

# Salvo novo dataframe:
#write_xlsx(dados_padronizados, "dados_padronizados.xlsx")

# Carrega os dados:
dados_padronizados <- read_excel('dados_padronizados.xlsx', sheet = 'Sheet1')
View(dados_padronizados)

# Executa transformação em fator, pois importou as variáveis SIZE e STATUS como caracter:
dados_padronizados <- to.factors(df = dados_padronizados, variables = categorical.vars)
str(dados_padronizados)

# Divisão treino teste:
indexes <- sample(1:nrow(dados_padronizados), size = 0.6 * nrow(dados_padronizados))

# Cria dataset de treino: 
train.data <- dados_padronizados[indexes,]

# Cria dataset de teste:
test.data <- dados_padronizados[-indexes,]

# Verifica o balanceamento de classe da variável Target(STATUS) em train.data:
prop.table(table(train.data$STATUS)) * 100
# Verifica-se que as classes no Data Frame train.data também estão balanceadas dispensando técnicas de Oversampling para balanceamento.

# Separando os atributos e as classes:
# Cria um novo dataframe de teste sem a variáve Target.
test.feature.vars <- test.data[,-1]

# Cria um novo dataframe de teste apenas com a variáve Target.
test.class.var <- test.data[,1]

# Função para Plot ROC - Esta função será usada com todos os modelos deste projeto:
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf,col = "black",lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8,xaxs = "i", yaxs = "i")
  abline(0,1, col = "red")
  auc <- performance(predictions,"auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4,legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
  
}

# 6- Criação dos Modelos:

# 6.1.1 Construindo o modelo de Regressão logística(Benchmark):
formula.init <- "STATUS ~ ."
formula.init <- as.formula(formula.init)
modelo_Regressao_log <- glm(formula = formula.init, data = train.data, family = 'binomial')
modelo_Regressao_log

# Visualizando os detalhes do modelo:
summary(modelo_Regressao_log)

# 6.1.2 Fazendo as previsões e analisando o resultado:
previsoes_Regressao_log <- predict(modelo_Regressao_log, test.data, type = 'response')
previsoes_Regressao_log <- round(previsoes_Regressao_log)
View(previsoes_Regressao_log)

# 6.1.3 - Avaliacao do Modelo:
#Confusion Matrix.
resultados_Regressao_log <- caret::confusionMatrix(as.factor(previsoes_Regressao_log), test.class.var$STATUS)

# Medidas Globais modelo Regressão Logistica:
Acuracia_Regressao_log  <- resultados_Regressao_log$overall['Accuracy']
Precisao_Regressao_log  <-resultados_Regressao_log$byClass['Precision']
curva_roc_Regressao_log <- multiclass.roc(response = test.class.var$STATUS, predictor = previsoes_Regressao_log)
Auc_Regressao_log <- curva_roc_Regressao_log$auc

# Vetor com os resultados de avaliação do Modelo Regressão Logistica:
vetor_Reg_Log <- c("Modelo Regressao Logistica", round(Acuracia_Regressao_log, 4), round(Precisao_Regressao_log, 4), round(Auc_Regressao_log, 4))

# 6.1.4 ROC Curve:
modelo_final_Reg_Log <- modelo_Regressao_log
previsoes_Reg_Log <- predict(modelo_final_Reg_Log, test.feature.vars, type = "response")
previsoes_finais_Reg_Log <- prediction(previsoes_Reg_Log, test.class.var)

# Plot ROC:
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais_Reg_Log, title.text = "Curva ROC")
class(previsoes_finais_Reg_Log)

# 6.2.1 Construindo o modelo versão 1 com KNN(K Nearest Neighbors):

# Arquivo de controle para treino aplicando Cross-validation:
ctrl <- trainControl(method = "repeatedcv", repeats = 3) 

# Criação do modelo:
modelo_knn_v1 <- train(STATUS ~ ., 
                data = train.data, 
                method = "knn", 
                trControl = ctrl, 
                tuneLength = 20)

# 6.2.2 Avaliacao do Modelo:
modelo_knn_v1
class(modelo_knn_v1)
# O valor final usado para o modelo foi k = 7

## Plota Número de Vizinhos x Acurácia:
plot(modelo_knn_v1)

# 6.2.3 Fazendo as previsões e analisando o resultado:
previsoes_KNN_V1 <- predict(modelo_knn_v1, newdata = test.data)

# Confusion Matrix:
resultados_KNN_V1 <- caret::confusionMatrix(previsoes_KNN_V1,as.factor(test.data$STATUS))

# Medidas Globais modelo KNN V1:
Acuracia_KNN_V1  <- resultados_KNN_V1$overall['Accuracy']
Precisao_KNN_V1  <-resultados_KNN_V1$byClass['Precision']
curva_roc_KNN_V1 <- multiclass.roc(response = test.class.var$STATUS, predictor = as.numeric(previsoes_KNN_V1))
Auc_KNN_V1 <- curva_roc_KNN_V1$auc

# Vetor com os resultados de avaliação do Modelo Regressão Logistica:
vetor_KNN_V1 <- c("Modelo KNN V1", round(Acuracia_KNN_V1, 4), round(Precisao_KNN_V1, 4), round(Auc_KNN_V1, 4))

# 6.2.4 ROC Curve:
previsoes_finais_KNN_V1 <- prediction(as.numeric(previsoes_KNN_V1), test.class.var)
View(previsoes_finais_KNN_V1)

# Gera o Plot:
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais_KNN_V1, title.text = "Curva ROC")

# 6.3.1 Construindo o modelo versão 2 com KNN(K Nearest Neighbors):

# Arquivo de controle para treino aplicando Cross-validation.
ctrl <- trainControl(method = "repeatedcv", 
                     repeats = 3, 
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

# Criação do modelo:
# Transforma a variável Target em texto, devido à exigência do modelo com a metrica ROC:
train.data.x <- train.data
train.data.x$STATUS <- sapply(train.data.x$STATUS, function(x){ifelse(x=='1','Pos','Neg')})
modelo_knn_v2  <- train(STATUS ~ ., 
                              data = train.data.x, 
                              method  = "knn", 
                              trControl  = ctrl, 
                              metric  = "ROC",
                              tuneLength = 20)

# 6.3.2 Avaliacao do Modelo:
modelo_knn_v2
# O valor final usado no modelo foi k = 17. 

# Plota Número de Vizinhos x Acurácia:
plot(modelo_knn_v2)

# 6.3.3 Fazendo as previsões e analisando o resultado:
previsoes_KNN_V2 <- predict(modelo_knn_v2, newdata = test.data)

# Confusion Matrix:
# Tranformo a previsão em fator para gerar a confusionMatrix.
prev_KNN_fac<- factor(previsoes_KNN_V2, levels = c("Neg", "Pos"), labels = c(0, 1))
resultados_KNN_V2 <- caret::confusionMatrix(prev_KNN_fac,as.factor(test.data$STATUS))

# Medidas Globais modelo KNN V2:
Acuracia_KNN_V2  <- resultados_KNN_V2$overall['Accuracy']
Precisao_KNN_V2  <-resultados_KNN_V2$byClass['Precision']
curva_roc_KNN_V2 <- multiclass.roc(response = test.class.var$STATUS, predictor = as.numeric(previsoes_KNN_V2))
Auc_KNN_V2 <- curva_roc_KNN_V2$auc

# Vetor com os resultados de avaliação do modelo KNN V2:
vetor_KNN_V2 <- c("Modelo KNN V2", round(Acuracia_KNN_V2, 4), round(Precisao_KNN_V2, 4), round(Auc_KNN_V2, 4))

# 6.3.4 ROC Curve:
previsoes_finais_KNN_V2<- prediction(as.numeric(previsoes_KNN_V2), test.class.var)
View(previsoes_finais_KNN_V2)

# Gera o Plot:
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais_KNN_V2, title.text = "Curva ROC")

# 6.4.1 Construindo o modelo com Naive Bayes V1:

# Criação do modelo:
modelo_NB_v1  <- naiveBayes(STATUS ~ ., data = train.data)

# 6.4.2 Fazendo as previsões e analisando o resultado:
previsoes_NB_v1 <- predict(modelo_NB_v1, newdata = test.data)

# Confusion Matrix:
resultados_NB_V1 <- caret::confusionMatrix(previsoes_NB_v1,as.factor(test.data$STATUS))

# Medidas Globais modelo Naive Bayes V1:
Acuracia_NB_V1  <- resultados_NB_V1$overall['Accuracy']
Precisao_NB_V1  <-resultados_NB_V1$byClass['Precision']
curva_roc_NB_V1 <- multiclass.roc(response = test.class.var$STATUS, predictor = as.numeric(previsoes_NB_v1))
Auc_NB_V1 <- curva_roc_NB_V1$auc

# Vetor com os resultados de avaliação do Modelo Naive Bayes V1:
vetor_NB_V1 <- c("Modelo Naive Bayes V1", round(Acuracia_NB_V1, 4), round(Precisao_NB_V1, 4), round(Auc_NB_V1, 4))

# 6.4.3 ROC Curve:
previsoes_finais_NB_v1<- prediction(as.numeric(previsoes_NB_v1), test.class.var)

# Gera o Plot:
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais_NB_v1, title.text = "Curva ROC")

# 6.5.1 Construindo o modelo com Naive Bayes aplicando suavização laplace:

# Criação do modelo:
modelo_NB_v2  <- naiveBayes(STATUS ~ ., data = train.data,laplace = 1)

# 6.5.2 Fazendo as previsões e analisando o resultado:
previsoes_NB_v2 <- predict(modelo_NB_v2, newdata = test.data)

# Confusion Matrix:
resultados_NB_V2 <- caret::confusionMatrix(previsoes_NB_v2,as.factor(test.data$STATUS))

# Medidas Globais do modelo Naive Bayes V2:
Acuracia_NB_V2  <- resultados_NB_V2$overall['Accuracy']
Precisao_NB_V2  <-resultados_NB_V2$byClass['Precision']
curva_roc_NB_V2 <- multiclass.roc(response = test.class.var$STATUS, predictor = as.numeric(previsoes_NB_v2))
Auc_NB_V2 <- curva_roc_NB_V2$auc

# Vetor com os resultados de avaliação do Modelo Naive Bayes V2:
vetor_NB_V2 <- c("Modelo Naive Bayes V2", round(Acuracia_NB_V2, 4), round(Precisao_NB_V2, 4), round(Auc_NB_V2, 4))

# 6.5.3 ROC Curve:
previsoes_finais_NB_v2<- prediction(as.numeric(previsoes_NB_v2), test.class.var)

# Gera o Plot:
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais_NB_v2, title.text = "Curva ROC")

# 6.6.1 Construindo o modelo com SVM - Versão Padrão com Kernel Radial (RBF):

# Criação do modelo:
modelo_SVM_v1  <- svm(STATUS ~ ., data = train.data, na.action = na.omit, scale = TRUE)
summary(modelo_SVM_v1)
print(modelo_SVM_v1)

# 6.6.2 Fazendo as previsões e analisando o resultado:
previsoes_SVM_v1 <- predict(modelo_SVM_v1, newdata = test.data)

# Confusion Matrix:
resultados_SVM_v1 <- caret::confusionMatrix(previsoes_SVM_v1,as.factor(test.data$STATUS))

# Medidas Globais modelo SVM V1
Acuracia_SVM_v1  <- resultados_SVM_v1$overall['Accuracy']
Precisao_SVM_v1  <-resultados_SVM_v1$byClass['Precision']
curva_roc_SVM_v1 <- multiclass.roc(response = test.class.var$STATUS, predictor = as.numeric(previsoes_SVM_v1))
Auc_SVM_v1 <- curva_roc_SVM_v1$auc

# Vetor com os resultados de avaliação do Modelo SVM V1:
vetor_SVM_v1 <- c("Modelo SVM V1", round(Acuracia_SVM_v1, 4), round(Precisao_SVM_v1, 4), round(Auc_SVM_v1, 4))

# 6.6.3 ROC Curve:
previsoes_finais_SVM_v1<- prediction(as.numeric(previsoes_SVM_v1), test.class.var)

# Gera o Plot:
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais_SVM_v1, title.text = "Curva ROC")

# 6.7.1 Construindo o modelo com SVM - Versão com Kernel Linear e GridSearch:

# Criação do modelo:
# Realiza Grid Search para o ajuste de hiperparâmetros e usa Kernel linear. 
modelo_SVM_v2 <- tune(svm, 
                      STATUS ~ ., 
                     data = train.data, 
                     kernel = 'linear',
                     ranges = list(cost = c(0.05, 0.1, 0.5, 1, 2))) 
summary(modelo_SVM_v2)

# Parâmetros do melhor modelo:
modelo_SVM_v2$best.parameters

# Melhor modelo:
modelo_SVM_v2$best.model 
modelo_SVM_v2B <- modelo_SVM_v2$best.model 
summary(modelo_SVM_v2B)
# o Melhor modelo foi o de custo 1(0.05)

# 6.7.2 Fazendo as previsões e analisando o resultado:
previsoes_SVM_v2 <- predict(modelo_SVM_v2B, test.data)

# Confusion Matrix:
resultados_SVM_v2 <- caret::confusionMatrix(previsoes_SVM_v2,as.factor(test.data$STATUS))

# Medidas Globais modelo SVM V2
Acuracia_SVM_v2  <- resultados_SVM_v2$overall['Accuracy']
Precisao_SVM_v2  <-resultados_SVM_v2$byClass['Precision']
curva_roc_SVM_v2 <- multiclass.roc(response = test.class.var$STATUS, predictor = as.numeric(previsoes_SVM_v2))
Auc_SVM_v2 <- curva_roc_SVM_v2$auc

# Vetor com os resultados de avaliação do Modelo SVM V2:
vetor_SVM_v2 <- c("Modelo SVM V2", round(Acuracia_SVM_v2, 4), round(Precisao_SVM_v2, 4), round(Auc_SVM_v2, 4))

# 6.7.3 ROC Curve:
previsoes_finais_SVM_v2<- prediction(as.numeric(previsoes_SVM_v2), test.class.var)

# Gera o Plot:
par(mfrow = c(1, 2))
plot.roc.curve(previsoes_finais_SVM_v2, title.text = "Curva ROC")

# 7- Concatenando os resultados de todos os modelos:
compara_modelos <- rbind(vetor_Reg_Log, vetor_KNN_V1, vetor_KNN_V2, vetor_NB_V1, vetor_NB_V2, vetor_SVM_v1, vetor_SVM_v2)

# Ajusta nome das linhas:
rownames(compara_modelos) <- c("1", "2", "3", "4","5","6","7")

# Ajusta nome das colunas:
colnames(compara_modelos) <- c("Modelo", "Acuracia","Precisao", "AUC")

# Transforma a matriz em Vetor:
compara_modelos <- as.data.frame(compara_modelos)
View(compara_modelos)

# Plots:

# Plot Acurácia:
ggplot(compara_modelos, aes(x = Modelo, y = Acuracia, fill = Modelo)) + 
  geom_bar(stat = "identity") 

# Plot Precisão:
ggplot(compara_modelos, aes(x = Modelo, y = Precisao, fill = Modelo)) + 
  geom_bar(stat = "identity") 

# Plot AUC:
ggplot(compara_modelos, aes(x = Modelo, y = AUC, fill = Modelo)) + 
  geom_bar(stat = "identity")

# 8- Conclusões Finais:
# O modelo KNN V1 teve a maior acurácia(0.9312), com pouca diferença para o modelo KNN V2(0.9305) e para o modelo SVM V1(0.9302)
# O modelo KNN V2 teve a maior Precisao(0.933), com pouca diferença para o modelo SVM V1(0.9327) e para o modelo KNN V1(0.9326)
# O modelo KNN V1 teve a maior AUC(0.9311), com pouca diferença para o modelo KNN V2(0.9304) e para o modelo SVM V1(0.9301)
# Logo se pode utilizar qualquer um dos modelos KNN V1, KNN V2 ou SVM V1, mas vou optar pelo deploy do modelo KNN V1 pois este modelo obteve melhor 
# desempenho em 2 das 3 métricas analisadas.

# Salvando o modelo selecionado:
saveRDS(modelo_knn_v1, "modelo_knn_v1.rds")

# 9 -Deploy:

# 9.1 - Carrega o modelo KNNque foi salvo. 
modelo_knn_deploy <- readRDS("modelo_knn_v1.rds")
print(modelo_knn_deploy)

# 9.2 Carrega dados de novos testes de extinção sem a variável target, pois sua previsão será feita através do modelo salvo:
dados_previsao <- read_excel('dados_deploy.xlsx', sheet = 'Planilha1')

# Copia do Data Frame para preservar o Data Frame original e retonar o resultado da previsão nele: 
dados_deploy <- dados_previsao
View(dados_deploy)

# 9.3 Engenharia de atributos - A mesma aplicada nos dados originais:

# Retira a variável Fuel:
dados_deploy$FUEL <- NULL

# Crio um Data Frame em separado para as variáveis normalizadas para não duplicar estas variáveis no Dataset dados_deploy:
dados_scaled_deply <- dados_deploy[, lista_variaveis]
View(dados_scaled_deply)

# Padronização do z-score Usando a função scale():
dados_scaled_deply <- as.data.frame(scale(dados_scaled_deply))
View(dados_scaled_deply)

# Variável do tipo fator:
categorical.var <- c('SIZE')

# Executa transformação em fator:
dados_deploy <- to.factors(df = dados_deploy, variables = categorical.var)
str(dados_deploy)

# Retiro as colunas que foram Normalizadas do Data Frame dados_deploy:
dados_deploy$DISTANCE<-NULL
dados_deploy$DESIBEL<-NULL
dados_deploy$AIRFLOW<-NULL
dados_deploy$FREQUENCY<-NULL

# Adiciona as demais colunas do dataset dados_deploy:
dados_deploy <- cbind(dados_scaled_deply, dados_deploy)
View(dados_deploy)
str(dados_deploy)

# 9.4 Fazendo previsões:
previsoes_deploy <- predict(modelo_knn_deploy, dados_deploy)

# 9.5 Resultado final:

# Previsões:
previsoes_status <- data.frame(as.factor(previsoes_deploy))
colnames(previsoes_status) <- ("Previsão STATUS")

# Dataframe final:
resultado_final <- cbind(dados_previsao, previsoes_status)
View(resultado_final)

# Salva o resultado final:
write_xlsx(resultado_final, "resultado_final.xlsx")
