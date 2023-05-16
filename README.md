# Projeto02_V2_Eficiencia_Extintores_Incendio
Formação Cientista de Dados da Data Science Academy.

Curso: Big Data Analytics com R e Microsoft Azure Machine Learning.

Projeto com Feedback: Machine Learning na Segurança do Trabalho Prevendo a Eficiência de Extintores de Incêndio.

Dados: https://www.muratkoklu.com/datasets/vtdhnd07.php

Resumo do Projeto: Criação de projeto de Machine Learning para segurança do trabalho “Prevendo a Eficiênciade Extintores de Incêndio” e com o objetivo da Classificação preditiva atingir 85% de acurácia.
O projeto foi realizado de forma independente por 4 semanas para carregar, analisar, limpar, pré-processar, realizar as análises estatísticas e criar e avaliar os modelos de Machine Learning, utilizando linguagem R e seus pacotes readxl(para importar a fonte dedados em Excel), car(para geração de qqPlots), e1071(para analisar Assimetria e Curtose), caTools(para divisão em de dados em treino e teste), pROC(para visualizar, suavizar e comparar as características de curvas ROC) e multiROC(para calcular métricas de Specificity, Sensitivity e AUC)
Os algorítimos de Machine Learning utilizados neste projeto para criação dos modelos de Regressão foram: Regressão Logistica(Benchmark), KNN(2 versões), Naive Bayes(2versões) e SVM(2 versões).
Na avaliação dos modelos, verificou-se que o modelo KNN V1 teve a maior acurácia(0.9312) e o maior AUC(0.9311), sendo o escolhido para deploy, pois obteve melhord esempenho em 2 das 3 métricas analisadas. 

