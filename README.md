Objetivo do Projeto O objetivo é criar uma pipeline de pré-processamento reutilizável e eficiente que pode ser aplicada a diferentes conjuntos de dados para preparar os dados antes de treinar um modelo de aprendizado de máquina.

Como Usar Inicialize a Classe:

python code from DataPrep import DataPrep # Certifique-se de importar ou implementar essa classe no script prep = DataPrep(data=df) # Substitua 'df' pelo seu DataFrame. Execute o Pré-processamento Completo: A chamada a separar_dados() realiza todas as etapas de pré-processamento e retorna conjuntos de treino e teste.

python code treino, teste = prep.separar_dados() Customize o Código: Se necessário, personalize os métodos para lidar com o seu conjunto de dados. Por exemplo:

Adapte as colunas em remover_variaveis() para remover outras irrelevantes. Ajuste a estratégia de imputação de valores nulos em tratar_nulos(). Recursos Importantes Este tipo de classe geralmente é usado em projetos de aprendizado de máquina. Para se aprofundar:

Pandas (Manipulação de Dados): Documentação: https://pandas.pydata.org/docs/

Scikit-Learn (Normalização e Divisão de Dados): Documentação: https://scikit-learn.org/stable/

Kaggle Titanic Dataset: # usado no codigo para teste Conjunto de dados do Titanic: https://www.kaggle.com/c/titanic

Tutorials de Pré-Processamento de Dados:

Artigos no Medium: https://medium.com/ Hands-On Machine Learning with Scikit-Learn and TensorFlow (Livro): Amazon Este código pode ser adaptado para outros problemas, basta ajustar as etapas de pré-processamento para a realidade dos seus dados.

POSSIVEIS MELHORIAS !

Validação Cruzada: Ao invés de uma única divisão treino-teste, pode-se considerar o uso de validação cruzada para avaliar o modelo com maior robustez. Imputação mais sofisticada: Em vez de usar a mediana para imputação, pode-se considerar o uso de modelos de machine learning, como o KNN ou modelos de regressão, para imputação. Otimizando a normalização: O uso de normalização ou padronização depende do algoritmo que você usará depois. Algoritmos baseados em distância, como KNN, geralmente se beneficiam de normalização, enquanto modelos baseados em árvore, como Random Forest, não são tão sensíveis.
