import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataPrep:
    """
    Classe responsável por realizar a preparação de dados para análise e modelagem.
    As operações incluem: remoção de variáveis irrelevantes, tratamento de valores nulos,
    transformação de variáveis categóricas, normalização dos dados e criação de novas variáveis.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Inicializa a classe DataPrep com o DataFrame fornecido.

        Parâmetros:
        data : pd.DataFrame
            O conjunto de dados a ser processado.
        """
        self.data = data

    def remover_variaveis(self) -> None:
        """
        Remove variáveis irrelevantes para a modelagem.
        Variáveis que não contribuem para a análise preditiva, como identificadores e informações pessoais, são removidas.

        aqui sao dados usados como base, voce deve trocar os dados com a sua realidade.
        """
        colunas_para_remover = [
            'PassengerId',  # Identificador único do passageiro
            'Name',         # Nome do passageiro
            'Ticket',       # Número do bilhete
            'Cabin'         # Número da cabine
        ]
        self.data.drop(columns=colunas_para_remover, inplace=True)

    def tratar_nulos(self) -> None:
        """
        Trata os valores nulos no conjunto de dados.
        A imputação é feita com base na mediana por classe e sexo para a variável 'Age',
        e a variável 'Embarked' é preenchida com 'S' (valor mais comum).
        """
        # Imputa valores nulos de 'Age' com a mediana por classe e sexo
        self.data['Age'] = self.data.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.median()))

        # Imputa valores nulos de 'Embarked' com o valor mais frequente ('S')
        self.data['Embarked'] = self.data['Embarked'].fillna('S')

    def tratar_variaveis_categoricas(self) -> None:
        """
        Converte variáveis categóricas para formatos numéricos que podem ser usados nos modelos.
        A variável 'Sex' é convertida via label encoding, e 'Embarked' via one-hot encoding.
        """
        # Label encoding da variável 'Sex' (0 para masculino e 1 para feminino)
        sexo = {'male': 0, 'female': 1}
        self.data['Sex'] = self.data['Sex'].map(sexo)

        # One-hot encoding da variável 'Embarked'
        embarque = pd.get_dummies(self.data['Embarked'], prefix='Embarked')
        self.data = pd.concat([self.data, embarque], axis=1)
        self.data.drop(columns=['Embarked'], inplace=True)

    def normalizar_dados(self) -> None:
        """
        Normaliza as variáveis numéricas do conjunto de dados usando Min-Max Scaling.
        Essa técnica transforma os dados para um intervalo de 0 a 1.
        """
        # Separa as variáveis numéricas
        variaveis = self.data.drop(columns=['Survived'])  # Remove a variável target 'Survived'

        # Aplica Min-Max Scaling
        scaler = MinMaxScaler()
        variaveis_normalizadas = scaler.fit_transform(variaveis)

        # Substitui as variáveis originais pelas normalizadas
        self.data[variaveis.columns] = variaveis_normalizadas

    def criar_variaveis(self) -> None:
        """
        Cria novas variáveis baseadas em informações existentes.
        Exemplo: a variável 'FamilySize' soma o número de irmãos, cônjuges, pais e filhos.
        """
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1  # +1 para contar o próprio passageiro

    def separar_dados(self) -> tuple:
        """
        Separa os dados em conjuntos de treino e teste, após realizar o pré-processamento necessário.

        Retorna:
        tuple : (pd.DataFrame, pd.DataFrame)
            Conjunto de treino e conjunto de teste.
        """
        # Executa o tratamento completo antes de separar os dados
        self.tratar_nulos()
        self.tratar_variaveis_categoricas()
        self.criar_variaveis()
        self.remover_variaveis()
        self.normalizar_dados()

        # Divide os dados em treino e teste
        treino, teste = train_test_split(self.data, test_size=0.2, random_state=42)
        return treino, teste