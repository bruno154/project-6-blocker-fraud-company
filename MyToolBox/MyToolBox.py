import glob
import math
import pandas as pd
import numpy as np
import dask.array as da
import dask.dataframe as dd
import seaborn as sns
import matplotlib.pyplot as plt
from time import strftime


class CollectData:

    def __init__(self, path):
        self.path = path

    def get_data_csv(self, target, kind='train'):
        if kind == 'train':
            train = pd.read_csv(self.path)
            X = train.drop(target, axis=1)
            y = train[target]
            return train, X, y
        else:
            test = pd.read_csv(self.path)
            Xtest = test.copy()
            return Xtest

    def get_data_dask(path, target, kind='train'):
        if kind == 'train':
            train = dd.read_csv(path)
            X = train.drop(target, axis=1)
            y = train[target]
        else:
            test = dd.read_csv(path)
            X = test.copy()

        return train, X, y


class DataDescription:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def my_describe(self):

        return pd.DataFrame({'missingPerc': self.dataframe.isna().mean(),
                             'uniques': self.dataframe.nunique(),
                             '%uniquePerc': (self.dataframe.nunique()/self.dataframe.shape[0])*100,
                             'data_types': self.dataframe.dtypes,
                             'mean': self.dataframe.mean(),
                             'median': self.dataframe.median(),
                             'std': self.dataframe.std(),
                             'min': self.dataframe.min(),
                             'max': self.dataframe.max()})


class CleanData:

    def __init__(self, df):
        self.df = df

    def merge_csv_folder(self, prefix, path, memory='no'):
        """
        Unir todos os csvs de uma determinada pasta.

        :prefix: Prefixo do arquivo concatenado a ser salvo.
        :path:   Caminho do diretorio.
        :memory: Se os dados concatenados devem ser salvos em memoria(dataframe) ou em um arquivo .csv 
        """
        if path == '':
            path = input("Por favor digite o endereços dos arquivos?\n")
        path = path.replace('\\', '/')
        if path[:-1] != '/':
            path = path + '/'

        file_list = glob.glob(path + '*.csv')

        combined = pd.concat([pd.read_csv(f) for f in file_list])
        if memory == 'no':
            combined.to_csv(path + 'combined_{}_{}.csv'.format(prefix, strftime("%Y%m%d-%H%M%S")), index=False)
        else:
            return combined
        print('Feito')

    def header_cleaner(self):
        """
        Limpa os cabeçalhos do dataframe.

        :df: Dataframe.
        """
        cols = self.df.columns.str.strip().str.lower()
        trat1 = [col.replace(" ", '_') for col in cols]
        trat2 = [col.replace('.', '') for col in trat1]
        return trat2

    def split_columns(self, original, res1, res2, separator):
        """
        Divide uma coluna em duas.

        :df:        Objeto dataframe.
        :original:  Coluna original.
        :res1:      Nome para a primeira coluna resultante 
        :res2:      Nome para a segunda coluna resultante
        :separator: caracter separador
        """
        self.df[[res1, res2]] = self.df[original].str.split(separator, expand = True)
        return self.df

    def df_filter(self, cond):
        """ Filtra df conforme determinada condição.

        :df:   Dataframe
        :cond: Condição no qual haverá o filtro.
        """
        new_df = self.df.filter(regex=cond)
        return new_df


class EDA:

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def multi_histograms(self, variables: list) -> None:

        """
        Function to check for outliers visually through a boxplot

        data: DataFrame

        variable: list of numerical variables
        """

        # set of initial plot posistion
        n = 1

        plt.figure(figsize=(24, 16))
        for column in self.dataframe[variables].columns:
            plt.subplot(5, 5, n)
            _ = sns.histplot(x=self.dataframe[column], bins=50)
            n += 1

        plt.subplots_adjust(hspace=0.3)

        plt.show()

    def multi_boxplots(self, variables: list) -> None:

        """
        Function to check for outliers visually through a boxplot

        data: DataFrame

        variable: list of numerical variables
        """

        # set of initial plot posistion
        n = 1

        plt.figure(figsize=(18, 10))
        for column in self.dataframe[variables].columns:
            plt.subplot(3, 3, n)
            _ = sns.boxplot(x=column, data=self.dataframe)
            n += 1

        plt.subplots_adjust(hspace=0.3)

        plt.show()

    def balanced_target(self, target, hue=None):
        """
        Function to check the balancing of the target variable.

        :target:  An pd.Series of the target variable that will be checked.
        :dataset: An Dataframe object. 
        """
        sns.set(style='darkgrid', palette='Accent')
        ax = sns.countplot(x=target, hue=hue, data=self.dataframe)
        ax.figure.set_size_inches(10, 6)
        ax.set_title('Cardio Distribution', fontsize=18, loc='left')
        ax.set_xlabel(target, fontsize=14)
        ax.set_ylabel('Count', fontsize=14)
        ax=ax

    def hipo_test(self, *samples):

        samples = samples

        try:
            if len(samples) == 2:
                stat, p = ttest_ind(*samples)
            elif len(samples) > 2:
                stat, p = f_oneway(*samples)
        except:
            raise Exception("Deve ser fornecido pelo menos duas samples!!!")

        if p < 0.05:
            print(f'O valor de p é: {p}')
            print('Provável haver diferença')
        else:
            print(f'O valor de p é: {p}')
            print('Provável que não haja diferença')

        return stat, p

    def Myheat_map(self, dataset, variaveis):

        df_corr = self.dataframe[variaveis].corr()

        fig, ax = plt.subplots(figsize=(16, 10))
        # mask
        mask = np.triu(np.ones_like(df_corr, dtype=np.bool))
        # adjust mask and df
        mask = mask[1:, :-1]
        corr = df_corr.iloc[1:,:-1].copy()
        # color map
        cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)

        # plot heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    linewidths=5, cmap=cmap, vmin=-1, vmax=1,
                    cbar_kws={"shrink": .8}, square=True)
        yticks = [i.upper() for i in corr.index]
        xticks = [i.upper() for i in corr.columns]
        plt.yticks(plt.yticks()[0], labels=yticks, rotation=0)
        plt.xticks(plt.xticks()[0], labels=xticks)

        # title
        title = 'CORRELATION MATRIX\n'
        plt.title(title, loc='left', fontsize=18)
        plt.show()


class DataPreparation:
    pass


class ModelSelection:
    pass


class ModelEvaluation:

    def __init__(self, model, Xvalid, yvalid):
        self.model = model
        self.Xvalid = Xvalid
        self.yvalid = yvalid

    def confusion_matrix():
        pass

    def discrimination_threshold():
        pass

    def roc_curve(): # exemplaria
        pass


class StatsCalculations:


    def binomial_prob(self, n, p, x):
        """
        Retorna a probabilidade binomial.

        :n: Quantidade de trials.
        :p: Probabiliadade do evento 'sucesso' em cada trial.
        :x: Quantidade de eventos sucesso.
        """

        b = (math.factorial(n)/(math.factorial(x)*math.factorial(n-x)))*(p**x)*((1-p)**(n-x))
        return b

    def geometric_prob(self, p, x):
        """
        Retorna a probabilidade de um evento depois 
        de uma determinada quantiade de eventos.

        :p: Probabiliadade do evento 'sucesso' em cada trial.
        :x: Quantidade de trials.
        """

        g = (1-p)**(x-1)*p
        return g

    def poisson_prob(self, miu, x):
        """
        Retorna a probabilidade de uma quantidade de eventos em uma área.

        :miu: Média do numero de eventos 'sucesso' em uma determinada área.
        :x:   Quantidade de eventos 'sucesso' na região. 
        """

        p = ((miu**x)* math.exp(-miu))/math.factorial(x)
        return p

    def normal_prob(self, miu, stdev, x):
        """
        Retorna a probabilidade de um evento conforme um distribuição
        normal de probabilidades.

        :miu:   Média da distribuição.
        :stdev: Desvio-Padrão da distribuição.
        :x      Quantidade de eventos 'sucesso'.
        """
        norm = 0.5* (1+math.erf((x-miu)/ (stdev * 2**0.5)))
        return norm

    def confidence_interval(self, n, miu, stdev, conf=0.95):

        """
        Retorna o intervalo onde temos determinada certeza que os
        valores verdadeiros estarão.

        :n:      Quantidade de observações.
        :miu:    Média da distribuição.
        :stdev:  Desvio-Padrão da distribuição.
        :conf:   Grau de confiança que queremos obter - (0.99, 0.95, 0.90).
        """
        if conf == 0.95:
            z = 1.96
        elif conf == 0.99:
            z = 2.576
        elif conf == 0.90:
            z = 1.645

        interval_min = miu-z*(stdev/n**0.5)
        interval_max = miu+z*(stdev/n**0.5)
        return print(f'{interval_min} , {interval_max}')

    def corr_heatmap(self, df, method='pearson'):

        """
        Retorna mapa de calor das correlações, conforme metodo escolhido.

        :df:     Dataframe com os dados.
        :method: Método escolhido para o calculo da correlação - (pearson,spearman,kendall)
        """
        return df.corr(method=method).style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)



