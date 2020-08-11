# Importando pagotes
import numpy as np
import pandas as pd

# ... para o Extrator LBP
import os
import imghdr
from PIL import Image
from skimage.feature import local_binary_pattern

# ... para o Classificador Hierarquico
from node import Node
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import csv

"""
Este 'notebook' está organizado da seguinta maneira. 
- Por primeiro, temos o extrator de características das imagens RaioX;
- Por segundo, temos o classificador hierarquico utilizado;

Para escolher a funcao do script altere a variavel 'FUNCAO_SCRIPT'.
Opções: 'extrair', 'classificar'

Para usar o extrator, lembre-se de alterar o caminhos dos diretórios na variaveis "train_directory" e "test_directory".

Para usar o classificador, lembre-se de alterar o caminho dos dos arquivos nas variáveis "train_data_path" e "test_data_path".

"""

FUNCAO_SCRIPT = 'classificar'

# Caminho onde estão localizadas as imagens de treino e teste para o extrator
train_directory = 'C:/Users/Fabio Barros/Git/covid-sp/Images/Train'
test_directory = 'C:/Users/Fabio Barros/Git/covid-sp/Images/Test'

#Qual extrator de características utilizar - nri_uniform = 59 features / uniform = 10 features
lbp_extractor = 'nri_uniform'

# Caminho para os arquivos de treino e teste para o classificador
train_data_path = 'C:/Users/Fabio Barros/Git/covid-sp/classifiers/hierarchical/feature_matrix_train.csv'
test_data_path = 'C:/Users/Fabio Barros/Git/covid-sp/classifiers/hierarchical/feature_matrix_test.csv'


# Parâmetros do classificador

# Algoritmo de classificação
classifier = "rf"  # rf, mlp or svm

# Rebalanceamento flat
resample = True

# Rebalanceamento hierárquico
local_resample = False

# Diretório de resultados
result_dir = 'Result_Hierarchical'

# Algoritmo de balanceamento
resampler_option = 'smote-enn'


"""ABAIXO ESTÃO TODAS AS FUNÇÕES"""
if FUNCAO_SCRIPT == 'extrair':
    print('Você escolheu EXTRAIR as características das imagens...')

    """ VARIAVEIS E DIRETORIOS PARA O EXTRATOR """

    UNIFORM_FEATURE_NUMBER = 10
    NRI_UNIFORM_FEATURE_NUMBER = 59

    class LocalBinaryPatterns:
        def __init__(self, numPoints, radius):
            self.numPoints = numPoints
            self.radius = radius

        # LBP Feature Extractor  - 10 features
        def describe_lbp_method_rd(self, image, eps=1e-7):
            lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
            (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))

            hist = hist.astype("float")
            hist /= (hist.sum() + eps)

            return hist

        # LBP Feature Extractor  - 59 features
        def describe_lbp_method_ag(self, image):
            lbpU = local_binary_pattern(image, self.numPoints, self.radius, method='nri_uniform')
            hist0, nbins0 = np.histogram(np.uint8(lbpU), bins=range(60), density=True)

            return hist0

    # Aqui começa a definição de todas as funções necessárias
    # Funcao para abrir/carregar uma imagem no diretorio escolhido
    def open_img(filename):
        img = Image.open(filename)
        return img


    # Verifica se a imagem escolhida contem um formato valido como: 'JPG', 'PNG'
    def verify_valid_img(path):
        possible_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']
        if imghdr.what(path) in possible_formats:
            return True
        else:
            return False


    # Chamada para a extração das características
    def feature_extraction(image, lbp_extractor):
        lbp = LocalBinaryPatterns(8, 2)
        image_matrix = np.array(image.convert('L'))

        if lbp_extractor == 'uniform':
            img_features = lbp.describe_lbp_method_rd(image_matrix)
        elif lbp_extractor == 'nri_uniform':
            img_features = lbp.describe_lbp_method_ag(image_matrix)

        return img_features.tolist()


    def create_columns(column_number, property):
        columns = []
        for i in range(0, column_number):
            columns.append(str(i))

        columns.append(property)
        return columns


    # Função para criar a a matriz de características do treinamento, contem a classe esperada para cada amostra
    def create_feature_matrix_train(train_directory, lbp_extractor):
        # Variavel para guardar as data_rows
        rows_list = []

        print("Iniciando a extração de recursos para o conjunto TREINAMENTO\n\n\n")

        # Repete para cada sub-diretório na pasta de treinamento (1 pasta para cada classe)
        for dir in os.listdir(train_directory):

            # Este é o caminho para cada subdiretório
            sub_directory = train_directory + '/' + dir

            # Lista cada arquivo presente dentro de um sub-diretório
            training_filelist = os.listdir(sub_directory)

            # Repete para cada um dos arquivos presente no sub-diretório
            for file in training_filelist:
                file_path = sub_directory + '/' + file

                # Apenas prossegue caso seja uma imagem valida
                if verify_valid_img(file_path):
                    _fileNumber = str(training_filelist.index(file) + 1) + " de " + str(len(training_filelist))
                    print("Processando: {} -- PATH: {}".format(_fileNumber, file_path))

                    image = open_img(file_path)
                    img_features = feature_extraction(image, lbp_extractor)

                    # O nome do diretório é o nome da classe
                    img_features.append(dir)

                    rows_list.append(img_features)
                else:
                    print("O arquivo encontrado não é uma imagem válida: " + file_path)

            print("-" * 100)
        # Criando um dataframe para guardar todas as características
        if lbp_extractor == 'uniform':
            columns = create_columns(UNIFORM_FEATURE_NUMBER, 'class')
        elif lbp_extractor == 'nri_uniform':
            columns = create_columns(NRI_UNIFORM_FEATURE_NUMBER, 'class')

        feature_matrix = pd.DataFrame(rows_list, columns=columns)

        print("Concluída a criação da Matriz de Características de Treinamento")
        print('-=' * 50)
        print('\n')

        return feature_matrix


    # Function to create the testing feature matrix, it has the id of each sample instead of the class
    def create_feature_matrix_test(test_directory, lbp_extractor):
        # Variable to store the data_rows
        rows_list = []

        print("Iniciando a extração de recursos para o conjunto de TESTES\n\n\n")

        # Lista cada arquivo presente dentro de um sub-diretório
        test_filelist = os.listdir(test_directory)

        # Repete para todos os arquivos presentes na pasta de teste
        for file in test_filelist:
            file_path = test_directory + '/' + file

            # Verifica se é uma imagem valida
            if verify_valid_img(file_path):
                _fileNumber = str(test_filelist.index(file) + 1) + " de " + str(len(test_filelist))
                print("Processando: {} -- PATH: {}".format(_fileNumber, file_path))

                image = open_img(file_path)
                img_features = feature_extraction(image, lbp_extractor)

                # ID da amostra
                img_features.append(file[:-4])

                rows_list.append(img_features)
            else:
                print("O arquivo encontrado não é uma imagem válida: " + file_path)

        print("-" * 100)

        # Criando um dataframe para guardar todas as características
        if lbp_extractor == 'uniform':
            columns = create_columns(UNIFORM_FEATURE_NUMBER, 'id_exame')
        elif lbp_extractor == 'nri_uniform':
            columns = create_columns(NRI_UNIFORM_FEATURE_NUMBER, 'id_exame')

        feature_matrix = pd.DataFrame(rows_list, columns=columns)

        print("Concluída a criação da Matriz de Características de Treinamento")
        print('-=' * 50)
        print('\n')

        return feature_matrix

    ## Aqui começa a execução dos scripts de extração de características
    # Salva os arquivos no diretório atual
    feature_matrix_train_path = os.getcwd()
    feature_matrix_test_path = os.getcwd()

    feature_matrix_train = create_feature_matrix_train(train_directory, lbp_extractor)
    print("Salvando a Matriz de Treinamento em formato CSV...", end=' ')
    feature_matrix_train.to_csv(feature_matrix_train_path + '/feature_matrix_train.csv', index=False)
    print('ARQUIVO SALVO!')
    print('\n\n- Arquivo de treino salvo em: ', str(feature_matrix_train_path + '/feature_matrix_train.csv'))

    feature_matrix_test = create_feature_matrix_test(test_directory, lbp_extractor)
    print("Salvando Matriz de Teste em formato CSV...", end='')
    feature_matrix_test.to_csv(feature_matrix_test_path + '/feature_matrix_test.csv', index=False)
    print('- Arquivo de teste salvo em: ', str('./' + feature_matrix_test_path + '/feature_matrix_test.csv'))
    print('ARQUIVO SALVO!')

elif FUNCAO_SCRIPT == 'classificar':

    POSITIVE_CLASS = 'COVID'
    NEGATIVE_CLASS_1 = 'NORMAIS'
    NEGATIVE_CLASS_2 = 'notCOVID'
    INTERMEDIATE_NEGATIVE_CLASS = 'NOT_NORMAL'

    CSV_SPACER = ";"

    # Aqui começa a definição de objetos e funções necessárias para a classificação

    class Node:
        class_name = None
        is_leaf = False
        data = pd.DataFrame()
        left = None
        right = None
        local_clf = None

        def __init__(self, class_name):
            self.class_name = class_name

        def set_data(self, data):
            self.data = data

        def set_new_child(self, child):
            self.children.append(Node(child))

        def is_parent(self, is_parent):
            self.is_parent = is_parent


    class Result:

        def __init__(self, predicted_class, proba):
            self.predicted_class = predicted_class
            self.proba = proba

        def set_proba(self, proba):
            self.proba = proba


    # Fatia entradas e saídas
    def slice_data(dataset):
        # Fatiando dados de entrada e saída
        input_data = dataset.iloc[:, :-1].values
        output_data = dataset.iloc[:, -1].values

        return [input_data, output_data]

    # Define o classificador que será utilizado
    def define_classifier():
        if classifier == 'rf':
            return RandomForestClassifier(criterion="gini", n_estimators=150)
        elif classifier == 'mlp':
            return MLPClassifier(hidden_layer_sizes=(60), activation='logistic', verbose=False, early_stopping=True,
                                 validation_fraction=0.2)
        elif classifier == 'svm':
            return SVC(gamma='auto', probability=True)

    # Cria uma árvore que representa a hierarquia de classes da base.
    # Foi utilizado aqui um classsificador hierárquico por nó pai
    def create_class_tree():

        tree = Node('R')
        normal = Node(NEGATIVE_CLASS_1)
        not_normal = Node(INTERMEDIATE_NEGATIVE_CLASS)
        altered_not_covid = Node(NEGATIVE_CLASS_2)
        covid = Node(POSITIVE_CLASS)
        normal.is_leaf = True
        altered_not_covid.is_leaf = True
        covid.is_leaf = True
        not_normal.left = altered_not_covid
        not_normal.right = covid
        tree.right = not_normal
        tree.left = normal

        return tree

    # 'Renomeia' as classes para a classe atual.
    # Usado no processo de montagem da estrutura hierárquica de classes
    def relabel_to_current_class(class_name, relabeled_data_frame):
        relabeled_data_frame['class'] = class_name
        return relabeled_data_frame

    # Função para trazer os dados de cada classe para cada nó do classificador hierárquico por nó pai
    def retrieve_data_lcpn(tree, data_frame):
        class_data = pd.DataFrame()
        if tree.is_leaf is True:
            return data_frame[data_frame['class'] == tree.class_name]
        else:
            class_data = class_data.append(retrieve_data_lcpn(tree.left, data_frame))
            class_data = class_data.append(retrieve_data_lcpn(tree.right, data_frame))

            tree.data = class_data

            # Renomeie para a classe atual antes de retornar para a classe pai
            class_data_relabeled = relabel_to_current_class(tree.class_name, class_data.copy())

            return class_data_relabeled

    # Função para balancear a base por nós-pai
    def resample_data_lcpn(tree, data_frame):
        if tree.is_leaf is True:
            return
        else:
            # Vai treinar apenas para classes de nó pai
            [input_data_train, output_data_train] = slice_data(tree.data)
            print('\n------------Local Distribution for class: {}----------------'.format(tree.class_name))
            class_data = resample_data(input_data_train, output_data_train, resampler_option)
            tree.data = class_data

            resample_data_lcpn(tree.left, data_frame)
            resample_data_lcpn(tree.right, data_frame)

            return

    # Função para treinar os classificadores por nó pai
    def train_lcpn(tree):
        if tree.is_leaf is True:
            return
        else:
            # Will train only for parent node classes
            [input_data_train, output_data_train] = slice_data(tree.data)

            clf = clone(define_classifier())
            print('Training classifier for node: ' + tree.class_name)
            trained_clf = clf.fit(input_data_train, output_data_train)
            tree.local_clf = trained_clf

            train_lcpn(tree.left)
            train_lcpn(tree.right)

            return

    # Função para retornar a classe e a probabilidade para uma dada amostra de dados
    def prediction_proba(data, model):

        data = data.reshape(1, -1)
        predicted = model.predict(data)
        proba = model.predict_proba(data)
        proba = proba[0]

        possible_classes = model.classes_
        # Encontra o índice da classe prevista e salva apenas este índice
        index = np.where(possible_classes == predicted)
        index = index[0]

        proba_predicted_class = proba[index[0]]
        result = Result(predicted, proba_predicted_class)
        return result

    # Função para realizar a predição da classe para uma determinada amostra de dados.
    # Retorna o resultado com classe prevista e probabilidade
    def predict_lcpn(row, tree):

        if tree.left is not None or tree.right is not None:

            result = prediction_proba(row, tree.local_clf)

            if result.predicted_class[0] == tree.left.class_name:
                prediction_result = predict_lcpn(row, tree.left)
            elif result.predicted_class[0] == tree.right.class_name:
                prediction_result = predict_lcpn(row, tree.right)

            # Se probabilidade for vazio, isso significa que a última
            # previsão foi um nó folha, como queremos
            if (prediction_result.proba is None):
                prediction_result.set_proba(result.proba)

            return prediction_result

        else:
            result = Result(tree.class_name, None)
            return result


    # Função para converter em saída binária, de acordo com o formato do desafio
    def convert_to_binary_output(predicted):
        predicted_binary = []

        for predicted_class in predicted:
            if (predicted_class == POSITIVE_CLASS):
                predicted_binary.append(1)
            elif (predicted_class == NEGATIVE_CLASS_1):
                predicted_binary.append(0)
            elif (predicted_class == NEGATIVE_CLASS_2):
                predicted_binary.append(0)

        return predicted_binary


    # Função para escrever o resultado final em um csv, no formato requisitado pelo desafio
    def write_csv(sample_ids, predicted, probability_array, file_path):
        header = ['id_exame', 'covid', 'prob']

        with open(file_path, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=CSV_SPACER, quotechar='|', quoting=csv.QUOTE_MINIMAL,
                                    dialect='excel')
            filewriter.writerow(header)

            for i in range(0, len(predicted)):
                if predicted[i] == 1:
                    result_row = [sample_ids[i], predicted[i], probability_array[i]]
                else:
                    result_row = [sample_ids[i], predicted[i]]

                filewriter.writerow(result_row)

    # Função para contar quantas amostras por classe existem
    def count_per_class(output_data):
        original_count = np.unique(output_data, return_counts=True)
        classes = original_count[0]
        count = original_count[1]

        for i in range(0, len(classes)):
            print('Classe {}, Cont {}'.format(classes[i], count[i]))
        print('')

    # Função para definir qual algoritmo de balanceamento será utilizado, baseado no parâmetro fornecido no início do script
    def define_resampler(resampler_option):
        if (resampler_option == 'smote'):
            return SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42, n_jobs=4)
        elif (resampler_option == 'smote-enn'):
            return SMOTEENN(sampling_strategy='auto', random_state=42, n_jobs=4)
        elif (resampler_option == 'smote-tomek'):
            return SMOTETomek(sampling_strategy='auto', random_state=42, n_jobs=4)
        elif (resampler_option == 'borderline-smote'):
            return BorderlineSMOTE(sampling_strategy='auto', random_state=42, n_jobs=4)
        elif (resampler_option == 'adasyn'):
            return ADASYN(sampling_strategy='auto', random_state=42, n_jobs=4)
        elif (resampler_option == 'ros'):
            return RandomOverSampler(sampling_strategy='auto', random_state=42)


    # Função para realizar balanceamento da base de dados
    def resample_data(input_data_train, output_data_train, resampler_option):

        # Distribuição de classes original
        print('Distribuição Original de Classes')
        count_per_class(output_data_train)

        # Instanciar um algoritmo de balanceamento
        resampler = define_resampler(resampler_option)
        print("Balanceando dados...")
        [input_data_train, output_data_train] = resampler.fit_resample(input_data_train,
                                                                       output_data_train)  # Original class distribution
        print("Balanceamento completado.")
        # Distribuição de classes após o balanceamento
        count_per_class(output_data_train)

        train_data_frame = pd.DataFrame(input_data_train)
        train_data_frame['class'] = output_data_train

        return train_data_frame


    # Aqui se inicia o script de classificação
    # Carrega os dados

    print('Carregando arquivos .CSV...', end=' ')
    train_data_frame = pd.read_csv(train_data_path)
    test_data_frame = pd.read_csv(test_data_path)

    # Carrega o classificador
    print('Iniciando classificador...', end=' ')
    clf = define_classifier()

    print('Iniciando hierarquia...')
    class_tree = create_class_tree()

    # If resample flag is True, we need to resample the training dataset by generating new synthetic samples
    if resample is True:
        [input_data_train, output_data_train] = slice_data(train_data_frame)
        train_data_frame = resample_data(input_data_train, output_data_train, resampler_option)
        train_data_frame = pd.DataFrame(input_data_train)
        train_data_frame['class'] = output_data_train

    retrieve_data_lcpn(class_tree, train_data_frame)

    if local_resample is True:
        resample_data_lcpn(class_tree, train_data_frame)

    # Treinamento
    print('Trenimento iniciado...', end='')
    train_lcpn(class_tree)
    print('Finalizado.')

    # Predição
    [input_data_test, sample_ids] = slice_data(test_data_frame)

    prediction = []
    proba_array = []

    print('Classificando amostras...')

    # Itera sobre as amostras de teste
    for input_test_row in input_data_test:
        prediction_result = predict_lcpn(input_test_row, class_tree)
        prediction.append(prediction_result.predicted_class)
        proba_array.append(prediction_result.proba)
    print('Classificação Finalizada')

    # Converte os resultados para o formato do desafio
    predicted = convert_to_binary_output(prediction)

    # Cria diretório para armazenar arquivo de resultados
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    # Define nome do arquivo de resultados de acordo com parâmetros de entrada utilizados
    if resample is True:
        file_path = result_dir + '\\result_hierarchical_' + classifier + '_resample_' + str(resample) + '_' + str(
            resampler_option) + '.csv'
    elif local_resample is True:
        file_path = result_dir + '\\result_hierarchical_' + classifier + '_local_resample_' + str(
            local_resample) + '_' + str(
            resampler_option) + '.csv'
    else:
        file_path = result_dir + '\\result_hierarchical_' + classifier + '_resample_' + str(resample) + '.csv'

    print('Writing result to csv')
    # Save result in csv
    write_csv(sample_ids, predicted, proba_array, file_path)
    print('Arquivo Gerado')
    print("END")
