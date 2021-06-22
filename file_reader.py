import math


class FileReader:

    def __init__(self, input_file):
        self.file = input_file
        self.words = {}
        self.df = {}
        self.stop_words = []
        self.create_stop_words_list()
        self.create_words_bank()
        self.inv_words = {v: k for k, v in self.words.items()}
        self.num_of_documents = self.count_documents()

    def build_set(self, vector_type, file_to_vector):
        if vector_type == 'boolean':
            return self.build_set_boolean(file_to_vector)
        if vector_type == 'tf':
            return self.build_set_tf(file_to_vector)
        if vector_type == 'tfidf':
            return self.build_set_tfidf(file_to_vector)

    def create_stop_words_list(self):
        with open('./stop_words.txt') as stop_words_file:
            for stop_word in stop_words_file:
                self.stop_words.append(stop_word.rstrip())

    def pre_process_word(self, word_original):
        word = word_original.lower()
        word = word.rstrip().rstrip('!').rstrip('.').rstrip(',').rstrip('?').rstrip(':')
        if word in self.stop_words:
            return ''
        return word

    def create_words_bank(self):
        index = 0
        with open(self.file, 'r') as reader: # open the file "file"
            for line in reader: # for each line in file (consider each line to be a document).
                seen_in_this_line = []
                for word in line.split("\t")[0].split(): # for each word in the line
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    if word not in self.df:
                        self.df[word] = 1 # document frequency
                        seen_in_this_line.append(word)
                    if word not in seen_in_this_line:
                        self.df[word] += 1
                        seen_in_this_line.append(word)
                    if word not in self.words.keys(): # if the word doesnt already exists in the words dictionary
                        self.words[word] = index # add it
                        index += 1

    def build_set_boolean(self, file_to_vector):
        doc_set = {}
        reg_representation = {}
        index = 0
        with open(file_to_vector, 'r') as reader:
            for line in reader:
                vec = len(self.words)*[0,]
                for word in line.split("\t")[0].split():
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    vec[self.words[word]] = 1
                doc_class = line.split("\t")[1].rstrip()
                vec.append(doc_class)
                doc_set['doc'+str(index)] = vec
                reg_representation['doc' + str(index)] = line.split("\t")[0]
                index += 1
        return doc_set, reg_representation

    def build_set_tf(self, file_to_vector):
        """

        :param file_to_vector: file to represent as a vector in the corpus's lexical vector space.
        :return:
        """
        doc_set = {}
        reg_representation = {}
        index = 0
        with open(file_to_vector, 'r') as reader:
            for line in reader:
                vec = len(self.words) * [0, ]
                for word in line.split("\t")[0].split():
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    # count the number of occurrences of 'word' in the line
                    vec[self.words[word]] += 1
                # for each term (word) calculate its wf_t,d
                for i, tf in enumerate(vec):
                    vec[i] = FileReader.calc_wf(tf)
                doc_class = line.split("\t")[1].rstrip()
                vec.append(doc_class)
                doc_set['doc' + str(index)] = vec
                reg_representation['doc' + str(index)] = line.split("\t")[0]
                index += 1
        return doc_set, reg_representation

    def build_set_tfidf(self, file_to_vector):
        doc_set = {}
        reg_representation = {}
        index = 0
        with open(file_to_vector, 'r') as reader:
            for line in reader:
                words_in_line = set([])
                vec = len(self.words) * [0, ]
                for word in line.split("\t")[0].split():
                    word = self.pre_process_word(word)
                    if word == '':
                        continue
                    words_in_line.add(word)
                    vec[self.words[word]] += 1
                # for each term (word) in the line (document) calculate its w_i,d
                for word in words_in_line:
                    tf = vec[self.words[word]]
                    vec[self.words[word]] = self.calc_tf_idf(tf, word)
                doc_class = line.split("\t")[1].rstrip()
                vec.append(doc_class)
                doc_set['doc' + str(index)] = vec
                reg_representation['doc' + str(index)] = line.split("\t")[0]
                index += 1
        return doc_set, reg_representation

    @staticmethod
    def calc_wf(tf):
        """
        :param tf: term frequency value
        :return: 0 if tf=0, 1+log_10(tf) otherwise
        """
        if tf == 0:
            return 0

        return 1 + math.log10(tf)

    def calc_tf_idf(self, tf, term):
        """
        calculate tf*idf measure
        :param tf: term frequency
        :param term: (word) to calculate measure for
        :return:
        """
        return tf*math.log10(self.num_of_documents/self.df[term])

    def count_documents(self):
        """
        count the number of lines (documents) in the corpus (input file)
        :return: number of lines in the input file
        """
        f = open(self.file, "r")
        num_lines = len(f.readlines())
        f.close()
        return num_lines

    def print_top_idf(self, n):
        """
        print top n words ranked by their inverse document frequencies
        :return: None
        """
        idf_dict = {}
        for word in self.words:
            idf_dict[word] = math.log10(self.num_of_documents/self.df[word])

        idf_dict = {k: v for k, v in sorted(idf_dict.items(), key=lambda item: item[1], reverse=True)}
        for i, key in enumerate(idf_dict.keys()):
            print(f"idf({key}) = {idf_dict[key]}")
            if i >= n:
                break