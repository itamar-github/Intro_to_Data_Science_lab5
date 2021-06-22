import math
import sys


class RocchioClassifier:
    def __init__(self, train_set, sim='euclidean'):
        self.training_set = train_set
        self.class_centroids = {}
        self.training()
        self.similarity = sim

    def training(self):
        class_size = {}
        for doc_name, document_vector in self.training_set.items():
            doc_class = document_vector[-1]
            if doc_class not in self.class_centroids.keys():
                self.class_centroids[doc_class] = document_vector[0:-1]
                class_size[doc_class] = 1
            else:
                self.class_centroids[doc_class] = [self.class_centroids[doc_class][i] + document_vector[i]
                                                   for i in range(len(document_vector) - 1)]
                class_size[doc_class] += 1
        for c in self.class_centroids.keys():
            for i in range(len(self.class_centroids[c])):
                self.class_centroids[c][i] /= float(class_size[c])

    @staticmethod
    def euclidean_dist(vec1, vec2):
        if len(vec1) != len(vec2):
            print('Error. Vectors of different size')
            print(vec1)
            print(vec2)
            exit(0)

        return sum([(vec1[i] - vec2[i])**2 for i in range(len(vec1))])**0.5

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        calculate cosine similarity between vec1 and vec2
        :param vec1: vector
        :param vec2: vector
        :return: cosine similarity between vec1 and vec2
        """
        if len(vec1) != len(vec2):
            print('Error. Vectors of different size')
            print(vec1)
            print(vec2)
            exit(0)

        dot_product = 0
        for e1, e2 in zip(vec1, vec2):
            dot_product += e1 * e2
        zero_vec = [0, ] * len(vec1)

        v1_norm = RocchioClassifier.euclidean_dist(vec1, zero_vec)
        v2_norm = RocchioClassifier.euclidean_dist(vec2, zero_vec)

        return dot_product/(v1_norm * v2_norm)

    def predict(self, vector):
        winner_class = -1
        if self.similarity == 'euclidean':
            lowest_distance = sys.float_info.max
            for class_name, class_vector in self.class_centroids.items():
                distance = self.euclidean_dist(vector, class_vector)
                if distance < lowest_distance:
                    winner_class = class_name
                    lowest_distance = distance

        elif self.similarity == 'cosine':
            highest_similarity = -1
            for class_name, class_vector in self.class_centroids.items():
                similarity_ = self.cosine_similarity(vector, class_vector)
                if similarity_ > highest_similarity:
                    winner_class = class_name
                    highest_similarity = similarity_

        return winner_class
