from warnings import warn
import warnings
import numpy as np
import pickle
from sklearn.decomposition import SparsePCA, MiniBatchSparsePCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from deap import base
from deap import creator
from deap import tools
from abc import ABC, abstractmethod
from xgboost import XGBClassifier

"""
Test the feature selection algorithms by running the following from this directory:

python tests/run-tests.py

"""
class BaseFeatureSelectionModel(ABC):
    @abstractmethod
    def fit(self, features, labels, params):
        pass

    @abstractmethod
    def transform(self, features, num_keep_features):
        pass

    @abstractmethod
    def save(self, location):
        pass

    @abstractmethod
    def load(self, location):
        pass


class RandomForestFeatureSelection(BaseFeatureSelectionModel):
    def __init__(self):
        self.feature_importance = None

    def fit(self, features, labels, params):
        num_trials = params.get('num_trials',25)
        randseed = params.get('randseed', 42)
        np.random.seed(randseed)

        assert(len(features.shape) is 2)

        n_dims = features.shape[1]
        random_seeds = np.random.choice(num_trials ** 2, size=num_trials, replace=False).tolist()
        self.feature_importance = np.zeros(N_DIMS)
        for i in range(num_trials):
            rf = RandomForestClassifier(random_state=random_seeds.pop(), n_estimators=10)
            rf.fit(features, labels)
            self.feature_importance += rf.feature_importances_

    def transform(self, features, num_keep_features):
        return np.take(features, np.argsort(self.feature_importance)[:num_keep_features], axis=1)

    def save(self, location, serialized_feature_selector=None):
        pickle_out = open(location,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, location):
        serialized_fs = open(location, 'rb')
        fs = pickle.load(serialized_fs)
        self.feature_importance = fs.feature_importance


"""
# Sparse PCA method: usage example
feature_selector = SparsePCAFeatureSelection()
feature_selector.fit(input_array, activity_labels, params)
reduced_features = feature_selector.transform(input_array)
"""
class SparsePCAFeatureSelection:
    def __init__(self):
        self.spca = SparsePCA(normalize_components=True, random_state=42)
        self.support_cols = None
        self.dim = None

    def fit(self, features, labels, params):
        _, self.dim = features.shape
        lifted_data = np.hstack([features, labels.reshape(1, -1).transpose()])
        self.spca.fit(lifted_data)
        components = self.spca.components_
        rows, cols = components.nonzero()
        indicated_rows = np.take(rows, np.where(cols == N_DIMS)[0])
        _, self.support_cols = components[indicated_rows, :].nonzero()
        self.support_cols = self.support_cols.tolist()
        self.support_cols.remove(self.dim)
        self.support_cols = np.array(self.support_cols)
        if self.support_cols.size == 0:
            warn("No correlations found")


class XGBoostFeatureSelection(BaseFeatureSelectionModel):
    def __init__(self):
        self.feature_importance = None

    def fit(self, features, labels, params):
        num_trials = params.get('num_trials',25)
        randseed = params.get('randseed', 42)
        np.random.seed(randseed)
        
        assert(len(features.shape) is 2)

        n_dims = features.shape[1]
        random_seeds = np.random.choice(num_trials ** 2, size=num_trials, replace=False).tolist()
        self.feature_importance = np.zeros(n_dims)
        for i in range(num_trials):
            xgb = XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.5, 
                learning_rate = 0.1, max_depth = 200, alpha = 10, n_estimators = 10)

            xgb.fit(features, labels)
            self.feature_importance += xgb.feature_importances_

    def transform(self, features, num_keep_features):
        return np.take(features, np.argsort(self.feature_importance)[:num_keep_features], axis=1)

    def save(self, location, serialized_feature_selector=None):
        pickle_out = open(location,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, location):
        serialized_fs = open(location, 'rb')
        fs = pickle.load(serialized_fs)
        self.feature_importance = fs.feature_importance


"""
# Genetic Algorithm Method: usage example
feature_selector = GeneticFeatureSelection()
feature_selector.fit(input_array, activity_labels)
reduced_features = feature_selector.transform(input_array)
"""

warnings.simplefilter(action='ignore')


model = RandomForestClassifier()


def mask_value(mask, features, labels):
    """Returns the mean roc_auc_score of a random forest model trained with
    the indicated subset of features."""
    features_tmp = features[:, mask == 1]
    np.random.seed(42)
    scores = []
    mini_batches_generator = StratifiedKFold(n_splits=4, random_state=42, shuffle=True)
    try:
        for training_index, validation_index in mini_batches_generator.split(features_tmp, labels):
            training_features = features_tmp[training_index]
            training_labels = np.ravel(labels[training_index])
            validation_features = features_tmp[validation_index]
            validation_labels = np.ravel(labels[validation_index])
            model.fit(training_features, training_labels)
            predictions = model.predict_proba(validation_features)[:, 1]
            scores.append(roc_auc_score(validation_labels, predictions))
        return np.mean(scores)
    except ValueError:
        return 0


def mask_opt_function(mask, features, labels):
    """The function being maximized by genetic algorithm.
     It attempts to balance the output of mask_value with the feature dimension."""
    feature_dim = np.sum(mask)
    model_auc = mask_value(mask, features, labels)
    return model_auc ** 2 / (1 + feature_dim),


def genetic_algorithm(features, labels, num_gens):
    _, num_features = np.shape(features)
    np.random.seed(42)
    print_freq = 10
    target_ratio = 0.1
    pop_size = 50
    INDPB = 0.05
    TOURNSIZE = 3
    CXPB = 0.5
    MUTPB = 0.2

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, typecode='b', fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.choice, 2, p=[1 - target_ratio, target_ratio])
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, num_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", mask_opt_function, features=features, labels=labels)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=INDPB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)

    max_record = []
    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    gen = 0
    while gen < num_gens:
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if np.random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        scores = [toolbox.evaluate(ind) for ind in pop]
        if (gen % print_freq == 0) or (gen == num_gens - 1):
            best_mask = pop[np.argmax(scores)]
            max_score = np.round(np.max(scores), 3)
            print('Best ROC-AUC: {},'
                  ' Feature Dimension: {}'.format(max_score, np.sum(best_mask)))
            max_record.append([max_score, int(np.sum(best_mask))])
        gen += 1
    return max_record, np.array(pop[np.argmax(scores)])


class GeneticFeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_mask = None
        self.opt_record = None

    def fit(self, features, labels, params):
        num_gens = params.get('num_gens', 100)

        self.opt_record, self.feature_mask = genetic_algorithm(features, labels, num_gens)

    def transform(self, features):
        return features[:, self.feature_mask == 1]

    def save(self, location, serialized_feature_selector=None):
        pickle_out = open(location,"wb")
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, location):
        serialized_fs = open(location, 'rb')
        fs = pickle.load(serialized_fs)
        self.opt_record = fs.opt_record
        self.feature_mask = fs.feature_mask




N_SAMPLES = 1000
N_DIMS = 100

def main():
    # synthetic data
    input_array = np.random.sample(size=(N_SAMPLES, N_DIMS))
    activity_labels = np.random.choice(2, size=N_SAMPLES, p=(0.95, 0.05))

    # Random Forest Method: usage example
    model_location = 'example_model'

    feature_selector = RandomForestFeatureSelection()
    feature_selector.fit(input_array, activity_labels, {})
    feature_selector.save('example_model')

    feature_selector2 = RandomForestFeatureSelection()
    feature_selector2.load(model_location)
    reduced_features = feature_selector2.transform(input_array, 10)
    print(reduced_features)
  
if __name__== "__main__":
    main()

