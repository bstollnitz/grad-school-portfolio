
import itertools
import pprint
import random

from typing import List, Dict

class Hyperparameters:

    def __init__(self, definitions: Dict) -> None:
        self.definitions = definitions
        self.combinations = Hyperparameters._generate_combinations(definitions)

    def sample_combinations(self, count: int = None) -> List[Dict]:
        """
        Returns a random subset of hyperparameter combinations.
        """
        max_count = len(self.combinations) 
        if count is None:
            count = max_count
        elif count > max_count:
            print(f"Using maximum number of hyperparameter combinations ({max_count}) instead of the number provided ({count}).")
            count = max_count
        
        return random.sample(self.combinations, count)

    @staticmethod
    def _generate_combinations(dictionary: Dict) -> List[Dict]:
        """
        Generates a list of hyperparameter combinations from a dictionary of
        possibilities.
        """
        expanded_dictionary = {}
        for key in dictionary:
            original_value = dictionary[key]
            expanded_values = []

            if isinstance(original_value, list):
                for value in original_value:
                    if isinstance(value, dict):
                        value = Hyperparameters._generate_combinations(value)
                        expanded_values += value
                    else:
                        expanded_values.append(value)
            else:
                expanded_values.append(original_value)

            expanded_dictionary[key] = expanded_values

        return [dict(zip(expanded_dictionary, v)) 
            for v in itertools.product(*expanded_dictionary.values())]

    def print_combinations(self) -> None:
        """
        Prints hyperparameter combinations.
        """

        pprint.pprint(self.combinations)