from dieline_classifier import hyperparameters

d1 = {
    "a": 1,
    "b": 2,
    "c": 3,
}

d2 = {
    "d": [4],
    "e": [5],
    "f": [6],
}

d3 = {
    "g": [7, 8, 9],
    "h": ["h1", "h2", "h3"],
}

d4 = {
    "A": [10],
    "B": ["B1"],
    "C": [d1, d3]
}

def test_hyperparameters():
    h1 = hyperparameters.Hyperparameters(d1)
    assert(h1.combinations == [d1])

    h2 = hyperparameters.Hyperparameters(d2)
    assert(h2.combinations == [{"d": 4, "e": 5, "f": 6}])

    h3 = hyperparameters.Hyperparameters(d3)
    assert(h3.combinations == [{"g": 7, "h": "h1"}, {"g": 7, "h": "h2"}, {"g": 7, "h": "h3"}, {"g": 8, "h": "h1"}, {"g": 8, "h": "h2"}, {"g": 8, "h": "h3"}, {"g": 9, "h": "h1"}, {"g": 9, "h": "h2"}, {"g": 9, "h": "h3"}])

    h4 = hyperparameters.Hyperparameters(d4)
    assert(h4.combinations == [{"A": 10, "B": "B1", "C": {"a": 1, "b": 2, "c": 3}}, {"A": 10, "B": "B1", "C": {"g": 7, "h": "h1"}}, {"A": 10, "B": "B1", "C": {"g": 7, "h": "h2"}}, {"A": 10, "B": "B1", "C": {"g": 7, "h": "h3"}}, {"A": 10, "B": "B1", "C": {"g": 8, "h": "h1"}}, {"A": 10, "B": "B1", "C": {"g": 8, "h": "h2"}}, {"A": 10, "B": "B1", "C": {"g": 8, "h": "h3"}}, {"A": 10, "B": "B1", "C": {"g": 9, "h": "h1"}}, {"A": 10, "B": "B1", "C": {"g": 9, "h": "h2"}}, {"A": 10, "B": "B1", "C": {"g": 9, "h": "h3"}}])
