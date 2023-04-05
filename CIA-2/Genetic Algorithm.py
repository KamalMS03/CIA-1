#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np


# In[72]:


data = pd.read_csv(r"C:\Users\kamal\Downloads\Bank_Personal_Loan_Modelling.csv")

x = data.drop(["Personal Loan","ID","ZIP Code"],axis = 1)
y = data["Personal Loan"]
x = torch.tensor(x.values, dtype=torch.float64)
y = torch.tensor(y, dtype=torch.float64)


# In[73]:


N_input =  11
learning = 0.001
batch_size = 10
n_epochs = 10


# In[74]:


model = nn.Sequential(
    nn.Linear(N_input,16),
    nn.ReLU(),
    nn.Linear(16,2),
    nn.Sigmoid()
    )


# In[75]:


population_size = 100
num_generations = 100
mutation_rate = 0.1
crossover_rate = 0.8
num_parents = 2
criterion = nn.CrossEntropyLoss()


# In[76]:


def fitness_func(weights, inputs, targets):
    start_idx = 0
    for param in model.parameters():
        end_idx = start_idx + np.prod(param.shape)
        param.data = torch.from_numpy(weights[start_idx:end_idx].reshape(param.shape)).float()
        start_idx = end_idx
    outputs = model(inputs)
    loss = criterion(outputs, target)
    print(loss.item())
    return -loss.item()


# In[77]:


inputs = x
target = y
weights = np.concatenate([parameter.detach().numpy().flatten() for parameter in model.parameters()])
population = [np.random.choice([0, 1], size=weights.shape) for _ in range(population_size)]


# In[78]:


for generation in range(num_generations):
    fitness_values = np.array([fitness_func(weights * individual, inputs, targets) for individual in population])
    fitness_values = fitness_values.to(torch.float64)
    parents = []
    for i in range(num_parents):
        idx = np.argmax(fitness_values)
        parents.append(population[idx])
        fitness_values[idx] = -np.inf
    offspring = []
    while len(offspring) < population_size - num_parents:
        if np.random.rand() < crossover_rate:
            idx1, idx2 = np.random.choice(range(num_parents), size=2, replace=False)
            child = np.zeros(weights.shape)
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    if np.random.rand() < 0.5:
                        child[i][j] = parents[idx1][i][j]
                    else:
                        child[i][j] = parents[idx2][i][j]
        else:
            child = parents[np.random.choice(range(num_parents))]
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if np.random.rand() < mutation_rate:
                    child[i][j] = 1 - child[i][j]
        offspring.append(child)
    population = parents + offspring

