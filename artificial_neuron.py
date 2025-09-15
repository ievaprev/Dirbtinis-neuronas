
import numpy as np
from numpy.random import uniform, rand, seed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

"""
Dirbtinio neurono klase ir slenkstinė funkcija
    ~ input_size - įėjimų skaičius 
    ~ activation_function - vartotojo pasirinkta aktivacijos funkcija
"""

class Neuron(nn.Module):
    def __init__(self, input_size, activation_function):
        super(Neuron, self).__init__()
        self.neuron = nn.Linear(input_size, 1) # 1 reprezentuoja išėjimų skaičių
        self.activation_function = activation_function

    def forward(self, x):
        return self.activation_function(self.neuron(x))

def step_function(x): #Slenkstinė funkcija
    return (x >= 0).float()

"""
Duomenų generavimas:
  ~ Sugeneruojamos dvi grupės c1, c2
  ~ Dvimačiai taškai turi būt tiesiškai atskiriami 
"""
seed(42) # Naudojama, kad kiekvieną kartą būtų generuojami tokie patys duomenys

c1 = rand(10, 2) + np.array([-3, 0])
c2 = rand(10,2) + np.array([3, 0])

x = np.vstack((c1, c2))
targets = [0]* 10 + [1]*10

"""
Vartotojas turi teise pasirinkti kurią aktivavimo funkciją naudoti
    ~ 0 - Slenkstinė aktivavimo funkcija
    ~ 1 - Sigmoidinė aktivavimo funkcija
"""

choice = int(input("Pasirinkite kurią aktivavimo funkcija norėsite naudoti: (0 - Slenkstinė, 1 - Sigmoidinė)\n"))
if choice != 0 and choice != 1:
    raise ValueError("Galimi pasirinkimai yra 0 - Slenkstinė ir 1 - Sigmoidinė aktivavimo funkcijos")

activation_function = step_function if choice == 0 else torch.sigmoid

"""
    ~ Implementuojamas dirbtinis neuronas, naudojant kodo pradžioje sukurtą neurono klasę
    ~ Nustatomos trys svorių (w1, w1) ir poslinkio (b) reikšmės, kurios teisingai klasifikuoja klases
"""

solutions = []
attempts = 1000

for __ in range(attempts):
    w = uniform(-5, 5, 2) # sugeneruoja du svorius 
    b = uniform(-5, 5)

    a = np.dot(x, w) + b
    y = activation_function(a)

    if np.array_equal(y, targets):
        solutions.append((w, b))
        if len(solutions) == 3:
            break

print(solutions)

# sukuriamas neuronas su dviem įėjimais ir specifikuojama aktivacijos funkcija
neuron = Neuron(2, activation_function=activation_function) 

w, b = solutions[0] # Naudojamas pirmas rinkinys, kad patikrinti neurono veikimą
neuron.neuron.weight.data = torch.from_numpy(w.astype(np.float32)).unsqueeze(0)
neuron.neuron.bias.data = torch.tensor([b], dtype= torch.float32)

input = torch.tensor(x, dtype = torch.float32) 
output = neuron(input)

if choice == 1:
    output = (output >= 0.5).float()

print(output)

"""
Grafinis užduoties vaizdavimas:
    ~ Klasių pavaizdavimas skirtingomis spalvomis 
        ~ Pirma klasė pavaizduota raudona spalva
        ~ Antra klasė pavaizduota mėlyna spalva 
    ~ Klases skiriančios tiesės 
"""

colors = ['orange', 'green', 'brown']

plt.scatter(c1[:,0], c1[:,1], color = "red")
plt.scatter(c2[:,0], c2[:,1], color="blue")

x_vals = np.linspace(x[:,0].min(), x[:,0].max(), 100)

for i, (w, b) in enumerate(solutions):
    
    y_vals = (-w[0] * x_vals - b)/w[1]
    plt.plot(x_vals, y_vals, color=colors[i], label = (f'{i+1} rinkinio tiesė'))

    # sukurti rodyklei buvo pasitelkta dirbtinio intelekto pagalba
    x0 = 0
    y0 = -b / w[1] 
    # np.linalg.norm(w) - euklidini atstuma apsakiciuoti, o w / np.linarlg.norm(w) normalizuoja vektoriu
    w_norm = w / np.linalg.norm(w) * 1.5 
    plt.arrow(x0, y0, w_norm[0], w_norm[1],
          head_width=0.1, head_length=0.1,
          fc=colors[i], ec=colors[i])

    
plt.grid(True, alpha = 0.3)  
plt.gca().set_aspect('equal', adjustable='box') # buvo pasitelkta dirbtinio intelekto komanda
plt.ylim(-8, 8)
plt.xlim(-8, 8)
plt.legend(loc = "lower right")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()