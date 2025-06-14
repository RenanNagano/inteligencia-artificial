import pandas as pd
import numpy as np
import random
import time
import itertools
import os

CAPACIDADES = {
    'knapsack_1.csv': 50,
    'knapsack_2.csv': 60,
    'knapsack_3.csv': 80,
    'knapsack_4.csv': 100,
    'knapsack_5.csv': 120,
    'knapsack_6.csv': 150,
    'knapsack_7.csv': 180,
    'knapsack_8.csv': 200,
    'knapsack_9.csv': 250,
    'knapsack_10.csv': 300,
}

def crossover(parent1, parent2, method):
    if method == "um_ponto":
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:]
    elif method == "dois_pontos":
        p1, p2 = sorted(random.sample(range(len(parent1)), 2))
        return parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    elif method == "uniforme":
        return [random.choice([g1, g2]) for g1, g2 in zip(parent1, parent2)]

def mutacao(individuo, taxa):
    return [1 - g if random.random() < taxa else g for g in individuo]

def fitness(individuo, pesos, valores, capacidade):
    peso_total = sum(g * p for g, p in zip(individuo, pesos))
    valor_total = sum(g * v for g, v in zip(individuo, valores))
    if peso_total > capacidade or np.isnan(valor_total):
        return 0
    return valor_total

def gerar_individuo_valido(pesos, capacidade):
    n = len(pesos)
    tentativa = [0] * n
    indices = list(range(n))
    random.shuffle(indices)
    peso_total = 0
    for i in indices:
        if peso_total + pesos[i] <= capacidade:
            tentativa[i] = 1
            peso_total += pesos[i]
    return tentativa

def populacao_inicial(tamanho, n_genes, tipo, pesos, valores, capacidade):
    pop = []
    for _ in range(tamanho):
        if tipo == "aleatoria":
            ind = gerar_individuo_valido(pesos, capacidade)
        else:
            # Ordenar por valor/peso, tratando divisões por zero
            indices = sorted(
                range(n_genes),
                key=lambda i: valores[i] / pesos[i] if pesos[i] > 0 else 0,
                reverse=True
            )
            ind = [0] * n_genes
            total_peso = 0
            for i in indices:
                if total_peso + pesos[i] <= capacidade:
                    ind[i] = 1
                    total_peso += pesos[i]
        pop.append(ind)
    return pop


def torneio(pop, scores, k=3):
    selecionados = random.sample(list(zip(pop, scores)), k)
    selecionados.sort(key=lambda x: x[1], reverse=True)
    return selecionados[0][0]

def algoritmo_genetico(pesos, valores, capacidade, crossover_tipo, mutacao_taxa, init_tipo, parada_tipo, max_geracoes=200, sem_melhora_limite=20):
    n = len(pesos)
    pop_size = 50
    pop = populacao_inicial(pop_size, n, init_tipo, pesos, valores, capacidade)
    scores = [fitness(ind, pesos, valores, capacidade) for ind in pop]

    melhor_valor = max(scores)
    melhor_individuo = pop[scores.index(melhor_valor)]
    sem_melhora = 0

    for geracao in range(max_geracoes):
        nova_pop = []
        for _ in range(pop_size):
            p1 = torneio(pop, scores)
            p2 = torneio(pop, scores)
            filho = crossover(p1, p2, crossover_tipo)
            filho = mutacao(filho, mutacao_taxa)
            nova_pop.append(filho)

        pop = nova_pop
        scores = [fitness(ind, pesos, valores, capacidade) for ind in pop]

        max_score = max(scores)
        if max_score > melhor_valor:
            melhor_valor = max_score
            melhor_individuo = pop[scores.index(max_score)]
            sem_melhora = 0
        else:
            sem_melhora += 1

        if parada_tipo == "convergencia" and sem_melhora >= sem_melhora_limite:
            break

    peso_total = sum(g * p for g, p in zip(melhor_individuo, pesos))
    return melhor_valor, peso_total, geracao + 1

def executar_experimentos(pasta):
    resultados = []

    crossovers = ["um_ponto", "dois_pontos", "uniforme"]
    mutacoes = [0.01, 0.05, 0.1]
    inits = ["aleatoria", "heuristica"]
    paradas = ["fixo", "convergencia"]

    combinacoes = list(itertools.product(crossovers, mutacoes, inits, paradas))

    arquivos = [f for f in os.listdir(pasta) if f.startswith("knapsack") and f.endswith(".csv")]

    for nome_arquivo in arquivos:
        df = pd.read_csv(os.path.join(pasta, nome_arquivo))
        pesos = df['Peso'].tolist()
        valores = df['Valor'].tolist()
        capacidade = CAPACIDADES[nome_arquivo]

        for (cx, mt, init, parada) in combinacoes:
            for repeticao in range(5):
                
                inicio = time.time()
                resultado = algoritmo_genetico(pesos, valores, capacidade, cx, mt, init, parada, 200)
                fim = time.time()

                resultados.append({
                    "Instancia": nome_arquivo,
                    "Crossover": cx,
                    "Mutacao": mt,
                    "Inicializacao": init,
                    "Parada": parada,
                    "Execucao": repeticao + 1,
                    "Valor Total": resultado[0],
                    "Peso Total": resultado[1],
                    "Gerações": resultado[2],
                    "Tempo": round(fim - inicio, 4)
                })
                print(f"{nome_arquivo} - {cx}, {mt}, {init}, {parada}, Rep {repeticao+1} => Valor: {resultado[0]}")

    df_resultado = pd.DataFrame(resultados)
    df_resultado.to_csv("resultados_genetico_knapsack.csv", index=False)
    print("Experimentos finalizados! Resultados salvos em 'resultados_genetico_knapsack.csv'")

if __name__ == "__main__":
    executar_experimentos(".")
