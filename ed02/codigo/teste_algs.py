import csv
import heapq
import time
import os
import tracemalloc
import matplotlib.pyplot as plt
from collections import deque

class MemoryTracker:
    def __init__(self):
        self.max_memory = 0
    
    def start(self):
        tracemalloc.start()
    
    def measure(self):
        _, peak = tracemalloc.get_traced_memory()
        peak_mb = peak / (1024 * 1024)
        if peak_mb > self.max_memory:
            self.max_memory = peak_mb
        return peak_mb
    
    def stop(self):
        tracemalloc.stop()
        return self.max_memory

class Grafo:
    def __init__(self, direcionado=False):
        self.adj = {}
        self.direcionado = direcionado
    
    def add_vertice(self, v):
        if v not in self.adj:
            self.adj[v] = {}
    
    def add_aresta(self, u, v, peso=1):
        self.add_vertice(u)
        self.add_vertice(v)
        self.adj[u][v] = peso
        if not self.direcionado:
            self.adj[v][u] = peso

    
    
    @classmethod
    def from_csv(cls, arquivo, direcionado=False):
        grafo = cls(direcionado)
        with open(arquivo, 'r') as f:
            reader = csv.reader(f)
            for linha in reader:
                if len(linha) >= 2:
                    u = linha[0]
                    v = linha[1]
                    peso = float(linha[2]) if len(linha) > 2 else 1
                    grafo.add_aresta(u, v, peso)
        return grafo
    
def gerar_graficos(resultados, pasta_resultados):
    grafos = [res['grafo'] for res in resultados]
    
    # Memória utilizada
    memorias_bfs = [res['bfs']['memoria'] for res in resultados]
    memorias_dfs = [res['dfs']['memoria'] for res in resultados]
    memorias_gulosa = [res['gulosa']['memoria'] for res in resultados]
    memorias_a_estrela = [res['a_estrela']['memoria'] for res in resultados]
    
    plt.figure(figsize=(12, 6))
    plt.bar([i-0.3 for i in range(len(grafos))], memorias_bfs, width=0.2, label='BFS')
    plt.bar([i-0.1 for i in range(len(grafos))], memorias_dfs, width=0.2, label='DFS')
    plt.bar([i+0.1 for i in range(len(grafos))], memorias_gulosa, width=0.2, label='Gulosa')
    plt.bar([i+0.3 for i in range(len(grafos))], memorias_a_estrela, width=0.2, label='A*')
    plt.xticks(range(len(grafos)), grafos, rotation=45)
    plt.xlabel('Grafos')
    plt.ylabel('Memória (MB)')
    plt.title('Uso de Memória por Algoritmo')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_resultados, 'memoria.png'))
    plt.close()
    
    # Tempo de execução
    tempos_bfs = [res['bfs']['tempo'] for res in resultados]
    tempos_dfs = [res['dfs']['tempo'] for res in resultados]
    tempos_gulosa = [res['gulosa']['tempo'] for res in resultados]
    tempos_a_estrela = [res['a_estrela']['tempo'] for res in resultados]
    
    plt.figure(figsize=(12, 6))
    plt.bar([i-0.3 for i in range(len(grafos))], tempos_bfs, width=0.2, label='BFS')
    plt.bar([i-0.1 for i in range(len(grafos))], tempos_dfs, width=0.2, label='DFS')
    plt.bar([i+0.1 for i in range(len(grafos))], tempos_gulosa, width=0.2, label='Gulosa')
    plt.bar([i+0.3 for i in range(len(grafos))], tempos_a_estrela, width=0.2, label='A*')
    plt.xticks(range(len(grafos)), grafos, rotation=45)
    plt.xlabel('Grafos')
    plt.ylabel('Tempo (s)')
    plt.title('Tempo de Execução por Algoritmo')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(pasta_resultados, 'tempo.png'))
    plt.close()

def salvar_resultados_csv(resultados, pasta_resultados):
    with open(os.path.join(pasta_resultados, 'resultados.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Grafo', 'Vértices', 'Arestas',
            'BFS_Caminho', 'BFS_Memória(MB)', 'BFS_Tempo(s)', 'BFS_Comprimento',
            'DFS_Caminho', 'DFS_Memória(MB)', 'DFS_Tempo(s)', 'DFS_Comprimento',
            'Gulosa_Caminho', 'Gulosa_Memória(MB)', 'Gulosa_Tempo(s)', 'Gulosa_Comprimento',
            'A*_Caminho', 'A*_Memória(MB)', 'A*_Tempo(s)', 'A*_Comprimento'
        ])
        
        for res in resultados:
            writer.writerow([
                res['grafo'], res['vertices'], res['arestas'],
                str(res['bfs']['caminho']), res['bfs']['memoria'], res['bfs']['tempo'], res['bfs']['comprimento_caminho'],
                str(res['dfs']['caminho']), res['dfs']['memoria'], res['dfs']['tempo'], res['dfs']['comprimento_caminho'],
                str(res['gulosa']['caminho']), res['gulosa']['memoria'], res['gulosa']['tempo'], res['gulosa']['comprimento_caminho'],
                str(res['a_estrela']['caminho']), res['a_estrela']['memoria'], res['a_estrela']['tempo'], res['a_estrela']['comprimento_caminho']
            ])

def bfs(grafo, inicio, objetivo):
    tracker = MemoryTracker()
    tracker.start()
    
    visitado = set()
    fila = deque([(inicio, [inicio])])
    
    # Medição inicial
    tracker.measure()
    
    while fila:
        vertice, caminho = fila.popleft()
        
        if vertice == objetivo:
            memoria_maxima = tracker.stop()
            return caminho, memoria_maxima
        
        if vertice not in visitado:
            visitado.add(vertice)
            for vizinho in grafo.adj[vertice]:
                if vizinho not in visitado:
                    fila.append((vizinho, caminho + [vizinho]))
        
        # Medição a cada 100 nós visitados
        if len(visitado) % 100 == 0:
            tracker.measure()
    
    memoria_maxima = tracker.stop()
    return None, memoria_maxima

def dfs(grafo, inicio, objetivo, limite_profundidade=100):
    tracker = MemoryTracker()
    tracker.start()
    
    visitado = set()
    pilha = [(inicio, [inicio], 0)]
    
    # Medição inicial
    tracker.measure()
    
    while pilha:
        vertice, caminho, profundidade = pilha.pop()
        
        if vertice == objetivo:
            memoria_maxima = tracker.stop()
            return caminho, memoria_maxima
        
        if vertice not in visitado and profundidade < limite_profundidade:
            visitado.add(vertice)
            for vizinho in grafo.adj[vertice]:
                if vizinho not in visitado:
                    pilha.append((vizinho, caminho + [vizinho], profundidade + 1))
        
        # Medição a cada 100 nós visitados
        if len(visitado) % 100 == 0:
            tracker.measure()
    
    memoria_maxima = tracker.stop()
    return None, memoria_maxima

def busca_gulosa(grafo, inicio, objetivo, heuristica):
    tracker = MemoryTracker()
    tracker.start()
    
    visitado = set()
    heap = []
    heapq.heappush(heap, (heuristica(inicio, objetivo), inicio, [inicio]))
    
    # Medição inicial
    tracker.measure()
    
    while heap:
        _, vertice, caminho = heapq.heappop(heap)
        
        if vertice == objetivo:
            memoria_maxima = tracker.stop()
            return caminho, memoria_maxima
        
        if vertice not in visitado:
            visitado.add(vertice)
            for vizinho in grafo.adj[vertice]:
                if vizinho not in visitado:
                    heapq.heappush(heap, (heuristica(vizinho, objetivo), vizinho, caminho + [vizinho]))
        
        # Medição a cada 100 nós visitados
        if len(visitado) % 100 == 0:
            tracker.measure()
    
    memoria_maxima = tracker.stop()
    return None, memoria_maxima

def a_estrela(grafo, inicio, objetivo, heuristica):
    tracker = MemoryTracker()
    tracker.start()
    
    visitado = set()
    heap = []
    heapq.heappush(heap, (0 + heuristica(inicio, objetivo), 0, inicio, [inicio]))
    
    # Medição inicial
    tracker.measure()
    
    while heap:
        _, custo, vertice, caminho = heapq.heappop(heap)
        
        if vertice == objetivo:
            memoria_maxima = tracker.stop()
            return caminho, memoria_maxima
        
        if vertice not in visitado:
            visitado.add(vertice)
            for vizinho, peso in grafo.adj[vertice].items():
                if vizinho not in visitado:
                    novo_custo = custo + peso
                    heapq.heappush(heap, (novo_custo + heuristica(vizinho, objetivo), novo_custo, vizinho, caminho + [vizinho]))
        
        # Medição a cada 100 nós visitados
        if len(visitado) % 100 == 0:
            tracker.measure()
    
    memoria_maxima = tracker.stop()
    return None, memoria_maxima

def heuristica_simples(u, v):
    return 0

def testar_algoritmos_em_grafos(pasta_grafos, pasta_resultados):
    resultados = []
    
    if not os.path.exists(pasta_resultados):
        os.makedirs(pasta_resultados)
    
    arquivos_grafos = [f for f in os.listdir(pasta_grafos) if f.endswith('.csv')]
    
    for arquivo in arquivos_grafos[:10]:  # Limitar a 10 grafos
        caminho_arquivo = os.path.join(pasta_grafos, arquivo)
        grafo = Grafo.from_csv(caminho_arquivo)
        
        vertices = list(grafo.adj.keys())
        if len(vertices) < 2:
            continue
            
        inicio = vertices[0]
        objetivo = vertices[-1]
        
        print(f"\nTestando grafo: {arquivo}")
        print(f"Vértices: {len(vertices)}, Arestas: {sum(len(vizinhos) for vizinhos in grafo.adj.values())}")
        print(f"Origem: {inicio}, Destino: {objetivo}")
        
        # Testar BFS
        inicio_tempo = time.time()
        caminho_bfs, memoria_bfs = bfs(grafo, inicio, objetivo)
        tempo_bfs = time.time() - inicio_tempo
        
        # Testar DFS
        inicio_tempo = time.time()
        caminho_dfs, memoria_dfs = dfs(grafo, inicio, objetivo)
        tempo_dfs = time.time() - inicio_tempo
        
        # Testar Busca Gulosa
        inicio_tempo = time.time()
        caminho_gulosa, memoria_gulosa = busca_gulosa(grafo, inicio, objetivo, heuristica_simples)
        tempo_gulosa = time.time() - inicio_tempo
        
        # Testar A*
        inicio_tempo = time.time()
        caminho_a_estrela, memoria_a_estrela = a_estrela(grafo, inicio, objetivo, heuristica_simples)
        tempo_a_estrela = time.time() - inicio_tempo
        
        # Armazenar resultados
        resultados.append({
            'grafo': arquivo,
            'vertices': len(vertices),
            'arestas': sum(len(vizinhos) for vizinhos in grafo.adj.values()),
            'bfs': {
                'caminho': caminho_bfs,
                'memoria': memoria_bfs,
                'tempo': tempo_bfs,
                'comprimento_caminho': len(caminho_bfs) if caminho_bfs else None
            },
            'dfs': {
                'caminho': caminho_dfs,
                'memoria': memoria_dfs,
                'tempo': tempo_dfs,
                'comprimento_caminho': len(caminho_dfs) if caminho_dfs else None
            },
            'gulosa': {
                'caminho': caminho_gulosa,
                'memoria': memoria_gulosa,
                'tempo': tempo_gulosa,
                'comprimento_caminho': len(caminho_gulosa) if caminho_gulosa else None
            },
            'a_estrela': {
                'caminho': caminho_a_estrela,
                'memoria': memoria_a_estrela,
                'tempo': tempo_a_estrela,
                'comprimento_caminho': len(caminho_a_estrela) if caminho_a_estrela else None
            }
        })
    
    # Gerar gráficos comparativos (função mantida igual)
    gerar_graficos(resultados, pasta_resultados)
    
    # Salvar resultados em CSV (função mantida igual)
    salvar_resultados_csv(resultados, pasta_resultados)
    
    return resultados

# As funções gerar_graficos() e salvar_resultados_csv() permanecem exatamente como no código original

if __name__ == "__main__":
    pasta_grafos = 'grafos'
    pasta_resultados = 'resultados'
    
    if not os.path.exists(pasta_grafos):
        os.makedirs(pasta_grafos)
        print(f"Diretório '{pasta_grafos}' criado. Por favor, adicione os arquivos CSV com os grafos.")
    else:
        resultados = testar_algoritmos_em_grafos(pasta_grafos, pasta_resultados)
        print("\nAnálise concluída. Resultados salvos na pasta 'resultados'.")