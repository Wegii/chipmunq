import numpy as np
import json
import random


def square(m = 5, n = 5):
    if not m:
        m = 5
        n = 5
    
    Edges = []
    
    for i in range(m):
        for j in range(n - 1):
            Edges.append((i * n + j, i * n + j + 1))
    
    for i in range(m - 1):
        for j in range(n):
                Edges.append((i * n + j, (i + 1) * n + j))
    
    graph = np.zeros((m * n, m * n), dtype=np.int64)
    for (i, j) in Edges:
        graph[i][j] = 1
        graph[j][i] = 1
    
    index2coor = []
    for i in range(m):
        for j in range(n):
            index2coor.append((i, j))
            
    return graph, index2coor

def hexagon(m = 3, n = 7):
    if not m:
        m = 3
        n = 7
         
    Edges = []
    
    for i in range(m):
        for j in range(n - 1):
            Edges.append((i * n + j, i * n + j + 1))
    
    for i in range(m - 1):
        for j in range(n):
            if (i + j) % 2 == 1:
                Edges.append((i * n + j, (i + 1) * n + j))
    
    graph = np.zeros((m * n, m * n), dtype=np.int64)
    for (i, j) in Edges:
        graph[i][j] = 1
        graph[j][i] = 1
    
    index2coor = []
    for i in range(m):
        for j in range(n):
            index2coor.append((i, j))
            
    return graph, index2coor



def octagon(m = 3, n = 9):
    if not m:
        m = 3
        n = 9
    
    Edges = []
    
    for i in range(m):
        for j in range(n - 1):
                Edges.append((i * n + j, i * n + j + 1))
    
    for i in range(m - 1):
        for j in range(n):
            if (i*2 + j) % 4 == 0 or (i*2 + j) % 4 == 1 :
                Edges.append((i * n + j, (i + 1) * n + j))
    
    graph = graph = np.zeros((m * n, m * n), dtype=np.int64)
    for (i, j) in Edges:
        graph[i][j] = 1
        graph[j][i] = 1
        
    index2coor = []
    for i in range(m):
        for j in range(n):
            index2coor.append((i, j))    
    
    return graph, index2coor

def square_minus():
    Edges = []
    
    for i in range(3):
        for j in range(4):
            Edges.append((i * 5 + j, i * 5 + j + 1))
    
    for i in range(2):
        for j in range(5):
                Edges.append((i * 5 + j, (i + 1) * 5 + j))
    Edges.extend([(15, 17), (16, 17), (18, 17), (16, 1), (17, 2), (18, 3)])
    graph = graph = np.zeros((19, 19), dtype=np.int64)
    for (i, j) in Edges:
        graph[i][j] = 1
        graph[j][i] = 1
    return graph


def tokyo(m, n):
    if not m:
        m = 5
        n = 5
    
    def coor2ind(i, j):
        return i * n + j
    Edges = []
    
    for i in range(m):
        for j in range(n - 1):
            Edges.append((coor2ind(i, j), coor2ind(i, j+1)))
    
    for i in range(m - 1):
        for j in range(n):
            Edges.append((coor2ind(i, j), coor2ind(i+1, j)))
            
    for i in range(m - 1):
        for j in range(n - 1):
            if (i + j) % 2 == 0:
                Edges.append((coor2ind(i, j), coor2ind(i+1, j+1)))
                Edges.append((coor2ind(i+1, j), coor2ind(i, j+1))) 
                    
    graph = graph = np.zeros((m * n, m * n), dtype=np.int64)
    for (i, j) in Edges:
        graph[i][j] = 1
        graph[j][i] = 1
    return graph, []


def tokyo_plus(m, n):
    if not m:
        m = 5
        n = 5
    
    def coor2ind(i, j):
        return i * n + j
    Edges = []
    
    for i in range(m):
        for j in range(n - 1):
            Edges.append((coor2ind(i, j), coor2ind(i, j+1)))
    
    for i in range(m - 1):
        for j in range(n):
            Edges.append((coor2ind(i, j), coor2ind(i+1, j)))
            
    for i in range(m - 1):
        for j in range(n - 1):
            Edges.append((coor2ind(i, j), coor2ind(i+1, j+1)))
            Edges.append((coor2ind(i+1, j), coor2ind(i, j+1))) 
                    
    graph = graph = np.zeros((m * n, m * n), dtype=np.int64)
    for (i, j) in Edges:
        graph[i][j] = 1
        graph[j][i] = 1
    return graph


def heavy_hexagon(m = 3, n = 7):
    if not m:
        m = 3
        n = 7
         
    Edges = []
    
    for i in range(m):
        for j in range(n - 1):
            Edges.append((i * n + j, i * n + j + 1))
    
    for i in range(m - 1):
        for j in range(n):
            if (i + j) % 2 == 1:
                Edges.append((i * n + j, (i + 1) * n + j))
                
    H_Edges = []
    k = 0
    for (i, j) in Edges:
        H_Edges.append((i, m * n + k))
        H_Edges.append((j, m * n + k))
        k += 1
    
    graph = np.zeros((m * n + k, m * n + k), dtype=np.int64)
    for (i, j) in H_Edges:
        graph[i][j] = 1
        graph[j][i] = 1
        
    index2coor = []
    for i in range(m):
        for j in range(n):
            index2coor.append((2 * i + 1, 2 * j + 1))
    for (i, j) in Edges:
        x = (index2coor[i][0] + index2coor[j][0]) // 2
        y = (index2coor[i][1] + index2coor[j][1]) // 2
        index2coor.append((x, y))
    return graph, index2coor

def heavy_square(m = 5, n = 5):
    Edges = []
    
    for i in range(m):
        for j in range(n - 1):
            Edges.append((i * n + j, i * n + j + 1))
    
    for i in range(m - 1):
        for j in range(n):
                Edges.append((i * n + j, (i + 1) * n + j))
    
    H_Edges = []
    k = 0
    for (i, j) in Edges:
        H_Edges.append((i, m * n + k))
        H_Edges.append((j, m * n + k))
        k += 1
    
    graph = np.zeros((m * n + k, m * n + k), dtype=np.int64)
    for (i, j) in H_Edges:
        graph[i][j] = 1
        graph[j][i] = 1
        
    index2coor = []
    for i in range(m):
        for j in range(n):
            index2coor.append((2 * i + 1, 2 * j + 1))
    for (i, j) in Edges:
        x = (index2coor[i][0] + index2coor[j][0]) // 2
        y = (index2coor[i][1] + index2coor[j][1]) // 2
        index2coor.append((x, y))
    return graph, index2coor


graph, index2coor = hexagon(m = 7, n = 25)
print(graph.shape)

graph, index2coor = heavy_hexagon(m = 9, n = 18)
print(graph.shape)