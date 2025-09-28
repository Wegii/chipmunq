import argparse
import json
import numpy as np

def square(m, n):
    def coord2idx(i, j):
        return i * n + j

    idx2coord = []
    for i in range(m):
        for j in range(n):
            idx2coord.append((i, j))

    Edges = []
    for i in range(m):
        for j in range(n - 1):
            Edges.append((coord2idx(i, j), coord2idx(i, j+1)))

    for i in range(m - 1):
        for j in range(n):
                Edges.append((coord2idx(i, j), coord2idx(i+1, j)))
    
    graph = np.zeros((m * n, m * n), dtype=np.int64)
    for (i, j) in Edges:
        graph[i][j] = 1
        graph[j][i] = 1
    
    return graph, idx2coord


def hexagon(m, n):
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
    
    idx2coord = []
    for i in range(m):
        for j in range(n):
            idx2coord.append((i, j))
            
    return graph, idx2coord


def h_square(m, n):
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
        
    idx2coord = []
    for i in range(m):
        for j in range(n):
            idx2coord.append((2 * i + 1, 2 * j + 1))
    for (i, j) in Edges:
        x = (idx2coord[i][0] + idx2coord[j][0]) // 2
        y = (idx2coord[i][1] + idx2coord[j][1]) // 2
        idx2coord.append((x, y))
    return graph, idx2coord


def h_hexagon(m, n):
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
        
    idx2coord = []
    for i in range(m):
        for j in range(n):
            idx2coord.append((2 * i + 1, 2 * j + 1))
    for (i, j) in Edges:
        x = (idx2coord[i][0] + idx2coord[j][0]) // 2
        y = (idx2coord[i][1] + idx2coord[j][1]) // 2
        idx2coord.append((x, y))
    return graph, idx2coord


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='path to architecture file')
    parser.add_argument('-a', '--arch', help='name of architecture')
    parser.add_argument('-m', '--row', type=int)
    parser.add_argument('-n', '--col', type=int)
    parser.add_argument('-d', '--defect', choices=["on", "off"], default="off")
    args = parser.parse_args()

    if args.arch == 'square':
       CG, idx2coord = square(args.row, args.col)
    elif args.arch == 'hexagon':
       CG, idx2coord = hexagon(args.row, args.col)
    elif args.arch == 'h_square':
       CG, idx2coord = h_square(args.row, args.col)
    elif args.arch == 'h_hexagon':
       CG, idx2coord = h_hexagon(args.row, args.col)
    else:
        raise Exception('Error')

    if args.defect == 'on':
        i, j = idx2coord[args.row * args.col - 1]
        i = i // 2
        j = j // 2
        if (i, j) not in idx2coord:
            i += 1
        idx = idx2coord.index((i, j))
        
        keep_idx = np.ones(len(idx2coord), dtype=bool)
        keep_idx[idx] = False

        CG = CG[keep_idx, :][:, keep_idx]
        idx2coord.pop(idx)

    print('# Physical Qubit: %d' % len(CG))

    with open(args.fname, 'w') as f:
        json.dump({'CG': CG.tolist(), 'idx2coord':idx2coord}, f)