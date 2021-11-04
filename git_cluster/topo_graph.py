import numpy as np
from scipy.optimize import linprog


def wasserstein_distance(p, q):
    """Compute Wasserstein distance by linear programming
        p.shape=[m], q.shape=[n], D.shape=[m, n]
        p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    p = p / p.sum()
    q = q / q.sum()
    D = np.sqrt(np.square(p[:, None] - q[None, :]))

    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun


class TopoGraph:
    def __init__(
        self,
        target_ratio
    ):
        self.target_ratio = target_ratio

    def graph_construction(self, V, X_extend, Boundary, Dis):
        E = {}
        X_dis = []
        for (i, j, ri, rj) in Boundary:
            E[(ri, rj)] = 0
            E[(rj, ri)] = 0
            X_dis.append(np.linalg.norm(X_extend[i, :-4]-X_extend[j, :-4]))

        if len(X_dis) == 0:
            RT = list(V.keys())
            return {(RT[0], RT[0]): 0}

        for idx, (i, j, ri, rj) in enumerate(Boundary):
            s = np.exp(-X_dis[idx])/(len(V[ri])*len(V[rj]))
            E[(ri, rj)] += s
            E[(rj, ri)] = E[(ri, rj)]
        return E

    def graph_pruning(self, E_raw, V, target, DataNumber):
        assert E_raw is not None
        edges = [(i, j, v) for (i, j), v in E_raw.items()]
        edges = sorted(edges, key=lambda x: x[2], reverse=True)
        C = [{i: i for i in V.keys()}]    # node_idx: cls_idx
        M = [{i: [i] for i in V.keys()}]  # cls_idx: [node_idx]
        # number of points in each cluster
        N = [{i: len(V[i]) for i in V.keys()}]
        score = [np.inf]
        c = len(target)
        target = np.array(target)/np.array(target).sum()*DataNumber
        useless_edges = []
        E_final = {}

        for i, j, v in edges:
            C_prev, M_prev, N_prev = C[-1], M[-1], N[-1]
            if C_prev[i] != C_prev[j]:
                S_i, S_j = M_prev[i], M_prev[j]
                S = S_i + S_j

                C_now = C_prev.copy()
                for k in S_j:
                    C_now[k] = C_prev[i]

                M_now = M_prev.copy()
                for k in S:
                    M_now[k] = S

                N_now = N_prev.copy()
                for k in S:
                    N_now[k] = N_prev[i] + N_prev[j]

                seq = np.array(
                    sorted([N_now[c] for c in set(C_now.values())], reverse=True))
                if len(seq) < c:
                    break
                sc = wasserstein_distance(seq, target)
                if sc <= score[-1]:
                    score.append(sc)
                    C.append(C_now)
                    M.append(M_now)
                    N.append(N_now)
                    E_final[(i, j)] = v
                else:
                    useless_edges.append((i, j, v))

        cls_set = list(set(C[-1].values()))
        # ordered number-->index of local clusters in the same classes, e.g., 1:[2658, 2691, 2579, 2556, 2258, 2645]
        classes = {idx: M[-1][root_idx]
                   for idx, root_idx in enumerate(cls_set)}

        point_numbers = {idx: N[-1][root_idx]
                         for idx, root_idx in enumerate(cls_set)}
        root2order = {root_idx: idx for idx, root_idx in enumerate(cls_set)}

        A = np.zeros((len(cls_set), len(cls_set)))
        for i, j, v in useless_edges:
            row, col = root2order[C[-1][i]], root2order[C[-1][j]]
            A[row, col] += v
            A[col, row] += v

        N_pred = np.array(list(point_numbers.values()))
        merge_idx = np.argsort(N_pred)
        target_set = merge_idx[-c:].tolist()
        source_set = merge_idx[:-c].tolist()
        A[:, target_set] = 0
        target_idx, source_idx = A.nonzero()
        edges = [[i, j, A[i, j]] for i in target_idx for j in source_idx]
        edges = sorted(edges, key=lambda x: x[2], reverse=True)

        re_direct = {i: i for i in range(A.shape[0])}
        for target, source, value in edges:
            if len(source_set) == 0:
                break

            if source in source_set:
                if re_direct[target] != source:
                    classes[re_direct[target]] += classes[source]
                    source_set.remove(source)
                    del classes[source]

                    for key, val in re_direct.items():
                        if val == source:
                            re_direct[key] = re_direct[target]

        return E_final, classes

    def topograph_construction_pruning(self, V, X_extend, Boundary, Dis):
        E_raw = self.graph_construction(V, X_extend, Boundary, Dis)
        E_final, node2cls = self.graph_pruning(
            E_raw, V, self.target_ratio, X_extend.shape[0])
        return E_raw, E_final, node2cls
