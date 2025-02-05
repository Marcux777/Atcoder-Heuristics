import sys
import math
import random
import time
import copy

def gerar_candidatos(i, j, N, board):
    """
    Gera uma lista de candidatos (movimentos) para a célula (i, j).
    Cada candidato é uma tupla: (direcao, repetições, custo, sentido)

    direcao ∈ {U, D, L, R}
    repetições = quantas vezes mover
    custo = repetições * 2
    sentido = 'up', 'down', 'left', 'right' (apenas texto auxiliar)
    """
    candidatos = []

    # Candidato para cima
    if all(board[k][j] != 'o' for k in range(i)):
        rep = i + 1
        custo = rep * 2
        candidatos.append(("U", rep, custo, "up"))

    # Candidato para baixo
    if all(board[k][j] != 'o' for k in range(i+1, N)):
        rep = (N - i)
        custo = rep * 2
        candidatos.append(("D", rep, custo, "down"))

    # Candidato para a esquerda
    if all(board[i][k] != 'o' for k in range(j)):
        rep = j + 1
        custo = rep * 2
        candidatos.append(("L", rep, custo, "left"))

    # Candidato para a direita
    if all(board[i][k] != 'o' for k in range(j+1, N)):
        rep = (N - j)
        custo = rep * 2
        candidatos.append(("R", rep, custo, "right"))

    return candidatos

def marcar_removidos(i, j, direcao, sentido, board, removed, N):
    """
    Marca como removidos os Oni ('x') na linha ou coluna dependendo da direção.
    O par de movimentos (ida/volta) garante que somente os Oni naquela 'faixa' saem.
    """
    if direcao in ("U", "D"):
        # Para movimentos verticais => coluna j
        if sentido == "up":
            for k in range(i + 1):
                if board[k][j] == 'x':
                    removed[k][j] = True
        else:  # down
            for k in range(i, N):
                if board[k][j] == 'x':
                    removed[k][j] = True
    else:
        # Movimentos horizontais => linha i
        if sentido == "left":
            for k in range(j + 1):
                if board[i][k] == 'x':
                    removed[i][k] = True
        else:  # right
            for k in range(j, N):
                if board[i][k] == 'x':
                    removed[i][k] = True

def construir_solucao_inicial(board, N):
    """
    Constrói a solução inicial ingênua:
    - Para cada Oni, gera o par de deslocamentos minimal e adiciona.
    """
    removed = [[False]*N for _ in range(N)]
    reverse_dir = {"U": "D", "D": "U", "L": "R", "R": "L"}

    # Lista das posições de Oni
    posicoes_onis = []
    for i in range(N):
        for j in range(N):
            if board[i][j] == 'x':
                posicoes_onis.append((i, j))

    moves = []

    for (i, j) in posicoes_onis:
        if removed[i][j]:
            continue
        candidatos = gerar_candidatos(i, j, N, board)
        if not candidatos:
            continue
        # Pega movimento de menor custo
        direcao, rep, _, sentido = min(candidatos, key=lambda x: x[2])

        if direcao in ("U", "D"):
            moves.extend([f"{direcao} {j}"] * rep)
            moves.extend([f"{reverse_dir[direcao]} {j}"] * rep)
        else:
            moves.extend([f"{direcao} {i}"] * rep)
            moves.extend([f"{reverse_dir[direcao]} {i}"] * rep)

        marcar_removidos(i, j, direcao, sentido, board, removed, N)

    return moves


# ------------------------------------------------------------
# Avaliação e simulação
# ------------------------------------------------------------

def simular_movimentos(board_original, moves):
    """
    Aplica todos os 'moves' em uma cópia do board_original,
    retornando:
    - onis_restantes
    - fukus_removidos
    - T (tamanho de moves)
    """
    import copy
    N = len(board_original)
    board = copy.deepcopy(board_original)

    for move in moves:
        direcao, idx_str = move.split()
        idx = int(idx_str)
        if direcao == 'L':
            linha = board[idx]
            removida = linha[0]
            for c in range(0, N-1):
                linha[c] = linha[c+1]
            linha[N-1] = '.'
        elif direcao == 'R':

            linha = board[idx]
            removida = linha[N-1]
            for c in range(N-1, 0, -1):
                linha[c] = linha[c-1]
            linha[0] = '.'
        elif direcao == 'U':
            removida = board[0][idx]
            for r in range(0, N-1):
                board[r][idx] = board[r+1][idx]
            board[N-1][idx] = '.'
        else:  # 'D'

            removida = board[N-1][idx]
            for r in range(N-1, 0, -1):
                board[r][idx] = board[r-1][idx]
            board[0][idx] = '.'


    # Contabiliza Onis remanescentes e Fukus removidos
    onis_restantes = 0
    fukus_iniciais = 0
    fukus_finais = 0
    for row in board_original:
        for c in row:
            if c == 'o':
                fukus_iniciais += 1

    for row in board:
        for c in row:
            if c == 'x':
                onis_restantes += 1
            if c == 'o':
                fukus_finais += 1

    fukus_removidos = fukus_iniciais - fukus_finais
    T = len(moves)
    return onis_restantes, fukus_removidos, T


def compute_score(onis_restantes, fukus_removidos, T, N):
    """
    Retorna a pontuação final segundo as regras:
      - Se X=Y=0 => 8N^2 - T
      - Se X>0 ou Y>0 => 4N^2 - N*(X+Y)
    """
    if onis_restantes == 0 and fukus_removidos == 0:
        return 8*N*N - T
    else:
        return 4*N*N - N*(onis_restantes + fukus_removidos)


def evaluate_solution(board_original, moves):
    """
    Faz a simulação e retorna a pontuação calculada.
    """
    N = len(board_original)
    X, Y, T = simular_movimentos(board_original, moves)
    return compute_score(X, Y, T, N)


# ------------------------------------------------------------
# Simulated Annealing
# ------------------------------------------------------------

def evaluate_solution_with_y(board_original, moves):
    """
    Retorna uma tupla (score, fukus_removidos) para a solução 'moves'
    """
    N = len(board_original)
    X, Y, T = simular_movimentos(board_original, moves)
    score = compute_score(X, Y, T, N)
    return score, Y

def perturb(moves, board_original):
    """
    Nova função de perturbação que inclui:
      - swap ou delete (já existentes)
      - inserir nova operação aleatória
      - fundir operações adjacentes idênticas
      - mover um bloco contíguo de operações para outra posição
      - remover operações que não removem nenhum Oni (inúteis)
    """
    new_moves = copy.deepcopy(moves)
    ops = ["swap", "delete", "insert", "merge", "block_move", "remove_useless"]
    op = random.choice(ops)

    if op == "swap" and len(new_moves) >= 2:
        i = random.randint(0, len(new_moves)-1)
        j = random.randint(0, len(new_moves)-1)
        new_moves[i], new_moves[j] = new_moves[j], new_moves[i]

    elif op == "delete" and new_moves:
        idx = random.randint(0, len(new_moves)-1)
        new_moves.pop(idx)

    elif op == "insert":
        # Insere uma nova operação aleatória que não exista na lista.
        # Escolhe direção aleatória e idx de acordo com dimensão do board.
        # Supondo que board_original é quadrado: N x N.
        N = len(board_original)
        direcao = random.choice(["L", "R", "U", "D"])
        if direcao in ("L", "R"):
            idx = random.randint(0, N-1)
        else:
            idx = random.randint(0, N-1)
        new_op = f"{direcao} {idx}"
        pos = random.randint(0, len(new_moves))
        # Insere somente se a operação não estiver duplicada em posição próxima.
        if pos - 1 >= 0 and new_moves[pos - 1] == new_op:
            pass
        else:
            new_moves.insert(pos, new_op)

    elif op == "merge":
        # Fundir operações adjacentes iguais na mesma linha/coluna.
        i = 0
        while i < len(new_moves)-1:
            op1 = new_moves[i].split()
            op2 = new_moves[i+1].split()
            if op1[0] == op2[0] and op1[1] == op2[1]:
                # Remove uma das operações redundantes.
                new_moves.pop(i+1)
            else:
                i += 1

    elif op == "block_move" and len(new_moves) >= 2:
        # Seleciona um bloco contíguo de operações e o move para outra posição.
        start = random.randint(0, len(new_moves)-2)
        end = random.randint(start+1, len(new_moves)-1)
        block = new_moves[start:end+1]
        del new_moves[start:end+1]
        pos = random.randint(0, len(new_moves))
        new_moves[pos:pos] = block

    elif op == "remove_useless":
        # Remove operações que não alteram o board (inúteis).
        # Simula até cada operação e descarta se o board não muda.
        board_temp = copy.deepcopy(board_original)
        useful_moves = []
        N = len(board_original)
        for move in new_moves:
            # Cópia do estado atual.
            board_before = copy.deepcopy(board_temp)
            direcao, idx_str = move.split()
            idx = int(idx_str)
            if direcao == 'L':
                linha = board_temp[idx]
                for c in range(0, N-1):
                    linha[c] = linha[c+1]
                linha[N-1] = '.'
            elif direcao == 'R':
                linha = board_temp[idx]
                for c in range(N-1, 0, -1):
                    linha[c] = linha[c-1]
                linha[0] = '.'
            elif direcao == 'U':
                for r in range(0, N-1):
                    board_temp[r][idx] = board_temp[r+1][idx]
                board_temp[N-1][idx] = '.'
            else:  # 'D'
                for r in range(N-1, 0, -1):
                    board_temp[r][idx] = board_temp[r-1][idx]
                board_temp[0][idx] = '.'
            if board_temp != board_before:
                useful_moves.append(move)
        new_moves = useful_moves

    return new_moves

def simulated_annealing(board_original, initial_moves, time_limit=2.0, max_iter=8000,
                        temp_init=2000.0, temp_final=50.0):
    """
    SA com ajustes:
      - Parâmetros de tempo e iterações aumentados.
      - Estratégia de resfriamento exponencial (ajustada).
      - Penaliza soluções que removam Fukus quando a corrente não remove.
    """
    start = time.time()
    current_sol = initial_moves
    current_score, current_y = evaluate_solution_with_y(board_original, current_sol)
    best_sol = current_sol
    best_score = current_score

    for step in range(max_iter):
        if time.time() - start > time_limit:
            break
        frac = step / max_iter
        temp = temp_init * (temp_final / temp_init) ** frac

        candidate = perturb(current_sol, board_original)
        candidate_score, candidate_y = evaluate_solution_with_y(board_original, candidate)
        # Filtragem: se a solução corrente não remove Fukus (y==0) e
        # o candidato remove (candidate_y > 0), penaliza fortemente.
        if current_y == 0 and candidate_y > 0:
            candidate_score -= 10000

        delta = candidate_score - current_score
        if delta > 0:
            current_sol = candidate
            current_score = candidate_score
            current_y = candidate_y
            if candidate_score > best_score:
                best_sol = candidate
                best_score = candidate_score
        else:
            if random.random() < math.exp(delta / temp):
                current_sol = candidate
                current_score = candidate_score
                current_y = candidate_y

    return best_sol, best_score


# ------------------------------------------------------------

# Alteração no refinamento para iterar de forma aleatória e reduzir iterações desnecessárias, evitando TLE.
def refinar_solucao(board_original, moves, max_refine_iter=3000, refine_time_limit=0.5):
    start = time.time()
    best_moves = moves
    best_score = evaluate_solution(board_original, best_moves)
    iter_count = 0
    while iter_count < max_refine_iter and (time.time() - start) < refine_time_limit:
        # Cria uma ordem aleatória para as posições e tenta remover em ordem randômica
        indices = list(range(len(best_moves)))
        random.shuffle(indices)
        improved = False
        for i in indices:
            nova_sol = best_moves[:i] + best_moves[i+1:]
            novo_score = evaluate_solution(board_original, nova_sol)
            if novo_score >= best_score:
                best_moves = nova_sol
                best_score = novo_score
                improved = True
                break  # Reinicia o loop ao encontrar melhoria
        if not improved:
            break
        iter_count += 1
    return best_moves

def main():
    entrada = sys.stdin.readline
    N = int(entrada().strip())
    board_original = [list(entrada().strip()) for _ in range(N)]

    board_copia = copy.deepcopy(board_original)
    initial_moves = construir_solucao_inicial(board_copia, N)

    best_sol, best_score = simulated_annealing(board_original,
                                    initial_moves, time_limit=1, max_iter=3000,
                                    temp_init=2000.0, temp_final=50.0)

    # Aplica refinamento local do tipo greedy para remoção de redundâncias.
    refined_sol = refinar_solucao(board_original, best_sol)
    print("\n".join(refined_sol))

if __name__ == '__main__':
    main()
