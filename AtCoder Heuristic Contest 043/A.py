import sys, math, random
from heapq import heappush, heappop

Pos = tuple[int, int]
EMPTY = -1
DO_NOTHING = -1
STATION = 0
RAIL_HORIZONTAL = 1
RAIL_VERTICAL = 2
RAIL_LEFT_DOWN = 3
RAIL_LEFT_UP = 4
RAIL_RIGHT_UP = 5
RAIL_RIGHT_DOWN = 6
COST_STATION = 5000
COST_RAIL = 100


class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.parents = [-1] * (n * n)

    def _find_root(self, idx: int) -> int:
        if self.parents[idx] < 0:
            return idx
        self.parents[idx] = self._find_root(self.parents[idx])
        return self.parents[idx]

    def is_same(self, p: Pos, q: Pos) -> bool:
        return self._find_root(p[0]*self.n + p[1]) == self._find_root(q[0]*self.n + q[1])

    def unite(self, p: Pos, q: Pos) -> None:
        p_idx = p[0]*self.n + p[1]
        q_idx = q[0]*self.n + q[1]
        p_root = self._find_root(p_idx)
        q_root = self._find_root(q_idx)
        if p_root != q_root:
            if self.parents[p_root] > self.parents[q_root]:
                p_root, q_root = q_root, p_root
            self.parents[p_root] += self.parents[q_root]
            self.parents[q_root] = p_root


def distance(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Action:
    def __init__(self, type: int, pos: Pos):
        self.type = type
        self.pos = pos

    def __str__(self):
        if self.type == DO_NOTHING:
            return "-1"
        else:
            return f"{self.type} {self.pos[0]} {self.pos[1]}"


class Result:
    def __init__(self, actions: list[Action], score: int):
        self.actions = actions
        self.score = score

    def __str__(self):
        return "\n".join(map(str, self.actions))


class Field:
    def __init__(self, N: int):
        self.N = N
        self.rail = [[EMPTY] * N for _ in range(N)]
        self.uf = UnionFind(N)

    def build(self, type: int, r: int, c: int) -> None:
        # Se a célula já tem uma estação, não permite construir outro rail nela.
        if self.rail[r][c] == STATION:
            return False

        # Para rails, a célula deve estar vazia
        if 1 <= type <= 6 and self.rail[r][c] != EMPTY:
            return False

        self.rail[r][c] = type

        # Conecta células adjacentes (mesmo código existente)
        # top
        if type in (STATION, RAIL_VERTICAL, RAIL_LEFT_UP, RAIL_RIGHT_UP):
            if r > 0 and self.rail[r - 1][c] in (STATION, RAIL_VERTICAL, RAIL_LEFT_DOWN, RAIL_RIGHT_DOWN):
                self.uf.unite((r, c), (r - 1, c))
        # bottom
        if type in (STATION, RAIL_VERTICAL, RAIL_LEFT_DOWN, RAIL_RIGHT_DOWN):
            if r < self.N - 1 and self.rail[r + 1][c] in (STATION, RAIL_VERTICAL, RAIL_LEFT_UP, RAIL_RIGHT_UP):
                self.uf.unite((r, c), (r + 1, c))
        # left
        if type in (STATION, RAIL_HORIZONTAL, RAIL_LEFT_DOWN, RAIL_LEFT_UP):
            if c > 0 and self.rail[r][c - 1] in (STATION, RAIL_HORIZONTAL, RAIL_RIGHT_DOWN, RAIL_RIGHT_UP):
                self.uf.unite((r, c), (r, c - 1))
        # right
        if type in (STATION, RAIL_HORIZONTAL, RAIL_RIGHT_DOWN, RAIL_RIGHT_UP):
            if c < self.N - 1 and self.rail[r][c + 1] in (STATION, RAIL_HORIZONTAL, RAIL_LEFT_DOWN, RAIL_LEFT_UP):
                self.uf.unite((r, c), (r, c + 1))

        return True

    def is_connected(self, s: Pos, t: Pos) -> bool:

        stations_s = self.collect_stations(s)
        stations_t = self.collect_stations(t)
        if not stations_s or not stations_t:
            return False

        groups_s = { self.uf._find_root(pos[0] * self.N + pos[1]) for pos in stations_s }
        groups_t = { self.uf._find_root(pos[0] * self.N + pos[1]) for pos in stations_t }
        return not groups_s.isdisjoint(groups_t)

    def collect_stations(self, pos: Pos) -> list[Pos]:
        stations = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) > 2:
                    continue
                r = pos[0] + dr
                c = pos[1] + dc
                if 0 <= r < self.N and 0 <= c < self.N and self.rail[r][c] == STATION:
                    stations.append((r, c))
        return stations


class Solver:
    def __init__(self, N: int, M: int, K: int, T: int, home: list[Pos], workplace: list[Pos]):
        self.N = N
        self.M = M
        self.K = K
        self.T = T
        self.home = home
        self.workplace = workplace
        self.field = Field(N)
        self.money = K
        self.actions = []

    def calc_income(self) -> int:
        # Se já calculamos antes, retornamos o valor em cache
        if hasattr(self, "_income_cache"):
            return self._income_cache

        income = 0
        for i in range(self.M):
            if self.field.is_connected(self.home[i], self.workplace[i]):
                income += distance(self.home[i], self.workplace[i])

        self._income_cache = income
        return income

    def build_rail(self, rail_type: int, r: int, c: int) -> bool:
        if self.money < COST_RAIL:
            return False
        self.money -= COST_RAIL
        ok = self.field.build(rail_type, r, c)
        if not ok:
            self.money += COST_RAIL
            return False
        if hasattr(self, "_income_cache"):
            del self._income_cache
        self.actions.append(Action(rail_type, (r, c)))
        return True

    def build_station(self, r: int, c: int) -> bool:
        if self.money < COST_STATION:
            return False
        self.money -= COST_STATION

        ok = self.field.build(STATION, r, c)
        if not ok:
            self.money += COST_STATION
            return False

        if hasattr(self, "_income_cache"):
            del self._income_cache

        self.actions.append(Action(STATION, (r, c)))
        return True

    def build_nothing(self) -> None:
        self.actions.append(Action(DO_NOTHING, (0, 0)))

    def solve(self) -> Result:
        # Tenta conectar quantos passageiros for possível com o dinheiro disponível.
        for i in range(self.M):
            # Se já estiver conectado, pula para o próximo.
            if self.field.is_connected(self.home[i], self.workplace[i]):
                continue

            # Verifica se há estações próximas; se não, constrói.
            if not self.field.collect_stations(self.home[i]):
                if self.money >= COST_STATION:
                    self.build_station(*self.home[i])
                else:
                    continue  # Sem dinheiro para construir estação.
            if not self.field.collect_stations(self.workplace[i]):
                if self.money >= COST_STATION:
                    self.build_station(*self.workplace[i])
                else:
                    continue

            # Conecta as estações usando caminho Manhattan, podendo aproveitar trilhos já existentes.
            r0, c0 = self.home[i]
            r1, c1 = self.workplace[i]
            if r0 < r1:
                for r in range(r0 + 1, r1):
                    if self.money < COST_RAIL:
                        break
                    if not self.build_rail(RAIL_VERTICAL, r, c0):
                        continue
                if c0 < c1 and self.money >= COST_RAIL:
                    self.build_rail(RAIL_RIGHT_UP, r1, c0)
                elif c0 > c1 and self.money >= COST_RAIL:
                    self.build_rail(RAIL_LEFT_UP, r1, c0)
            elif r0 > r1:
                for r in range(r0 - 1, r1, -1):
                    if self.money < COST_RAIL:
                        break
                    self.build_rail(RAIL_VERTICAL, r, c0)
                if c0 < c1 and self.money >= COST_RAIL:
                    self.build_rail(RAIL_RIGHT_DOWN, r1, c0)
                elif c0 > c1 and self.money >= COST_RAIL:
                    self.build_rail(RAIL_LEFT_DOWN, r1, c0)

            if c0 < c1:
                for c in range(c0 + 1, c1):
                    if self.money < COST_RAIL:
                        break
                    self.build_rail(RAIL_HORIZONTAL, r1, c)
            elif c0 > c1:
                for c in range(c0 - 1, c1, -1):
                    if self.money < COST_RAIL:
                        break
                    self.build_rail(RAIL_HORIZONTAL, r1, c)

            # Atualiza o saldo com a renda gerada pelas conexões.
            income = self.calc_income()
            self.money += income

        # Para os turnos restantes, não há ação (build nothing).
        while len(self.actions) < self.T:
            self.build_nothing()
            self.money += self.calc_income()

        return Result(self.actions, self.money)


def simulate_solution(actions: list[Action], N: int, M: int, K: int, T: int, home: list[Pos], workplace: list[Pos]) -> int:
    """
    Simula a execução de uma sequência de ações (solução candidata) e retorna o score final.
    Essa função recria um novo Solver e “reexecuta” os T turnos usando as ações candidatas.
    """
    # Cria uma cópia do estado inicial
    solver = Solver(N, M, K, T, home, workplace)
    # Zera as ações e reinicializa o campo e o dinheiro
    solver.actions = []
    solver.money = K
    solver.field = Field(N)
    turn = 0
    # Reaplica as ações candidato para cada turno (assumindo que len(actions)==T)
    for action in actions:
        if action.type == DO_NOTHING:
            pass  # não constrói nada
        elif action.type == STATION:
            if solver.money < COST_STATION:
                continue
            # Tenta construir a estação (se possível)
            solver.money -= COST_STATION
            ok = solver.field.build(action.type, action.pos[0], action.pos[1])
            if not ok:
                solver.money += COST_STATION
        else:
            if solver.money < COST_RAIL:
                continue
            # Para trilhos
            solver.money -= COST_RAIL
            ok = solver.field.build(action.type, action.pos[0], action.pos[1])
            if not ok:
                solver.money += COST_RAIL
        # Ao fim de cada turno, coleta renda
        income = solver.calc_income()
        solver.money += income
        turn += 1
    return solver.money

def perturb_solution(actions: list[Action], T: int) -> list[Action]:
    """
    Faz uma pequena modificação (perturbação) na solução candidata.
    Por exemplo, escolhe um turno aleatório e altera a ação:
      - Se for -1, tenta inserir uma construção aleatória (dentro de limites razoáveis);
      - Se já for uma construção, com certa probabilidade a substitui por -1 ou altera a posição.
    """
    new_actions = actions.copy()
    idx = random.randint(0, T - 1)
    old = new_actions[idx]
    # Definindo uma perturbação simples:
    if old.type == DO_NOTHING:
        # Tenta construir uma ação de rail ou estação em uma posição aleatória
        new_type = random.choice([STATION, RAIL_HORIZONTAL, RAIL_VERTICAL, RAIL_LEFT_DOWN, RAIL_LEFT_UP, RAIL_RIGHT_UP, RAIL_RIGHT_DOWN])
        new_pos = (random.randint(0, 49), random.randint(0, 49))  # supondo N==50
        new_actions[idx] = Action(new_type, new_pos)
    else:
        # Com 50% de chance, transforma a ação em DO_NOTHING
        if random.random() < 0.5:
            new_actions[idx] = Action(DO_NOTHING, (0, 0))
        else:
            # Altera levemente a posição (com limite para não sair do grid)
            r, c = old.pos
            r_new = max(0, min(49, r + random.randint(-1, 1)))
            c_new = max(0, min(49, c + random.randint(-1, 1)))
            new_actions[idx] = Action(old.type, (r_new, c_new))
    return new_actions

def simulated_annealing(initial_actions: list[Action], N: int, M: int, K: int, T: int, home: list[Pos], workplace: list[Pos],
                          iterations=1000, T0=1000, alpha=0.995):
    current_solution = initial_actions
    current_score = simulate_solution(current_solution, N, M, K, T, home, workplace)
    best_solution = current_solution
    best_score = current_score
    temp = T0
    for it in range(iterations):
        candidate = perturb_solution(current_solution, T)
        candidate_score = simulate_solution(candidate, N, M, K, T, home, workplace)
        delta = candidate_score - current_score
        if delta >= 0 or random.random() < math.exp(delta / temp):
            current_solution = candidate
            current_score = candidate_score
            if candidate_score > best_score:
                best_solution = candidate
                best_score = candidate_score
        temp *= alpha
        if temp < 1e-3:
            break
    return Result(best_solution, best_score)



def main_file():
    with open("/workspaces/Atcoder_Heuristic/AtCoder Heuristic Contest 043/in/0003.txt", "r") as f:
        N, M, K, T = map(int, f.readline().split())
        home = []
        workplace = []
        for _ in range(M):
            r0, c0, r1, c1 = map(int, f.readline().split())
            home.append((r0, c0))
            workplace.append((r1, c1))

    solver = Solver(N, M, K, T, home, workplace)
    initial_result = solver.solve()
    initial_actions = initial_result.actions
    initial_score = initial_result.score

    result = simulated_annealing(initial_actions, N, M, K, T, home, workplace,
                                                    iterations=1000, T0=1000, alpha=0.995)

    with open("output.txt", "w") as f:
        f.write(str(result))
    print(f"score={result.score}", file=sys.stderr)

def main():
    N, M, K, T = map(int, input().split())
    home = []
    workplace = []
    for _ in range(M):
        r0, c0, r1, c1 = map(int, input().split())
        home.append((r0, c0))
        workplace.append((r1, c1))

    solver = Solver(N, M, K, T, home, workplace)
    initial_result = solver.solve()
    initial_actions = initial_result.actions
    initial_score = initial_result.score

    result = simulated_annealing(initial_actions, N, M, K, T, home, workplace,
                                                    iterations=1000, T0=1000, alpha=0.995)
    print(result)
    #print(f"score={result.score}", file=sys.stderr)

if __name__ == "__main__":
    #main()
    main_file()
