from abc import ABC, abstractmethod
import math

from algorithms.evaluation import evaluation_function
from world.game_state import GameState


class MultiAgentSearchAgent(ABC):
    """Clase base para los agentes de búsqueda adversaria."""

    def __init__(self, depth: int | str = 2) -> None:
        self.depth = int(depth)
        if self.depth < 1:
            raise ValueError("La profundidad debe ser al menos 1 ply")
        self.nodes_evaluated = 0

    @abstractmethod
    def get_action(self, state: GameState) -> str | None:
        raise NotImplementedError


class MinimaxAgent(MultiAgentSearchAgent):
    """Agente Minimax para el defensor MAX frente al intruso MIN."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción del defensor con mayor valor Minimax.

        El defensor es MAX (agente 0), el intruso es MIN (agente 1) y cada
        acción consume un ply. Debe respetar el orden de las acciones legales,
        usar evaluation_function en terminales y cortes, y contar cada estado
        procesado una vez en self.nodes_evaluated, incluida la raíz.

        Tips:
        - Use state.get_legal_actions(agent_index) y
          state.generate_successor(agent_index, action) para expandir el árbol.
        - Compruebe state.is_win(), state.is_lose() y el corte de profundidad;
          evalúe esos estados con evaluation_function(state).
        - El siguiente agente es (agent_index + 1) % state.get_num_agents().
          depth=1 incluye una acción de MAX y depth=2 una de MAX y una de MIN.
        - Reinicie las métricas y cuente una vez cada estado procesado, incluida
          la raíz. Retorne la acción de MAX y conserve la primera en los empates.
        """
        # TODO: Add your code here
        # 1. Reiniciar el contador de nodos evaluados al inicio de get_action
        self.nodes_evaluated = 0
        num_agents = state.get_num_agents()

        # 2. Función para calcular el valor Minimax de los estados
        def minimax(current_state: GameState, current_depth: int, agent_index: int) -> float:
          
            self.nodes_evaluated += 1

            # Caso base: Estado terminal (victoria/derrota)
            if current_state.is_win() or current_state.is_lose() or current_depth == 0:
                return evaluation_function(current_state)

            legal_actions = current_state.get_legal_actions(agent_index)
            if not legal_actions:
                return evaluation_function(current_state)

            next_agent = (agent_index + 1) % num_agents
            next_depth = current_depth - 1

            # Nodo MAX (Defensor / Agente 0): Busca el valor más alto
            if agent_index == 0:
                max_eval = -math.inf
                for action in legal_actions:
                    successor = current_state.generate_successor(agent_index, action)
                    eval_val = minimax(successor, next_depth, next_agent)
                    if eval_val > max_eval:
                        max_eval = eval_val
                return max_eval

            # Nodo MIN (Intruso / Agente 1): Busca el valor más bajo
            else:
                min_eval = math.inf
                for action in legal_actions:
                    successor = current_state.generate_successor(agent_index, action)
                    eval_val = minimax(successor, next_depth, next_agent)
                    if eval_val < min_eval:
                        min_eval = eval_val
                return min_eval

        # 3. Selección de la mejor acción en la Raíz (Defensor / Agente 0)
        self.nodes_evaluated += 1

        best_action = None
        best_value = -math.inf
        legal_actions = state.get_legal_actions(0)

        for action in legal_actions:
            successor = state.generate_successor(0, action)
            # Se evalua el sucesor: se pasa al agente 1 (MIN) y se descuenta 1 ply de profundidad
            action_value = minimax(successor, self.depth - 1, 1)

            # Se conserva la mejor acción.
            if action_value > best_value:
                best_value = action_value
                best_action = action

        return best_action
        


class AlphaBetaAgent(MultiAgentSearchAgent):
    """Agente Minimax que evita explorar ramas mediante poda alfa-beta."""

    def get_action(self, state: GameState) -> str | None:
        """
        Retorna la acción de Minimax aplicando poda alfa-beta.

        Debe usar la misma profundidad, orden de acciones y función de
        evaluación que Minimax.

        Tips:
        - Conserve la misma estructura y casos base de MinimaxAgent.
        - Inicie alpha en -infinito y beta en +infinito, y páselos en las
          llamadas recursivas.
        - En MAX actualice alpha y corte si valor >= beta; en MIN actualice beta
          y corte si valor <= alpha.
        """
        # TODO: Add your code here
        self.nodes_evaluated = 0
        num_agents = state.get_num_agents()
        
        def alphabeta(current_state: GameState, current_depth: int, agent_index: int, alpha: float, beta: float) -> float:
            self.nodes_evaluated += 1

            # Caso base: Estado terminal (victoria/derrota) o profundidad alcanzada
            if current_state.is_win() or current_state.is_lose() or current_depth == 0:
                return evaluation_function(current_state)

            # Expandir sucesores
            legal_actions = current_state.get_legal_actions(agent_index) 
            if not legal_actions:
                return evaluation_function(current_state)

            # Determinar el siguiente agente y la profundidad para la llamada recursiva
            next_agent = (agent_index + 1) % num_agents
            next_depth = current_depth - 1

            # MAX
            if agent_index == 0: 
                max_eval = -float('inf')
                for action in legal_actions:
                    successor = current_state.generate_successor(agent_index, action)
                    eval_val = alphabeta(successor, next_depth, next_agent, alpha, beta)
                    
                    # Actualizar alpha y el valor máximo
                    if eval_val > max_eval:
                        max_eval = eval_val
                        
                    # Poda beta    
                    if max_eval >= beta:
                      return max_eval
                    
                    alpha = max(alpha, max_eval)
                return max_eval
              
            # MIN
            else:  
                min_eval = float('inf')
                for action in legal_actions:
                    successor = current_state.generate_successor(agent_index, action)
                    eval_val = alphabeta(successor, next_depth, next_agent, alpha, beta)
                   
                    # Actualizar beta y el valor mínimo 
                    if eval_val < min_eval:
                        min_eval = eval_val
                    
                    # Poda alfa
                    if min_eval <= alpha:
                      return min_eval

                    beta = min(beta, min_eval)
                return min_eval
              
        # Raíz del árbol
        self.nodes_evaluated += 1
        best_action = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        legal_actions = state.get_legal_actions(0)
        
        for action in legal_actions:
            successor = state.generate_successor(0, action)
            action_value = alphabeta(successor, self.depth - 1, 1, alpha, beta)

            if action_value > best_value:
                best_value = action_value
                best_action = action

            # Actualizar alpha después de evaluar la acción
            if best_value >= beta:
                return best_action
            alpha = max(alpha, best_value)
        return best_action