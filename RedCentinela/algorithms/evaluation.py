import math

from world.game_state import GameState


def base_evaluation_function(state: GameState) -> float:
    """
    Retorna la evaluación base entregada para desarrollar el punto 4.

    Esta función no forma parte del código que debe modificar el estudiante y
    permite probar Minimax antes de desarrollar la heurística del punto 5.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0
    return float(state.get_score())


def evaluation_function(state: GameState) -> float:
    """
    Evalúa un estado desde la perspectiva del defensor MAX.

    Debe conservar las utilidades terminales de la evaluación base y diseñar
    una valoración no trivial para estados de corte. Minimax y alfa-beta usan
    esta misma función al comparar sus decisiones en el punto 5.

    Tips:
    - Los estados terminales ya se resuelven antes del bloque TODO; diseñe allí
      únicamente la valoración de estados no terminales.
    - Consulte state.defender_position, state.intruder_position,
      state.pending_terminals, state.get_score() y state.get_legal_actions(0).
    - state.layout.distance(start, goal) calcula y almacena en caché la distancia
      real por el mapa respetando los muros.
    - Maneje conjuntos vacíos y distancias infinitas, y mantenga todo estado no
      terminal estrictamente entre -1000 y +1000.
    """
    if state.is_win() or state.is_lose():
        return base_evaluation_function(state)

    # TODO: Add your code here
    # Evaluar la posición del defensor y el intruso
    puntaje = state.get_score()
    def_pos = state.defender_position
    intr_pos = state.intruder_position
    pending_terminals = state.pending_terminals
    legal_actions = state.get_legal_actions(0)
    
    # Distancia entre el defensor y el intruso
    distance = state.layout.distance(def_pos, intr_pos)
    if math.isinf(distance):
        distance = 100.0  
    
    componente_puntaje = puntaje * 2.0 # Ponderar el puntaje
    componente_proximidad_intruso = -3.0 * distance  # Penalizar la proximidad del intruso
    
    # Distancia a la terminal pendienta más cercana amenazada por el intruso
    componente_terminal = 0.0
    if pending_terminals:
      distancia_term = [state.layout.distance(intr_pos, term) for term in pending_terminals]
      distancias_validas = [d for d in distancia_term if not math.isinf(d)]
      
      if distancias_validas:
          min_dist = min(distancias_validas)
          componente_terminal = -2.0 * max(0.0, 10.0 - min_dist)  # Penalizar la cercanía a la terminal
    
    # Movilidad disponible del defensor
    componente_movilidad = 0.5 * len(legal_actions)  
    valor_final = componente_puntaje + componente_proximidad_intruso + componente_terminal + componente_movilidad
    
     # Asegurar que el valor esté en el rango
    valor_final = max(-999.0, min(999.0, valor_final)) 
    return valor_final
