import math
import random

from optimization.problem import SmartGridOptimizationProblem
from optimization.result import Configuration, OptimizationResult


def configuration_score(
    problem: SmartGridOptimizationProblem, configuration: Configuration
) -> float:
    """
    Combina cobertura, redundancia y exposición en un puntaje a maximizar.

    Tips:
    - Use problem.score_components(configuration); ya retorna cobertura,
      redundancia y exposición en ese orden.
    """
    cobertura, redundancia, exposicion = problem.score_components(configuration)
    puntaje = cobertura - redundancia - exposicion
    return puntaje

def hill_climbing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    max_iterations: int = 500,
) -> OptimizationResult:
    """
    Ejecuta ascenso de colina con mejora estricta.

    Debe examinar todos los vecinos, seleccionar el de mayor puntaje y
    conservar el orden entregado por el problema para desempatar. La búsqueda
    termina cuando no existe una mejora estricta o se alcanza el límite.

    Tips:
    - problem.neighbors(current) retorna vecinos válidos en el orden que debe
      usarse para desempatar.
    - Cada llamada a configuration_score(...) cuenta como una evaluación.
    - Inicialice los historiales con la configuración inicial y agregue solo las
      mejoras aceptadas antes de retornar el OptimizationResult.
    """
    actual = initial_configuration
    puntaje = configuration_score(problem, actual)
    historial = [actual]
    puntaje_historial = [puntaje]
    evaluaciones = 1
    
    while evaluaciones < max_iterations:
        vecinos = problem.neighbors(actual)
        mejor_vecino = None
        mejor_puntaje = puntaje
        
        for vecino in vecinos:
            puntaje_vecino = configuration_score(problem, vecino)
            evaluaciones += 1
            
            if puntaje_vecino > mejor_puntaje:
                mejor_puntaje = puntaje_vecino
                mejor_vecino = vecino
        
        if mejor_vecino is None:
            break
        
        actual = mejor_vecino
        puntaje = mejor_puntaje
        historial.append(actual)
        puntaje_historial.append(puntaje)

    return OptimizationResult(
        best_configuration=actual,
        best_score=puntaje,
        evaluations=evaluaciones,
        iterations=max_iterations - 1,
        history=historial,
        score_history=puntaje_historial,
    )

def cooling_schedule(initial_temperature: float, cooling_rate: float, iteration: int) -> float:
    """
    Retorna el programa geométrico T(t) = T0 * alpha**t.

    Esta función se invoca desde simulated_annealing en cada iteración.
    """
    return initial_temperature * (cooling_rate ** iteration)


def simulated_annealing(
    problem: SmartGridOptimizationProblem,
    initial_configuration: Configuration,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.97,
    max_iterations: int = 500,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta recocido simulado para un problema de maximización.

    Debe proponer un vecino aleatorio por iteración, aceptar siempre las
    mejoras y aplicar exp(delta / temperature) en los demás casos. El estado
    actual y el mejor estado encontrado deben conservarse por separado.

    Tips:
    - Seleccione el candidato con rng.choice(problem.neighbors(current)) y use
      exclusivamente rng para conservar la reproducibilidad.
    - Obtenga la temperatura con cooling_schedule(...) y calcule la aceptación
      con delta = puntaje_candidato - puntaje_actual y math.exp(...).
    - Mantenga separados el estado actual y el mejor encontrado; registre el
      estado actual después de cada intento, incluso si se rechaza.
    - Detenga la ejecución cuando la temperatura alcance minimum_temperature.
    """
    rng = rng or random.Random()
    minimum_temperature = 1e-9

    actual = initial_configuration
    puntaje = configuration_score(problem, actual)
    mejor = actual
    mejor_puntaje = puntaje

    historial = [actual]
    puntaje_historial = [puntaje]
    evaluaciones = 1
    iteracion = 1

    while iteracion <= max_iterations:
        temperatura = cooling_schedule(initial_temperature, cooling_rate, iteracion)
        if temperatura < minimum_temperature:
            break

        vecino = rng.choice(problem.neighbors(actual))        
        puntaje_vecino = configuration_score(problem, vecino) 
        evaluaciones += 1

        delta = puntaje_vecino - puntaje
        if delta > 0 or (rng.random() < math.exp(delta / temperatura)):
            actual = vecino
            puntaje = puntaje_vecino

        if puntaje > mejor_puntaje:   
            mejor = actual
            mejor_puntaje = puntaje

        historial.append(actual)         
        puntaje_historial.append(puntaje) 

        iteracion += 1

    return OptimizationResult(
        best_configuration=mejor,       
        best_score=mejor_puntaje,       
        evaluations=evaluaciones,
        iterations=iteracion,
        history=historial,
        score_history=puntaje_historial,
    )

def one_point_crossover(
    parent1: Configuration, parent2: Configuration, rng: random.Random
) -> tuple[Configuration, Configuration]:
    """
    Realiza un cruce de un punto y retorna dos descendientes.

    La reparación de la cantidad de módulos se realiza posteriormente.

    Tips:
    - Seleccione con rng un corte interior, entre las posiciones 1 y len-1.
    - Cada descendiente combina el prefijo de un padre con el sufijo del otro.
    - Retorne tuplas y no repare aquí los descendientes.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener la misma longitud")
    if len(parent1) < 2:
        return parent1, parent2

    cut = rng.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + parent2[cut:]
    child2 = parent2[:cut] + parent2[cut:]

    return (child1, child2)

def swap_mutation(
    individual: Configuration, mutation_probability: float, rng: random.Random
) -> Configuration:
    """
    Aplica mutación por intercambio con la probabilidad indicada.

    Cuando ocurre una mutación, intercambia un bit activo y uno inactivo para
    conservar la cantidad de módulos instalados.

    Tips:
    - Use rng.random() para decidir si se aplica la mutación.
    - Identifique por separado los índices activos e inactivos y seleccione uno
      de cada grupo con rng.choice(...).
    - Si alguno de los dos grupos está vacío, no hay un intercambio posible.
    - Retorne una tupla nueva; no modifique el individuo recibido.
    """
    mutation = [x for x in individual]

    if rng.random() <= mutation_probability:
        active_index = []
        inactive_index = []
        for i in range(len(individual)):
            k = individual[i]
            if k == 1:
                active_index.append(i)
            else:
                inactive_index.append(i)
        rand_active = rng.choice(active_index)
        rand_inactive = rng.choice(inactive_index)
        mutation[rand_active] = individual[rand_inactive]
        mutation[rand_inactive] = individual[rand_active]

    mutation = tuple(mutation)

    return mutation


def genetic_algorithm(
    problem: SmartGridOptimizationProblem,
    population_size: int = 40,
    generations: int = 100,
    mutation_probability: float = 0.05,
    elite_size: int = 2,
    rng: random.Random | None = None,
) -> OptimizationResult:
    """
    Ejecuta un algoritmo genético generacional.

    Debe integrar la población inicial, la selección por torneo, el cruce, la
    reparación, la mutación y el elitismo entregados por el proyecto. Retorna
    el mejor individuo encontrado durante toda la ejecución.

    Tips:
    - Use problem.initial_population(...), problem.tournament_select(...) y
      problem.repair_configuration(...) para las operaciones ya entregadas.
    - Aplique one_point_crossover(...) antes de reparar y swap_mutation(...)
      después de la reparación.
    - Conserve los mejores individuos por elitismo y registre en los historiales
      el mejor global de cada generación.
    """
    rng = rng or random.Random()
    if population_size < 2:
        raise ValueError("La población debe tener al menos dos individuos")
    if generations < 0:
        raise ValueError("El número de generaciones no puede ser negativo")
    if not 0.0 <= mutation_probability <= 1.0:
        raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
    if not 0 <= elite_size <= population_size:
        raise ValueError("elite_size debe estar entre 0 y population_size")

    population = problem.initial_population(population_size, rng)
    scores = [configuration_score(problem, config) for config in population]
    evaluations = len(scores)
    best_score = max(scores)
    best_config = population[scores.index(best_score)]
    history = []
    score_history = []
    for gen in range(generations):
        new_pop = []
        new_scores = []
        for ind in range(population_size):
            parent1 = problem.tournament_select(population, scores, rng)
            parent2 = problem.tournament_select(population, scores, rng)
            children = one_point_crossover(parent1, parent2, rng)
            children = tuple([problem.repair_configuration(child, rng) for child
                              in children])
            child1, child2 = [swap_mutation(child, mutation_probability, rng)
                              for child in children]
            new_pop.append(child1)
            new_scores.append(configuration_score(problem, child1))
            new_pop.append(child2)
            new_scores.append(configuration_score(problem, child2))
            evaluations += 2
        sorted_pop = sorted(zip(new_scores, new_pop), reverse=True)
        new_pop = [config for score, config in sorted_pop]
        new_scores = [score for score, config in sorted_pop]
        population = new_pop[:elite_size]
        scores = new_scores[:elite_size]
        history.append(population[0])
        score_history.append(scores[0])
        if scores[0] > best_score:
            best_score = scores[0]
            best_config = population[0]


    result = OptimizationResult(
            best_config,
            best_score,
            evaluations,
            generations,
            history,
            score_history
            )

    return result
