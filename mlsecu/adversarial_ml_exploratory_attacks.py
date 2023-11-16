def get_adversarial_points(classifier, candidates, attack, dist_function, epsilon):
    
    originals = []
    altered = []
    epsilons = []
    

    for i in range(len(candidates)):
        candidate = candidates[i]
        adversarial_ex, current_epsilon = attack(classifier, candidate, dist_function, step=0.01, epsilon=epsilon)
        if adversarial_ex is  None:
            continue
        originals.append(candidate)
        altered.append(adversarial_ex)
        epsilons.append(current_epsilon)
    return originals, altered, epsilons
    

def get_smallest_alteration(classifier, candidates, attack, dist_function):
    
    origin, altered = None, None
    epsilon = float('inf')
    

    for candidate in candidates:
        example, current_epsilon = attack(classifier, candidate, dist_function, step=0.01)

        if not(example is not None and current_epsilon < epsilon):
            continue
        origin, altered, epsilon = candidate, example, current_epsilon

    return origin, altered, epsilon