def move_decision_frontier(classifier, poisoning_points, target_class, point_to_hide, max_step=200):
    predicted = classifier.predict([point_to_hide])[0]
    current = predicted
    step = 0
    
    for _ in range(max_step):
        if current != predicted:
            break
        classifier.partial_fit(poisoning_points, [target_class] * len(poisoning_points))
        current = classifier.predict([point_to_hide])[0]
        
    
    return classifier
