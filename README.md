étape 1: Implémenter le réseau de neurones pour être compatible avec le backpropagation
    - Implémenter une classe Layer, qui contient les activations des neurones, et la matrice des weights associée au layer
    - Implémenter une classe Model, à laquelle on pourra append les layers voulus, et qui permettra de lui donner une image, et selon l'activation des neurones,
    choisir celle dont l'activation est la plus grande (entre 0 et 1), et ainsi de suite jusqu'à arriver à l'output layer pour donner la prédiction

étape 2: Préparer et filtrer la donnée à passer au modèle
    - Repérer un jeu de donnée approprié et faire le clean et préparer la donnée pour l'ingestion, pour cela, séparer les données en batches.
    - Ecrire un module qui transforme une image (matrice de pixels) en array

étape 3: Implémenter l'algorithme de backpropagation
    - Cette partie devra créer un nombre de hidden, input et output layers, et voir comment optimiser le nombre de hidden layers.
    - Pour chaque layer, le nombre de neurones doit être entre celui de l'input et l'output layer.
    - Boucler sur plusieurs epochs pour améliorer les weights