# projet_mnist
Francais:
Projet d’IA embarquée dédié à la reconnaissance de chiffres manuscrits en temps réel.
Le projet consiste à entraîner et comparer deux architectures de réseaux de neurones (MLP et CNN) sur la base de données MNIST ainsi que sur un jeu de données personnalisé, en appliquant un prétraitement identique à celui utilisé en conditions embarquées.

Les modèles sont implémentés from scratch en C/C++, sans recours à des bibliothèques de deep learning, afin de garantir une maîtrise complète de l’inférence et une compatibilité avec les contraintes matérielles.
Le système est déployé sur une carte Raspberry Pi et permet la détection et la classification de chiffres capturés par une caméra en temps réel.

Le projet met l’accent sur la cohérence entre l’entraînement et l’inférence embarquée, la comparaison des performances (précision et temps d’inférence) ainsi que sur les compromis entre complexité des modèles et contraintes d’implémentation.
English
Embedded AI project focused on real-time handwritten digit recognition.
The project involves training and comparing two neural network architectures (MLP and CNN) on the MNIST dataset as well as on a custom dataset, using a preprocessing pipeline consistent with the embedded inference stage.

The trained models are implemented from scratch in C/C++, without relying on deep learning libraries, ensuring full control over the inference process and compatibility with embedded constraints.
The system is deployed on a Raspberry Pi and performs real-time digit detection and classification from a camera stream.

The project emphasizes the importance of training–inference consistency, performance evaluation (accuracy and inference time), and the trade-offs between model complexity and embedded implementation constraints.
