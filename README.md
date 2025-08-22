# 🚀 Projet_2: Extraction de Définitions
 
## 🎯 Objectif
Développer un modèle capable de classifier automatiquement des définitions extraites de textes en **bonnes** ou **mauvaises définitions**, à l’aide de **réseaux de neurones** (MLP, CNN, LSTM).

## 📌 Sommaire des tâches
1. **Extraction et Prétraitement des données**
   - Téléchargement des définitions (fichiers `wiki_good.txt` et `wiki_bad.txt`).  
   - Construction d’un DataFrame avec deux colonnes :  
     - `CONTENT` : phrases  
     - `CLASS` : étiquette (`1` = bonne définition, `0` = mauvaise définition).  
   - Nettoyage des textes :  
     - Suppression de parenthèses, chiffres, ponctuations.  
     - Tokenisation, mise en minuscules.  
     - Suppression des stopwords (sauf termes fréquents utiles comme *is, type, an*).  
     - Lemmatisation.  

2. **Construction des jeux de données**
   - Séparation : 80% **train**, 20% **test**.  
   - Tokenisation avec les 5000 mots les plus fréquents.  
   - Séquences normalisées avec `pad_sequences` (longueur 50).  
   - Intégration d’un **embedding pré-entraîné (GloVe, 50d)**.

3. **Modèles de classification testés**
   - **Réseau de neurones simple (MLP)**  
     - Accuracy test : ~83%  
     - Présence d’overfitting (écart moyen train/val ≈ 0.08).  

   - **CNN (1D convolution)**  
     - Accuracy test : ~87%  
     - Performance meilleure mais overfitting plus marqué (écart ≈ 0.10).  

   - **LSTM (128 unités)**  
     - Accuracy test : ~89%  
     - Meilleur modèle avec écart moyen plus faible (~0.06).  
     - Plus adapté pour traiter les séquences textuelles.  

4. **Optimisation des hyperparamètres (LSTM)**
   - Recherche du meilleur **batch_size** et nombre de neurones.  
   - Meilleure configuration trouvée : **(128 neurones, batch_size = 200)**.  
   - Accuracy sur test set : **≈ 90%**.  

5. **Prédiction sur une instance**
   - Test sur un exemple réel → prédiction correcte avec sortie sigmoid > 0.5.  

## 📊 Résultats
- Le **LSTM** est le plus performant pour la tâche d’extraction de définitions.  
- Bonne généralisation avec ~90% d’accuracy sur l’ensemble test.  

## 🛠️ Technologies utilisées
- **Python**  
- **NLTK, Pandas, Numpy, Scikit-learn**  
- **Keras (Tensorflow backend)**  
- **Matplotlib, Seaborn**  
- **GloVe embeddings**

## ✅ Conclusion
Le projet démontre que les réseaux de neurones récurrents (**LSTM**) surpassent les modèles simples (MLP) et convolutionnels (CNN) pour la tâche de classification de définitions, grâce à leur capacité à capturer les dépendances séquentielles du langage.  
