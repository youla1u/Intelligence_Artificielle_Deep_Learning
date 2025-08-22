# 🌾 Projet_1: Classification de routes et champs par CNN

## Description
Ce projet explore la classification d'images en deux catégories : **routes** et **champs**, en comparant trois types de représentations :
- Images en **RGB** (couleur originale)
- Images en **niveaux de gris (L)**
- Images en **niveaux de gris avec égalisation locale d'histogramme (L-LHE)**

L'objectif est de déterminer si la couleur est nécessaire ou si des pré-traitements des images améliorent la performance d'un réseau de neurones convolutionnel (CNN).

---

## Jeu de données
- Nombre total d’images : **90**
- Catégories : `route`, `champ`
- Taille des images : 150 × 150 pixels

| Type de données | Description |
|-----------------|------------|
| RGB             | Image couleur originale |
| L               | Conversion en niveaux de gris |
| L-LHE           | Niveaux de gris + égalisation locale d’histogramme |

---

## Architecture du modèle CNN
- 2 blocs convolution + max pooling
- 1 couche `Flatten`
- 1 couche dense (128 neurones, ReLU)
- 1 couche de sortie (sigmoïde pour classification binaire)
- Dropout = 0.5 pour régularisation
- Optimiseur : `Adam`
- Fonction de perte : `binary_crossentropy`

---

## Résultats

| Jeu de données | Précision entraînement | Précision validation |
|----------------|------------------------|---------------------|
| RGB            | ~87,5 %               | ~66,7 %             |
| L (gris)       | ~90 %                 | ~83,3 %             |
| L-LHE          | ~75 % (instable)      | ~50–55 %            |

### Interprétation
- ✅ **Niveaux de gris (L)** : meilleur compromis entre apprentissage et généralisation.
- ⚠️ **RGB** : surapprentissage possible.
- ❌ **L-LHE** : égalisation locale trop perturbante pour le CNN.

---

## Améliorations possibles
- Augmentation de données (rotations, flips, variations de luminosité)
- Régularisation des couches convolutionnelles (`kernel_regularizer=L2`)
- Ajustement de la taille du batch (8 ou 16)
- Early stopping pour éviter le surapprentissage
- Validation croisée k-fold pour plus de robustesse
- Apprentissage par transfert avec modèles pré-entraînés (VGG16, MobileNetV2, ResNet)

---

## Conclusion
- La **couleur n’est pas indispensable** pour ce jeu de données.
- L’**égalisation locale** n’améliore pas la performance et peut la dégrader.
- Avec un petit dataset, l’**augmentation de données** et/ou **l’apprentissage par transfert** est fortement recommandé.

---

## Exemple d’utilisation (Python/Keras)
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


**-----------------------------------------------------------------------------------------------------------------------------------------------------**


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
