#                  Projet d’Intelligence Artificielle : Deep Learning 

Ce dépôt contient deux projets explorant l’utilisation des réseaux de neurones pour des tâches différentes :  

1. **Projet_1 : Classification d’images de routes et champs avec CNN**  
2. **Projet_2 : Extraction et classification de définitions textuelles avec MLP, CNN et LSTM**  

---

# 🌾 Projet_1: Classification de routes et champs par CNN

## Description
Ce projet explore la **classification d’images** en deux catégories : **routes** et **champs**, en comparant trois types de représentations :  

- **RGB** : images couleur originales  
- **Niveaux de gris (L)** : conversion en nuances de gris  
- **Niveaux de gris + égalisation locale (L-LHE)** : pré-traitement pour améliorer le contraste  

**Objectif** : déterminer si la couleur est nécessaire ou si des pré-traitements améliorent la performance d’un CNN.

---

## Jeu de données
- **Nombre total d’images** : 90  
- **Catégories** : `route`, `champ`  
- **Taille des images** : 150 × 150 pixels  

| Type de données | Description |
|-----------------|------------|
| RGB             | Image couleur originale |
| L               | Conversion en niveaux de gris |
| L-LHE           | Niveaux de gris + égalisation locale d’histogramme |

---

## Architecture du modèle CNN
- **2 blocs convolution + max pooling**  
- **Couche Flatten** pour aplatir les feature maps  
- **Couche Dense** : 128 neurones, activation ReLU  
- **Couche de sortie** : 1 neurone, activation sigmoïde  
- **Dropout** : 0.5 (régularisation)  
- **Optimiseur** : Adam  
- **Fonction de perte** : binary_crossentropy  

---

## Résultats

| Jeu de données | Précision entraînement | Précision validation |
|----------------|------------------------|---------------------|
| RGB            | ~87,5 %               | ~66,7 %             |
| L (gris)       | ~90 %                 | ~83,3 %             |
| L-LHE          | ~75 % (instable)      | ~50–55 %            |

### Interprétation
- ✅ **Niveaux de gris (L)** : meilleur compromis apprentissage / généralisation  
- ⚠️ **RGB** : risque de surapprentissage  
- ❌ **L-LHE** : égalisation locale trop perturbante pour le CNN  

---

## Améliorations possibles
- Augmentation de données (rotations, flips, variations de luminosité)  
- Régularisation des couches convolutionnelles (`kernel_regularizer=L2`)  
- Ajustement de la taille du batch (8 ou 16)  
- Early stopping pour limiter le surapprentissage  
- Validation croisée k-fold pour plus de robustesse  
- Apprentissage par transfert avec modèles pré-entraînés (VGG16, MobileNetV2, ResNet)  

---

## Conclusion
- La **couleur n’est pas indispensable**  
- L’**égalisation locale** peut dégrader les performances  
- Avec un petit dataset, **augmentation de données** et/ou **apprentissage par transfert** sont fortement recommandés  

---

# 🚀 Projet_2: Extraction et classification de définitions

## Objectif
Classifier automatiquement des définitions en **bonnes** ou **mauvaises**, à l’aide de **réseaux de neurones** (MLP, CNN, LSTM).

---

## Prétraitement des données
- Fichiers : `wiki_good.txt` et `wiki_bad.txt`  
- Création d’un DataFrame :  
  - `CONTENT` : texte de la définition  
  - `CLASS` : étiquette (`1` = bonne, `0` = mauvaise)  
- Nettoyage du texte :  
  - Suppression parenthèses, chiffres, ponctuation  
  - Conversion en minuscules, tokenisation  
  - Suppression des stopwords (sauf *is, type, an*)  
  - Lemmatisation  

---

## Construction des jeux de données
- Séparation : 80% **train**, 20% **test**  
- Tokenisation : 5000 mots les plus fréquents  
- Séquences normalisées (`pad_sequences`, longueur 50)  
- Embedding pré-entraîné **GloVe (50 dimensions)**  

---

## Modèles testés

| Modèle | Accuracy test | Commentaire |
|--------|---------------|------------|
| MLP    | ~83%          | Overfitting léger (écart train/val ≈ 0.08) |
| CNN 1D | ~87%          | Performance meilleure mais overfitting plus marqué (écart ≈ 0.10) |
| LSTM   | ~89%          | Meilleur modèle, écart moyen plus faible (~0.06), adapté aux séquences |

---

## Optimisation des hyperparamètres (LSTM)
- Meilleure configuration : **128 neurones, batch_size = 200**  
- Accuracy sur test set : **≈ 90%**

---

## Prédiction sur une instance
- Exemple réel → prédiction correcte (sigmoid > 0.5)  

---

## Résultats
- Le **LSTM** est le modèle le plus performant  
- Bonne généralisation avec **≈ 90% d’accuracy** sur le test set  

---

## Technologies utilisées
- **Python**  
- **NLTK, Pandas, Numpy, Scikit-learn**  
- **Keras (Tensorflow backend)**  
- **Matplotlib, Seaborn**  
- **GloVe embeddings**  

---

## Conclusion
- Les **LSTM** surpassent les modèles MLP et CNN  
- Leur capacité à capturer les **dépendances séquentielles du langage** explique la performance  

---
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

