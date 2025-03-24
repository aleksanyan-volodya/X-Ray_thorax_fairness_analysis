# fairness_X-Ray_thorax_images
The aim of the following project is to identify and trying to reduce biases of thorax x ray images in order to fair learn


*NOTE:*  
I, II, III signifient l'ordre de priorité des actions :  
- **I** signifie que c'est important à regarder.  
- **II** signifie que ce n'est pas une priorité.  
- **III** est juste une idée de ce qui est possible de faire, mais dont l'utilité n'est pas encore claire.  

---
## **Pre-processing**  
Avant d'entraîner le modèle, on doit s'assurer que les données sont *fair*.

### **1. Gérer le déséquilibre des âges**  
- **I**)  **Sur-échantillonnage** des classes d'âges sous-représentées.  

### **2. Gérer le déséquilibre des positions d'imagerie (dos vs épaule)**  
On a peu d'images prises depuis l'épaule, donc on risque d'avoir un modèle biaisé contre ces cas.  
- **II**)  **Augmentation de données** : Appliquer *rotations*, *modifications de contraste*, ... pour générer plus d'exemples.  
- **I**)  **Pondération dans la fonction de perte** : Donner un poids plus important aux images d'épaule.  

### **3. Normalisation et pré-processing général**  
- **III**)  **Normalisation des images** (intéressant mais je ne sais pas comment faire).  
- **I**)  **Vérifier les fuites de données** : Revoir si les métadonnées ne sont pas exploitées involontairement par le modèle.  

---

## **Post-processing et analyse des biais**  
Une fois le modèle entraîné, on doit vérifier s'il présente des biais et, si nécessaire, appliquer des corrections.  

### **1. Détection des biais**  
- **I**)  **Comparer la performance du modèle selon l'âge**.  
- **I**)  **Comparer la performance selon le sexe** : Calculer l'AUC, la précision, le taux de faux positifs/faux négatifs séparément pour les hommes et les femmes.  
- **I**)  **Comparer la performance selon la position d'imagerie** : Vérifier si les images d'épaule sont mal classifiées.  

### **2. Réduction des biais**  
- **I**)  **Re-pondérer la fonction de perte** : Augmenter l'importance des classes sous-représentées dans la loss.  
- **III**)  **Débiasing adversarial** : Entraîner un deuxième modèle pour détecter et supprimer les biais dans les prédictions.  
- **III**)  **Calibration post-processing** : Appliquer **Platt Scaling** ou **Isotonic Regression** pour homogénéiser la confiance du modèle entre groupes.  

---

## **Plan d'action**  
1. **Effectuer une première analyse des biais en bref** après entraînement du modèle.  
2. **Sélectionner une méthode de correction** :  
   - Problème en tranches d'âge -> **Ré-échantillonnage ou pondération des classes**.  
   - Problème de genre -> **Ajustement des seuils**.  
   - Problème de position d'imagerie -> **Augmentation des données + re-pondération**.  
3. **Réentraîner le modèle et évaluer l'impact** des corrections.  
