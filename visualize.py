import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import plotly.express as px

def fairness_metrics_for_single_prediction(csv_path: str, sensitive_col: str = 'Patient Gender', labels_col: str = 'labels', preds_col: str = 'preds'):
    """
    Calcul des métriques de fairness pour une seule prédiction (fichier CSV) en fonction de l'attribut sensible.
    
    :param csv_path: Chemin vers le fichier CSV des prédictions
    :param sensitive_col: Colonne des attributs sensibles (par exemple, 'Patient Gender')
    :param labels_col: Colonne des vraies étiquettes (labels)
    :param preds_col: Colonne des prédictions
    :return: Dictionnaire des métriques de fairness pour chaque groupe
    """
    # Lire le CSV des prédictions
    df = pd.read_csv(csv_path)

    # Séparer les données selon l'attribut sensible (ici le genre)
    groups = df[sensitive_col].unique()
    
    fairness_results = []

    # Variables pour calculer le Disparate Impact
    group_positive_rates = {}
    
    for group in groups:
        group_df = df[df[sensitive_col] == group]
        
        # Calcul de la matrice de confusion pour chaque groupe
        tn, fp, fn, tp = confusion_matrix(group_df[labels_col], group_df[preds_col]).ravel()
        
        # Calcul de la Balanced Accuracy pour chaque groupe
        balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2
        
        # Calcul de l'Accuracy pour chaque groupe
        accuracy = accuracy_score(group_df[labels_col], group_df[preds_col])
        
        # Calcul du taux de classification positive pour chaque groupe
        positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Enregistrer les résultats pour chaque groupe
        fairness_results.append({
            'Group': group,
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_accuracy,
            'Positive Rate': positive_rate
        })
        
        # Enregistrer les taux de classification positive pour le calcul du Disparate Impact
        group_positive_rates[group] = positive_rate

    # Convertir les résultats en DataFrame pour affichage
    fairness_results_df = pd.DataFrame(fairness_results)
    
    # Calcul du Disparate Impact entre les groupes : Comparaison du taux de classification positive entre les groupes
    if len(groups) == 2:  # On ne peut calculer le Disparate Impact que pour 2 groupes
        group_1, group_2 = groups
        disparate_impact = group_positive_rates[group_2] / group_positive_rates[group_1]
        fairness_results_df['Disparate Impact'] = disparate_impact
    else:
        fairness_results_df['Disparate Impact'] = None

    print("Métriques de fairness par groupe :")
    print(fairness_results_df)
    
    return fairness_results_df

def fairness_metrics_comparison_multiple_models(csv_paths: list, labels_col: str = 'labels', preds_col: str = 'preds'):
    """
    Comparer les métriques de fairness de plusieurs modèles à partir de fichiers CSV.

    :param csv_paths: Liste des chemins vers les fichiers CSV des prédictions
    :param labels_col: Nom de la colonne des vraies étiquettes (labels)
    :param preds_col: Nom de la colonne des prédictions
    :return: Dictionnaire avec les résultats des métriques
    """
    fairness_results = {}

    # Charger les CSVs et calculer les métriques de fairness pour chaque modèle
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        model_name = csv_path.split('/')[-1].split('.')[0]  # Extraire le nom du modèle depuis le chemin du fichier

        # Calcul des métriques globales
        tn, fp, fn, tp = confusion_matrix(df[labels_col], df[preds_col]).ravel()

        # Calcul des métriques
        accuracy = accuracy_score(df[labels_col], df[preds_col])
        tpr = tp / (tp + fn)  # True Positive Rate (Sensibilité)
        fpr = fp / (fp + tn)  # False Positive Rate
        ppv = tp / (tp + fp)  # Positive Predictive Value (Précision)

        # Ajouter les résultats dans le dictionnaire pour chaque modèle
        fairness_results[model_name] = {
            'Accuracy': accuracy,
            'TPR': tpr,
            'FPR': fpr,
            'PPV': ppv
        }
    
    # Créer une liste des résultats dans un format adapté pour le DataFrame
    results_df_list = []

    for model_name, metrics in fairness_results.items():
        for metric, value in metrics.items():
            results_df_list.append({
                'Model': model_name,
                'Metric': metric,
                'Value': value
            })
    
    # Convertir en DataFrame
    results_df = pd.DataFrame(results_df_list)

    # Visualisation des résultats avec Plotly Express
    fig = px.line(results_df, 
                  x='Model', y='Value', color='Metric', 
                  line_group='Metric', markers=True,
                  title="Comparaison des Modèles",
                  labels={"Model": "Modèle", "Value": "Score", "Metric": "Métrique"})

    fig.show()

    return fairness_results

def fairs(df, preds_col, labels_col, sensitive_col):
    """
    Calcule les métriques de fairness, accuracy, balanced accuracy,
    statistical parity, puis affiche les graphiques pour la comparaison des groupes.
    """
    # Calcul de l'accuracy globale
    accuracy = accuracy_score(df[labels_col], df[preds_col])
    print(f"Accuracy: {accuracy}")

    # Calcul de la balanced accuracy
    balanced_accuracy = balanced_accuracy_score(df[labels_col], df[preds_col])
    print(f"Balanced Accuracy: {balanced_accuracy}")

    # Calcul de l'accuracy par groupe sensible (par exemple, par genre)
    group_accuracy = df.groupby(sensitive_col).apply(
        lambda group: accuracy_score(group[labels_col], group[preds_col])
    )
    print(f"Accuracy par groupe sensible : \n{group_accuracy}")

    # Calcul de la Balanced Accuracy par groupe sensible
    group_balanced_accuracy = df.groupby(sensitive_col).apply(
        lambda group: balanced_accuracy_score(group[labels_col], group[preds_col])
    )
    print(f"Balanced Accuracy par groupe sensible : \n{group_balanced_accuracy}")

    # Calcul de la matrice de confusion pour chaque groupe sensible
    print("\nMatrices de confusion pour chaque groupe sensible :")
    for group in df[sensitive_col].unique():
        group_df = df[df[sensitive_col] == group]
        cm = confusion_matrix(group_df[labels_col], group_df[preds_col])
        print(f"Confusion matrix pour le groupe {group}: \n{cm}")

    # Disparate Impact : Ratio des taux de classification positive par groupe
    print("\nCalcul du Disparate Impact :")
    group_positive_rate = df.groupby(sensitive_col).apply(
        lambda group: np.mean(group[preds_col] == 'malade')  # Calcul du taux positif (ici 'malade')
    )
    print(f"Taux de classification positive par groupe : \n{group_positive_rate}")

    # Calcul du Disparate Impact
    reference_group_rate = group_positive_rate.iloc[0]  # Utiliser le premier groupe comme référence
    disparate_impact = group_positive_rate / reference_group_rate
    print(f"Disparate Impact : \n{disparate_impact}")

    # Calcul de la Statistical Parity
    print("\nCalcul de la Statistical Parity :")
    statistical_parity = group_positive_rate / group_positive_rate.mean()
    print(f"Statistical Parity : \n{statistical_parity}")

    # Affichage des graphiques avec Plotly
    # Accuracy par groupe
    fig_accuracy = px.bar(group_accuracy, title="Accuracy par groupe sensible")
    fig_accuracy.update_layout(xaxis_title=sensitive_col, yaxis_title="Accuracy")
    fig_accuracy.show()

    # Balanced Accuracy par groupe
    fig_balanced_accuracy = px.bar(group_balanced_accuracy, title="Balanced Accuracy par groupe sensible")
    fig_balanced_accuracy.update_layout(xaxis_title=sensitive_col, yaxis_title="Balanced Accuracy")
    fig_balanced_accuracy.show()

    # Disparate Impact
    fig_disparate_impact = px.bar(disparate_impact, title="Disparate Impact par groupe sensible")
    fig_disparate_impact.update_layout(xaxis_title=sensitive_col, yaxis_title="Disparate Impact")
    fig_disparate_impact.show()

    # Statistical Parity
    fig_statistical_parity = px.bar(statistical_parity, title="Statistical Parity par groupe sensible")
    fig_statistical_parity.update_layout(xaxis_title=sensitive_col, yaxis_title="Statistical Parity")
    fig_statistical_parity.show()