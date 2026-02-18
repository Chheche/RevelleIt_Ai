# RevelleIt_Ai

Ce projet analyse des postes de travail (desktops) et calcule un score environnemental.

## Format du CSV
Le fichier `data/list_desktop.csv` doit contenir les colonnes suivantes :

- `Marque`, `Modele`, `Type_Processeur`, `Nb_Coeurs`, `Frequence_Processeur_GHz`,
  `RAM_Go`, `Stockage_Go`, `Type_Stockage`, `Carte_Graphique`, `Consommation_Watt`,
  `Duree_Vie_Moyenne_ans`, `Poids_kg`, `Date_Commercialisation`, **`Departement`**

La colonne **`Departement`** est prise en compte dans le calcul du score : les
matrices de référence (benchmarks) sont calculées par département et chaque
équipement est noté par rapport à la moyenne de son département. Un score
moyen par département est affiché à la fin du traitement.

## Exécution
Lancer le script principal :

```bash
python main.py
```

Le script ingère le CSV, calcule les impacts physiques, effectue des imputs par
KNN si nécessaire, puis calcule et stocke un score final dans la base SQLite.
Une fois terminé, il affiche également le score moyen par département.

## Notebook d'exploration
Le notebook `Test.ipynb` contient quelques requêtes d'exemple, par exemple
pour afficher le top 5 des équipements et le score moyen par département.