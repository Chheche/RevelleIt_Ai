import pandas as pd
import logging
import os
from sqlalchemy.orm import sessionmaker
from sklearn.neighbors import KNeighborsRegressor
from coefficient import coefficient
from models import Base, DesktopImpact
from sqlalchemy import create_engine

if not os.path.exists('logs'): os.makedirs('logs')

logging.basicConfig(
    filename='logs/revelleit.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode="a"
)

engine = create_engine('sqlite:///revelleit.db')
Session = sessionmaker(bind=engine)
# session will be instantiated after schema adjustments inside __main__
session = None


def calculer_impact_total(row):
    """ Calcule les 6 indicateurs pour tous les composants + usage """
    res = {}
    st_type = 'ssd' if 'ssd' in str(row['Type_Stockage']).lower() else 'hdd'
    gpu_type = 'graphique_dediee_pro' if any(
        x in str(row['Carte_Graphique']).lower() for x in ['rtx', 'quadro', 'pro']) else 'graphique_integree'

    for kpi in ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']:
        try:
            impact_fab = (
                    (row['Nb_Coeurs'] * coefficient['cpu_desktop'][kpi]) +
                    (row['RAM_Go'] * coefficient['ram'][kpi]) +
                    (row['Stockage_Go'] * coefficient[st_type][kpi]) +
                    coefficient[gpu_type][kpi] +
                    coefficient['chassis'][kpi]
            )
            impact_usage = 0
            kwh_total = (row['Consommation_Watt'] / 1000) * coefficient['usage']['heures_an'] * row[
                'Duree_Vie_Moyenne_ans']
            if kpi == 'GWP':
                impact_usage = kwh_total * coefficient['usage']['facteur_GWP']
            elif kpi == 'TPE':
                impact_usage = kwh_total * coefficient['usage']['facteur_TPE']

            res[kpi] = impact_fab + impact_usage
        except:
            logging.error(f"Erreur de calcul KPI {kpi} pour {row['Modele']} : {e}")
            res[kpi] = None
    return res


def train_knn_benchmark():
    """ Entraîne l'IA sur les 6 KPIs et calcule les moyennes de référence.

    Le benchmark est construit globalement et par département :
    - stats_global : moyenne/min/max pour l'ensemble des PC
    - stats_dept   : moyenne pour chaque département (index = département, colonnes = kpis)
    """
    logging.info("Démarrage de l'entraînement du modèle KNN...")
    data = pd.read_sql('SELECT * FROM impacts_calcules', con=engine)

    if len(data) < 3:
        logging.warning(f"Données insuffisantes ({len(data)}) pour entraîner le KNN.")
        return None, None, None

    # Apprendre sur les 6 KPIs
    kpis = ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']
    X = data[['Nb_Coeurs', 'RAM_Go', 'Stockage_Go']]
    y = data[kpis]

    knn = KNeighborsRegressor(n_neighbors=min(3, len(data)))
    knn.fit(X, y)

    stats_global = data[kpis].agg(['mean', 'min', 'max'])
    # moyenne par département (sert pour scorer par secteur)
    stats_dept = data.groupby('Departement')[kpis].mean()
    return knn, stats_global, stats_dept


def completer_impacts_par_knn(impacts, row, knn_model):
    """ Impute uniquement les valeurs manquantes (NaN/None) """
    kpis = ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']
    manquants = [k for k in kpis if impacts.get(k) is None or pd.isna(impacts.get(k))]

    if manquants and knn_model:
        logging.info(f"Valeurs manquantes détectées pour {row.Modele} ({manquants}). Appel au KNN...")
        X_input = pd.DataFrame([[row['Nb_Coeurs'], row['RAM_Go'], row['Stockage_Go']]],
                               columns=['Nb_Coeurs', 'RAM_Go', 'Stockage_Go'])
        preds = knn_model.predict(X_input)[0]
        preds_dict = dict(zip(kpis, preds))
        for k in manquants:
            impacts[k] = preds_dict[k]
            logging.info(f"Imputation réussie pour {k} : {preds_dict[k]:.4f}")
    return impacts


def calculer_score_final_complet(impacts, stats_benchmark_mean):
    """Moyenne des notes 1-5 sur les 6 KPIs using a benchmark mean row.

    stats_benchmark_mean doit être une série contenant la ligne 'mean' ou
    directement les valeurs moyennes par KPI (selon utilisation).
    """
    notes = []
    for kpi in ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']:
        mean_val = stats_benchmark_mean[kpi]
        # éviter division par zéro
        if mean_val <= 0 or pd.isna(mean_val):
            ratio = float('inf')
        else:
            ratio = impacts[kpi] / mean_val
        if ratio <= 0.7:
            note = 5
        elif ratio <= 0.9:
            note = 4
        elif ratio <= 1.1:
            note = 3
        elif ratio <= 1.4:
            note = 2
        else:
            note = 1
        notes.append(note)
    return round(sum(notes) / len(notes), 1)


if __name__ == '__main__':
    logging.info("--- LANCEMENT DU SCRIPT ---")
    Base.metadata.create_all(engine)

    # s'assurer que la colonne Departement existe (si base déjà créée auparavant)
    with engine.begin() as conn_check:  # begin() ouvre transaction et commit automatiquement
        # vérifier la structure actuelle de la table
        try:
            result = conn_check.execute("PRAGMA table_info(impacts_calcules)")
            cols = [row[1] for row in result.fetchall()]
        except Exception:
            cols = []
        if 'Departement' not in cols:
            try:
                conn_check.execute('ALTER TABLE impacts_calcules ADD COLUMN Departement TEXT')
                logging.info("Colonne Departement ajoutée à la base.")
            except Exception as e:
                logging.error(f"Impossible d'ajouter la colonne Departement : {e}")

    # créer la session après modification de la table pour que l'ORM soit synchronisé
    session = Session()

    """ Ingestion et Calcul Physique """
    logging.info("Ingestion des données CSV...")
    df_raw = pd.read_csv('data/list_desktop.csv', sep=';', encoding='utf-8')
    for _, row in df_raw.iterrows():
        impacts = calculer_impact_total(row)
        new_entry = DesktopImpact(
            Marque=row['Marque'], Modele=row['Modele'],
            Nb_Coeurs=row['Nb_Coeurs'], RAM_Go=row['RAM_Go'], Stockage_Go=row['Stockage_Go'],
            Departement=row.get('Departement', None),
            **impacts
        )
        session.add(new_entry)
    session.commit()
    logging.info(f"Ingestion terminée : {len(df_raw)} lignes traitées.")
    """ IA et Scoring """
    logging.info("Analyse IA et Scoring...")
    knn_model, stats_global, stats_dept = train_knn_benchmark()

    if stats_global is not None:
        tous_les_pcs = session.query(DesktopImpact).all()
        for pc in tous_les_pcs:

            impacts_dict = {k: getattr(pc, k) for k in ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']}

            impacts_dict = completer_impacts_par_knn(impacts_dict, pc, knn_model)

            for k, v in impacts_dict.items(): setattr(pc, k, v)
            # choisir les statistiques selon le département si disponible
            if pc.Departement and pc.Departement in stats_dept.index:
                bench_means = stats_dept.loc[pc.Departement]
            else:
                bench_means = stats_global.loc['mean']
            pc.Score_Final = calculer_score_final_complet(impacts_dict, bench_means)
            logging.info(f"Score calculé pour {pc.Marque} {pc.Modele} (dept {pc.Departement}) : {pc.Score_Final}/5")

        session.commit()
        
        # Afficher tous les desktops classés par score
        print("\n" + "="*120)
        print("TABLEAU DE TOUS LES ÉQUIPEMENTS CLASSÉS PAR SCORE")
        print("="*120)
        df_desktops = pd.read_sql("""
            SELECT Marque, Modele, Departement, Score_Final, GWP, ADPe, WU, ADPf, TPE, WEEE
            FROM impacts_calcules
            ORDER BY Score_Final DESC
        """, con=engine)
        
        # Formatage pour meilleure lisibilité
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df_desktops.to_string(index=False))
        
        # afficher score moyen par département
        print("\n" + "="*120)
        print("SCORE MOYEN PAR DÉPARTEMENT")
        print("="*120)
        df_scores = pd.read_sql("""
            SELECT Departement, AVG(Score_Final) as Score_Moyen, COUNT(*) as Nb_Equipements
            FROM impacts_calcules
            GROUP BY Departement
            ORDER BY Score_Moyen DESC
        """, con=engine)
        print(df_scores.to_string(index=False))
        print("="*120)
        
        logging.info("Scoring terminé avec succès.")
        print("\nAudit réussi : Ingestion, Imputation IA et Scoring terminés.")
    else:
        logging.warning("Audit partiel : Impossible de générer le benchmark (données insuffisantes).")
        print("Ingestion finie, mais pas assez de données pour le benchmark/IA.")