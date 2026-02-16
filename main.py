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
session = Session()


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
    """ Entraîne l'IA sur les 6 KPIs et calcule les moyennes de référence """
    logging.info("Démarrage de l'entraînement du modèle KNN...")
    data = pd.read_sql('SELECT * FROM impacts_calcules', con=engine)

    if len(data) < 3:
        logging.warning(f"Données insuffisantes ({len(data)}) pour entraîner le KNN.")
        return None, None

    # Apprendre sur les 6 KPIs
    kpis = ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']
    X = data[['Nb_Coeurs', 'RAM_Go', 'Stockage_Go']]
    y = data[kpis]

    knn = KNeighborsRegressor(n_neighbors=min(3, len(data)))
    knn.fit(X, y)

    stats = data[kpis].agg(['mean', 'min', 'max'])
    return knn, stats


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


def calculer_score_final_complet(impacts, stats_benchmark):
    """ Moyenne des notes 1-5 sur les 6 KPIs """
    notes = []
    for kpi in ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']:
        ratio = impacts[kpi] / stats_benchmark.loc['mean', kpi]
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

    """ Ingestion et Calcul Physique """
    logging.info("Ingestion des données CSV...")
    df_raw = pd.read_csv('data/list_desktop.csv', sep=';', encoding='utf-8')
    for _, row in df_raw.iterrows():
        impacts = calculer_impact_total(row)
        new_entry = DesktopImpact(
            Marque=row['Marque'], Modele=row['Modele'],
            Nb_Coeurs=row['Nb_Coeurs'], RAM_Go=row['RAM_Go'], Stockage_Go=row['Stockage_Go'],
            **impacts
        )
        session.add(new_entry)
    session.commit()
    logging.info(f"Ingestion terminée : {len(df_raw)} lignes traitées.")

    """ IA et Scoring """
    logging.info("Analyse IA et Scoring...")
    knn_model, stats_ref = train_knn_benchmark()

    if stats_ref is not None:
        tous_les_pcs = session.query(DesktopImpact).all()
        for pc in tous_les_pcs:

            impacts_dict = {k: getattr(pc, k) for k in ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']}

            impacts_dict = completer_impacts_par_knn(impacts_dict, pc, knn_model)

            for k, v in impacts_dict.items(): setattr(pc, k, v)
            pc.Score_Final = calculer_score_final_complet(impacts_dict, stats_ref)
            logging.info(f"Score calculé pour {pc.Marque} {pc.Modele} : {pc.Score_Final}/5")

        session.commit()
        logging.info("Scoring terminé avec succès.")
        print("Audit réussi : Ingestion, Imputation IA et Scoring terminés.")
    else:
        logging.warning("Audit partiel : Impossible de générer le benchmark (données insuffisantes).")
        print("Ingestion finie, mais pas assez de données pour le benchmark/IA.")