import pandas as pd
import logging
import os
from sqlalchemy.orm import sessionmaker
from sklearn.neighbors import KNeighborsRegressor
from coefficient import coefficient
from models import Base, DesktopImpact
from sqlalchemy import create_engine, text

if not os.path.exists('logs'): os.makedirs('logs')

logging.basicConfig(
    filename='logs/revelleit.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    filemode="a"
)

# use a database file located next to this script regardless of cwd
base_dir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(base_dir, 'revelleit.db')
engine = create_engine(f'sqlite:///{db_path}')
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
    # lecture explicite dans l'ordre voulu : marque, modele, departement, scores, caractéristiques
    data = pd.read_sql(
        '''SELECT Marque, Modele, Departement, Green_IT_Score,
                  Score_GWP, Score_ADPe, Score_WU, Score_ADPf, Score_TPE, Score_WEEE,
                  Nb_Coeurs, RAM_Go, Stockage_Go,
                  GWP, ADPe, WU, ADPf, TPE, WEEE
           FROM impacts_calcules''',
        con=engine)

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


def calculer_scores(impacts, stats_benchmark_mean):
    """Calcule la note globale et les notes individuelles par KPI (1-5).

    Chaque KPI est noté indépendamment sur l'échelle 1‑5 en comparant la valeur
    d'impact à la moyenne de référence. Le score global est simplement la moyenne
    de ces notes (équivalent de l'ancien Score_Final).

    Retourne un tuple ``(score_global, scores_par_kpi)`` où
    ``scores_par_kpi`` est un dictionnaire contenant une note 1‑5 pour chaque KPI
    et ``score_global`` est la moyenne de ces notes.
    """
    kpis = ['GWP', 'ADPe', 'WU', 'ADPf', 'TPE', 'WEEE']
    scores = {}
    notes = []

    for kpi in kpis:
        mean_val = stats_benchmark_mean[kpi]
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
        scores[kpi] = note
        notes.append(note)

    score_global = round(sum(notes) / len(notes), 1)
    return score_global, scores


if __name__ == '__main__':
    logging.info("--- LANCEMENT DU SCRIPT ---")
    Base.metadata.create_all(engine)

    # vérifier et adapter la structure de la table si nécessaire
    with engine.begin() as conn_check:  # begin() ouvre transaction et commit automatiquement
        try:
            result = conn_check.execute("PRAGMA table_info(impacts_calcules)")
            cols = [row[1] for row in result.fetchall()]
        except Exception:
            cols = []

        def add_column(name, sql_type):
            if name not in cols:
                try:
                    # use text() to avoid SQLAlchemy 2.0 errors about non-executable
                    conn_check.execute(text(f'ALTER TABLE impacts_calcules ADD COLUMN {name} {sql_type}'))
                    logging.info(f"Colonne {name} ajoutée à la base.")
                except Exception as e:
                    logging.error(f"Impossible d'ajouter la colonne {name} : {e}")

        # colonnes techniques et score
        add_column('Departement', 'TEXT')
        add_column('Green_IT_Score', 'FLOAT')
        add_column('Score_GWP', 'FLOAT')
        add_column('Score_ADPe', 'FLOAT')
        add_column('Score_WU', 'FLOAT')
        add_column('Score_ADPf', 'FLOAT')
        add_column('Score_TPE', 'FLOAT')
        add_column('Score_WEEE', 'FLOAT')

        # copier les anciennes valeurs Score_Final dans Green_IT_Score
        if 'Score_Final' in cols:
            try:
                conn_check.execute(text("UPDATE impacts_calcules SET Green_IT_Score = Score_Final"))
                logging.info("Valeurs de Score_Final copiées dans Green_IT_Score")
            except Exception as e:
                logging.error(f"Erreur migration Score_Final -> Green_IT_Score: {e}")

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

            overall, kpi_scores = calculer_scores(impacts_dict, bench_means)
            pc.Green_IT_Score = overall
            # chaque KPI a maintenant une note indépendante sur 5
            pc.Score_GWP = kpi_scores['GWP']
            pc.Score_ADPe = kpi_scores['ADPe']
            pc.Score_WU = kpi_scores['WU']
            pc.Score_ADPf = kpi_scores['ADPf']
            pc.Score_TPE = kpi_scores['TPE']
            pc.Score_WEEE = kpi_scores['WEEE']
            logging.info(
                f"Score global {overall}/5 calculé pour {pc.Marque} {pc.Modele} (dept {pc.Departement}), "
                f"KPI bruts {kpi_scores}")

        session.commit()
        
        # Afficher tous les desktops classés par score
        print("\n" + "="*120)
        print("TABLEAU DE TOUS LES ÉQUIPEMENTS CLASSÉS PAR SCORE")
        print("="*120)
        df_desktops = pd.read_sql("""
            SELECT Marque, Modele, Departement, Green_IT_Score, Score_GWP, Score_ADPe,
                   Score_WU, Score_ADPf, Score_TPE, Score_WEEE,
                   GWP, ADPe, WU, ADPf, TPE, WEEE
            FROM impacts_calcules
            ORDER BY Green_IT_Score DESC
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
            SELECT Departement, AVG(Green_IT_Score) as Score_Moyen, COUNT(*) as Nb_Equipements
            FROM impacts_calcules
            GROUP BY Departement
            ORDER BY Score_Moyen DESC
        """, con=engine)
        print(df_scores.to_string(index=False))
        print("="*120)
        
        # ajouter les scores KPI moyens par département
        print("\n" + "="*120)
        print("SCORES KPI MOYENS PAR DÉPARTEMENT")
        print("="*120)
        df_dept_kpi = pd.read_sql("""
            SELECT Departement,
                   AVG(Score_GWP) AS Avg_Score_GWP,
                   AVG(Score_ADPe) AS Avg_Score_ADPe,
                   AVG(Score_WU)  AS Avg_Score_WU,
                   AVG(Score_ADPf) AS Avg_Score_ADPf,
                   AVG(Score_TPE) AS Avg_Score_TPE,
                   AVG(Score_WEEE) AS Avg_Score_WEEE
            FROM impacts_calcules
            GROUP BY Departement
        """, con=engine)
        print(df_dept_kpi.to_string(index=False))
        print("="*120)

        # statistiques brutes par département pour repérer les plus gros consommateurs
        print("\n" + "="*120)
        print("STATISTIQUES BRUTES PAR DÉPARTEMENT")
        print("="*120)
        df_dept_raw = pd.read_sql("""
            SELECT Departement,
                   AVG(GWP) AS Avg_GWP, AVG(TPE) AS Avg_TPE,
                   MAX(GWP) AS Max_GWP, MAX(TPE) AS Max_TPE
            FROM impacts_calcules
            GROUP BY Departement
        """, con=engine)
        print(df_dept_raw.to_string(index=False))
        print("="*120)
        
        # moyenne des notes KPI globales – permet d'identifier le ou les indicateurs
        # les plus pénalisants sur l'ensemble du jeu de données
        print("\n" + "="*120)
        print("SCORE MOYEN GLOBAL PAR KPI")
        print("="*120)
        df_kpi_scores = pd.read_sql("""
            SELECT AVG(Score_GWP) as Moy_GWP, AVG(Score_ADPe) as Moy_ADPe,
                   AVG(Score_WU) as Moy_WU, AVG(Score_ADPf) as Moy_ADPf,
                   AVG(Score_TPE) as Moy_TPE, AVG(Score_WEEE) as Moy_WEEE
            FROM impacts_calcules
        """, con=engine)
        print(df_kpi_scores.to_string(index=False))
        print("="*120)

        logging.info("Scoring terminé avec succès.")
        print("\nAudit réussi : Ingestion, Imputation IA et Scoring terminés.")
    else:
        logging.warning("Audit partiel : Impossible de générer le benchmark (données insuffisantes).")
        print("Ingestion finie, mais pas assez de données pour le benchmark/IA.")