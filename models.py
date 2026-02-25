from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class DesktopImpact(Base):
    __tablename__ = 'impacts_calcules'

    id = Column(Integer, primary_key=True, autoincrement=True)

    Marque = Column(String)
    Modele = Column(String)

    # colonne département (déplacée pour correspondre à l'affichage)
    Departement = Column(String)

    # Colonne pour le scoring global (Note sur 5)
    # anciennement appelée Score_Final, renommée Green_IT_Score
    Green_IT_Score = Column(Float)

    # Scores individuels par KPI (permet d'identifier le KPI dominant)
    Score_GWP = Column(Float)
    Score_ADPe = Column(Float)
    Score_WU = Column(Float)
    Score_ADPf = Column(Float)
    Score_TPE = Column(Float)
    Score_WEEE = Column(Float)

    # Caractéristiques techniques
    Nb_Coeurs = Column(Integer)
    RAM_Go = Column(Integer)
    Stockage_Go = Column(Integer)

    # Les 6 KPIs (Indicateurs d'impact environnemental)
    GWP = Column(Float)  # Global Warming Potential (Carbone)
    ADPe = Column(Float)  # Abiotic Depletion Potential elements (Minéraux)
    WU = Column(Float)  # Water Use (Eau)
    ADPf = Column(Float)  # Abiotic Depletion Potential fossil (Énergie fossile)
    TPE = Column(Float)  # Total Primary Energy (Énergie primaire)
    WEEE = Column(Float)  # Waste Electrical and Electronic Equipment (Déchets)