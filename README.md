

# 🎮 Wellplayed.lol — Moteur d'Analyse Tactique League of Legends

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?style=flat&logo=pandas)](https://pandas.pydata.org)
[![Riot API](https://img.shields.io/badge/API-Riot_Games-D4AF37?style=flat)](https://developer.riotgames.com/)
[![Status](https://img.shields.io/badge/Status-Production-3FB950?style=flat)]()

> Moteur d'analyse de parties League of Legends développé pour **[Wellplayed.lol](https://wellplayed.lol)**.  
> Transforme les données brutes de l'API Riot Games en rapports tactiques détaillés pour le coaching.

---

## 🎯 Contexte

**Wellplayed.lol** est une application d'analyse et de coaching pour joueurs de League of Legends.  
Ce script constitue le cœur analytique du projet : il ingère les données JSON d'une partie et produit **9 modules d'analyse** couvrant tous les aspects du jeu.

> ⚠️ *Le code complet n'est pas partagé pour des raisons de confidentialité. Ce dépôt contient une partie représentative du moteur.*

---

## ⚙️ Architecture du moteur

```
API Riot Games (JSON)
        │
        ▼
┌─────────────────────┐
│   load_data()       │  Ingestion & validation du JSON
│   get_participant   │  Mapping joueurs / rôles / équipes
│   _info()           │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   9 MODULES D'ANALYSE                   │
├─────────────────────────────────────────────────────────┤
│  1. analyze_jungle_complete()   Jungle pathing & AFK    │
│  2. analyze_shop_timings()      Bad backs & power spikes│
│  3. analyze_advanced_split()    Stats combat & laning   │
│  4. analyze_solo_kills()        Snowball vs outplays     │
│  5. analyze_ratios_kda()        Rentabilité par KDA     │
│  6. analyze_spells_impact()     Impact sorts & ulti     │
│  7. analyze_synergies()         Botlane & Jgl/Support   │
│  8. analyze_tactical()          Splitpush, bias, vision │
│  9. analyze_coaching_metrics()  KPIs coaching @14min    │
└─────────────────────────────────────────────────────────┘
         │
         ▼
   DataFrames Pandas  →  Rapport console / Export
```

---

## 🧩 Détail des modules

### 1. 🌲 Jungle Pathing & AFK Detection
**Problème résolu :** Détecter si un jungler "farm dans son coin" pendant que ses alliés meurent.

**Approche technique :**
- Coordonnées GPS de tous les camps jungle des 2 équipes encodées manuellement (système de coordonnées Riot : `0 → 14820`)
- Symétrie de la map exploitée : `camps_t100[camp] = MAP_MAX - camps_t200[camp]`
- Pour chaque mort alliée, récupération de la **position exacte du jungler** dans la frame timeline correspondante
- Calcul de la **distance euclidienne** entre le jungler et le camp le plus proche
- Seuil `FARMING_RADIUS = 1000` unités pour classifier "en train de farmer"

```python
def get_distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)
```

**Output :** % de morts alliées où le jungler était en train de farmer (indicateur d'AFK farm)

---

### 2. 🛒 Bad Backs & Power Spikes
**Problème résolu :** Identifier les erreurs de timing d'achat (mourir juste après avoir acheté = or perdu) et les pics de puissance (tuer juste après un achat = item efficace).

**Approche technique :**
- Tracking du `last_shop[participantId]` = timestamp du dernier achat
- Fenêtre temporelle configurable (`time_window = 45s` par défaut)
- Croisement des événements `ITEM_PURCHASED` et `CHAMPION_KILL/DEATH` dans la timeline

**Output :** Tableau des "bad backs" et "power spikes" avec timing précis

---

### 3. 📊 Statistiques Avancées & Leads
Calcul de métriques d'avantage contextuel :
- **Différentiel d'or** à 5 minutes après le premier recall
- Séparation **combat** (dégâts, KDA, kills) vs **laning** (CS, vision, wards)

---

### 4. ⚔️ Solo Kills — Snowball vs Outplays
**Problème résolu :** Distinguer un kill "facile" (joueur en avance gold qui tue un joueur pauvre) d'un vrai **outplay** (kill en étant en retard économique).

**Approche technique :**
- Récupération du `currentGold` des deux joueurs au moment exact du kill via la frame timeline
- Classification : `gold_killer > gold_victim + seuil` → Snowball, sinon → Outplay/Comeback

---

### 5. 📐 Ratios d'Efficacité KDA
Calcul de métriques de rentabilité :
- **Or généré par mort** (coût d'une mort pour l'équipe adverse)
- **Dégâts par assist** (utilité du joueur dans les combats sans kill)

---

### 6. 🔮 Impact des Sorts & Structures
- Comptage des sorts utilisés par catégorie (ult, dash, CC)
- Détection des **multikills avec ultime** via croisement timeline
- Destruction de structures avec timing précis (`BUILDING_KILL` events)

---

### 7. 🤝 Synergies & Contexte de Combat
- **Synergie Botlane** : kills/assists partagés entre ADC et Support
- **Synergie Jungle/Support** : roams détectés via kills partagés
- **Catches** : kills en supériorité numérique (détection des alliés proches via distance ≤ 2000 unités)
- **Morts en infériorité** : joueurs "caught" isolés
- **Greed deaths** : morts avec > 2000 gold non dépensé
- **Shutdowns** : primes de kill > 300 gold

```python
# Détection de supériorité numérique via distance géospatiale
for pid, info in pid_map.items():
    dist = math.sqrt((vic_pos['x']-positions[pid]['x'])**2 + 
                     (vic_pos['y']-positions[pid]['y'])**2)
    if dist <= 2000: allies += 1

if attackers > allies:
    # → C'est un "catch" (kill en supériorité)
```

---

### 8. 🗺️ Analyse Tactique Avancée
- **Splitpush isolation** : % du temps passé seul sur une lane après 15 minutes
- **Jungle lane bias** : côté de la map privilégié par chaque jungler (0-14min)
- **Vision setup** : wards placées dans les 2 minutes précédant chaque objectif majeur

---

### 9. 🎓 Métriques de Coaching
- **Efficacité mécanique** : ratio Dégâts/Gold (un ratio élevé = joueur qui "fait plus avec moins")
- **Dominance early game** : dégâts, or, CS et XP comparés à 14 minutes via les données de timeline

---

## 📁 Structure

```
wellplayed-data-analysis/
├── PARTIE_du_projet.py   # Moteur d'analyse (extrait)
├── data_en_plus.py       # Modules complémentaires (extrait)
├── test_data.json        # Données d'exemple (1 partie, API Riot)
└── README.md
```

---

## 🚀 Utilisation

```bash
# Cloner le repo
git clone https://github.com/ZnKlcc/wellplayed-data-analysis.git
cd wellplayed-data-analysis

# Installer les dépendances
pip install pandas numpy

# Lancer l'analyse sur le fichier d'exemple
python PARTIE_du_projet.py
```

**Exemple de sortie :**
```
============================================================
   ANALYSE COMPLÈTE DE LA PARTIE : test_data.json
============================================================

>>> 1. JUNGLE PATHING & FARMING <<<
Jungler   K/D/A   Scuttles  Morts Alliées  Morts pdt Farm  % AFK Farm
  Briar   9/4/11         1             8               2      25.00%

--- Objectifs Neutres ---
Jungler  Temps  Objectif
  Briar   8:14  DRAGON MOUNTAIN_DRAGON
  Briar  22:07  BARON_NASHOR

>>> 4. SOLOKILLS (AVANCE vs RETARD) <<<
--- SNOWBALL (Kills en étant en avance Gold) ---
Tueur      Victime    Gold Tueur  Gold Victime  Avance   Time
 Cross   Mordekaiser       4200          2100   +2100   14m

--- OUTPLAYS / COMEBACK ---
Aucun outplay détecté
```

---

## 🛠️ Stack technique

| Composant | Technologie |
|-----------|-------------|
| Langage | Python 3.10 |
| Manipulation données | Pandas, NumPy |
| Calculs géométriques | `math` (distance euclidienne) |
| Source données | API Riot Games (JSON) |
| Parsing timeline | Frame-by-frame event processing |

---

## 👤 Auteur

**Augustin Maitre** — Étudiant Polytech Nice-Sophia (Maths & Modélisation)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/augustin-maitre-05229a37b/)
[![GitHub](https://img.shields.io/badge/GitHub-ZnKlcc-181717?style=flat&logo=github)](https://github.com/ZnKlcc)

# Analyse de Données - Projet Wellplayed.lol

Ce dépôt contient des extraits du code source utilisé pour le traitement des données des joueurs de League of Legends.

## ⚠️ Note sur la confidentialité
Ce projet a été réalisé en collaboration professionnelle. Pour des raisons de propriété intellectuelle et de confidentialité, **le code complet ne peut pas être partagé publiquement**.

Le fichier `PARTIE_du projet.py` est un échantillon représentatif qui démontre :
* La structure logique du traitement.
* L'utilisation de la librairie **Pandas** pour le nettoyage.
* La syntaxe et les bonnes pratiques utilisées.

## Fonctionnalités principales
L'objectif de ce script était de :
1. Importer les données brutes des parties (JSON/CSV).
2. Nettoyer les valeurs aberrantes.
3. Calculer de nouveaux indicateurs de performance (KPIs) pour le coaching.

## Technologies
* Python 3
* Pandas / NumPy
