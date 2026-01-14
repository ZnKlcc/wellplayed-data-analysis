import json
import pandas as pd
import math
import numpy as np

# Configuration Pandas pour un affichage propre dans la console
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# =============================================================================
#                               UTILITAIRES
# =============================================================================

def get_distance(p1, p2):
    """Calcule la distance euclidienne entre deux points (x, y)."""
    return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2)

def load_data(file_path):
    """Charge le fichier JSON en gérant les erreurs."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur fatale lors du chargement du fichier : {e}")
        return None

def get_participant_info(participants):
    """Prépare les informations sur les participants (noms, rôles, équipes)."""
    teams = {100: {'ADC': None, 'SUP': None, 'JGL': None}, 200: {'ADC': None, 'SUP': None, 'JGL': None}}
    pid_map = {}
    for p in participants:
        pid = p['participantId']
        tid = p['teamId']
        role = p.get('individualPosition', 'UNKNOWN')
        name = p['riotIdGameName']
        pid_map[pid] = {'name': name, 'team': tid, 'role': role, 'champion': p['championName']}

        if role == 'BOTTOM':
            teams[tid]['ADC'] = pid
        elif role == 'UTILITY':
            teams[tid]['SUP'] = pid
        elif role == 'JUNGLE':
            teams[tid]['JGL'] = pid
    return pid_map, teams

# =============================================================================
#                           1. ANALYSE JUNGLE COMPLETE
# =============================================================================

def analyze_jungle_complete(data):
    # Config
    MAP_MAX = 14820
    FARMING_RADIUS = 1000
    camps_t200 = {
        'Blue Buff': {'x': 11000, 'y': 6900}, 'Red Buff': {'x': 7100, 'y': 10900},
        'Gromp': {'x': 12700, 'y': 6400}, 'Loups': {'x': 11000, 'y': 8400},
        'Corbins': {'x': 7800, 'y': 9400}, 'Krugs': {'x': 6400, 'y': 12200}
    }
    camps_t100 = {name: {'x': MAP_MAX - pos['x'], 'y': MAP_MAX - pos['y']} for name, pos in camps_t200.items()}
    all_camps = {**{f"T100 {k}": v for k,v in camps_t100.items()}, **{f"T200 {k}": v for k,v in camps_t200.items()}}

    participants = data['match']['info']['participants']
    junglers = {}
    participant_teams = {}
    
    for p in participants:
        participant_teams[p['participantId']] = p['teamId']
        if p.get('individualPosition') == 'JUNGLE':
            stats_sup = {
                'kills': p.get('kills', 0), 'deaths': p.get('deaths', 0), 'assists': p.get('assists', 0),
                'scuttles': p.get('challenges', {}).get('scuttleCrabKills', 0)
            }
            junglers[p['participantId']] = {
                'name': p['riotIdGameName'], 'teamId': p['teamId'],
                'stats': {'total_ally_deaths': 0, 'deaths_while_farming': 0},
                'extra_stats': stats_sup
            }

    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    
    detailed_events = []
    objectives_list = []

    for frame in frames:
        for event in frame['events']:
            # Timeline Objectifs Elite
            if event.get('type') == 'ELITE_MONSTER_KILL':
                killer_id = event.get('killerId')
                if killer_id in junglers:
                    killer = junglers[killer_id]
                    obj_name = f"{event.get('monsterType')} {event.get('monsterSubType', '')}".strip()
                    objectives_list.append({
                        'Jungler': killer['name'],
                        'Temps': f"{int(event['timestamp']/1000//60)}:{int(event['timestamp']/1000%60):02d}",
                        'Objectif': obj_name,
                        'Position': f"({event['position']['x']}, {event['position']['y']})"
                    })

            # Analyse Morts Alliées (AFK Farm)
            if event.get('type') == 'CHAMPION_KILL':
                victim_id = event.get('victimId')
                killer_id = event.get('killerId')
                assists = event.get('assistingParticipantIds', [])
                victim_team = participant_teams.get(victim_id)
                
                for jgl_id, jgl_info in junglers.items():
                    # Si c'est un allié du jungler qui meurt (et pas le jungler lui-même)
                    if jgl_info['teamId'] == victim_team and jgl_id != victim_id:
                        jgl_info['stats']['total_ally_deaths'] += 1
                        # Le jungler a-t-il participé ?
                        if (killer_id == jgl_id) or (jgl_id in assists):
                            continue
                        
                        death_time = event['timestamp']
                        frame_idx = min(death_time // 60000, len(frames) - 1)
                        try:
                            jgl_pos = frames[frame_idx]['participantFrames'][str(jgl_id)]['position']
                        except KeyError:
                            continue 
                            
                        nearest_camp_name = "Aucun"
                        min_dist = float('inf')
                        for c_name, c_pos in all_camps.items():
                            d = get_distance(jgl_pos, c_pos)
                            if d < min_dist:
                                min_dist = d
                                nearest_camp_name = c_name
                        
                        is_farming = min_dist <= FARMING_RADIUS
                        if is_farming:
                            jgl_info['stats']['deaths_while_farming'] += 1
                        
                        detailed_events.append({
                            'Jungler': jgl_info['name'],
                            'Temps Mort': f"{int(death_time/1000//60)}:{int(death_time/1000%60):02d}",
                            'Position JGL': f"({jgl_pos['x']}, {jgl_pos['y']})",
                            'Camp Proche': nearest_camp_name,
                            'Dist': int(min_dist),
                            'Farming?': "OUI" if is_farming else "NON"
                        })

    df_details = pd.DataFrame(detailed_events)
    df_objectives = pd.DataFrame(objectives_list)
    if not df_objectives.empty: df_objectives.sort_values(by='Temps', inplace=True)
    
    summary_data = []
    for jid, info in junglers.items():
        total = info['stats']['total_ally_deaths']
        farming = info['stats']['deaths_while_farming']
        percentage = (farming / total * 100) if total > 0 else 0.0
        ext = info['extra_stats']
        summary_data.append({
            'Jungler': info['name'],
            'K/D/A': f"{ext['kills']}/{ext['deaths']}/{ext['assists']}",
            'Scuttles': ext['scuttles'],
            'Morts Alliées': total,
            'Morts pdt Farm': farming,
            '% AFK Farm': f"{percentage:.2f}%"
        })
    
    return pd.DataFrame(summary_data), df_objectives, df_details

# =============================================================================
# 2. ANALYSE TIMING & BOUTIQUE (Bad Backs & Power Spikes)
# =============================================================================

def analyze_shop_timings(data, time_window=45):
    participants = data['match']['info']['participants']
    players = {p['participantId']: {'name': p['riotIdGameName'], 'role': p.get('individualPosition'), 
                                    'deaths': 0, 'bad_backs': 0, 'shop_death_details': [],
                                    'kills': 0, 'shop_kills': 0, 'shop_kill_details': []} for p in participants}
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    last_shops = {}

    for frame in frames:
        for event in frame['events']:
            if event.get('type') == 'ITEM_PURCHASED':
                last_shops[event['participantId']] = event['timestamp']
            
            if event.get('type') == 'CHAMPION_KILL':
                # Bad Backs
                vic = event.get('victimId')
                if vic in players:
                    players[vic]['deaths'] += 1
                    last = last_shops.get(vic)
                    if last and (event['timestamp'] - last) / 1000 <= time_window:
                        players[vic]['bad_backs'] += 1
                        players[vic]['shop_death_details'].append(f"{int((event['timestamp'] - last)/1000)}s")
                
                # Power Spikes
                killer = event.get('killerId')
                if killer in players:
                    players[killer]['kills'] += 1
                    last = last_shops.get(killer)
                    if last and (event['timestamp'] - last) / 1000 <= time_window:
                        players[killer]['shop_kills'] += 1

    bad_backs_res = []
    for p in players.values():
        pct = (p['bad_backs'] / p['deaths'] * 100) if p['deaths'] > 0 else 0
        bad_backs_res.append({
            'Joueur': p['name'], 'Role': p['role'], 'Morts Tot': p['deaths'], 
            'Bad Backs': p['bad_backs'], '% Bad Backs': f"{pct:.1f}%", 'Détails': ",".join(p['shop_death_details'])
        })
        
    power_spikes_res = []
    for p in players.values():
        pct = (p['shop_kills'] / p['kills'] * 100) if p['kills'] > 0 else 0
        power_spikes_res.append({
            'Joueur': p['name'], 'Role': p['role'], 'Kills Tot': p['kills'],
            'Kills Post-Shop': p['shop_kills'], '% Power Spike': f"{pct:.1f}%"
        })

    return pd.DataFrame(bad_backs_res), pd.DataFrame(power_spikes_res)

def analyze_gold_diff_5min(data):
    participants = data['match']['info']['participants']
    pid_map = {p['participantId']: {'name': p['riotIdGameName'], 'role': p.get('individualPosition'), 'team': p['teamId']} for p in participants}
    
    opponents = {}
    for pid, info in pid_map.items():
        for oid, oinfo in pid_map.items():
            if info['team'] != oinfo['team'] and info['role'] == oinfo['role']:
                opponents[pid] = oid; break
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    first_resets = {}

    for frame in frames:
        for event in frame['events']:
            if event['type'] == 'ITEM_PURCHASED' and event['timestamp'] > 60000:
                if event['participantId'] not in first_resets:
                    first_resets[event['participantId']] = event['timestamp']

    def get_gold(pid, time_ms):
        idx = int(time_ms // 60000)
        if idx >= len(frames): idx = len(frames) - 1
        return frames[idx]['participantFrames'][str(pid)]['totalGold']

    res = []
    for pid, info in pid_map.items():
        back = first_resets.get(pid)
        opp = opponents.get(pid)
        
        if back and opp:
            g_start = get_gold(pid, back) - get_gold(opp, back)
            g_end = get_gold(pid, back + 300000) - get_gold(opp, back + 300000)
            change = g_end - g_start
            res.append({
                'Joueur': info['name'], 'Role': info['role'], 
                'First Back': f"{int(back/60000)}m", 
                'Gold Diff Start': g_start, 
                'Gold Diff +5min': g_end, 
                'Variation': change
            })
        else:
            res.append({'Joueur': info['name'], 'Role': info['role'], 'First Back': '-', 'Gold Diff Start': 'N/A', 'Gold Diff +5min': 'N/A', 'Variation': 'N/A'})
    return pd.DataFrame(res)

# =============================================================================
# 3. STATS AVANCÉES & LEADS
# =============================================================================

def analyze_advanced_split(data):
    info = data['match']['info']
    duration = info['gameDuration'] / 60
    participants = info['participants']
    
    team_totals = {100: {'dmg':0,'gold':0,'kills':0}, 200: {'dmg':0,'gold':0,'kills':0}}
    for p in participants:
        tid = p['teamId']
        team_totals[tid]['dmg'] += p['totalDamageDealtToChampions']
        team_totals[tid]['gold'] += p['goldEarned']
        team_totals[tid]['kills'] += p['kills']

    pid_map = {p['participantId']: p for p in participants}
    opponents = {}
    for pid, p in pid_map.items():
        for oid, op in pid_map.items():
            if p['teamId'] != op['teamId'] and p['individualPosition'] == op['individualPosition']:
                opponents[pid] = oid; break
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    
    fb_parts = set()
    first_resets = {}
    for frame in frames:
        for event in frame['events']:
            if event['type'] == 'CHAMPION_SPECIAL_KILL' and event.get('killType') == 'KILL_FIRST_BLOOD':
                for sub in frame['events']:
                    if sub.get('type') == 'CHAMPION_KILL' and sub.get('timestamp') == event['timestamp']:
                        fb_parts.add(sub['killerId'])
                        for a in sub.get('assistingParticipantIds', []): fb_parts.add(a)
            if event['type'] == 'ITEM_PURCHASED' and event['timestamp'] > 60000:
                if event['participantId'] not in first_resets: first_resets[event['participantId']] = event['timestamp']/1000

    def snap(min_val):
        d = {}
        if min_val < len(frames):
            for pid, stats in frames[min_val]['participantFrames'].items():
                d[int(pid)] = {'cs': stats['minionsKilled']+stats['jungleMinionsKilled'], 'xp': stats['xp']}
        return d

    s6, s12, s18 = snap(6), snap(12), snap(18)
    
    res = []
    for p in participants:
        pid = p['participantId']
        tid = p['teamId']
        opp = opponents.get(pid)
        
        k,d,a = p['kills'], p['deaths'], p['assists']
        dmg, gold = p['totalDamageDealtToChampions'], p['goldEarned']
        dpm = int(dmg/duration)
        kp = (k+a)/team_totals[tid]['kills']*100 if team_totals[tid]['kills'] else 0
        
        leads = {'C6':'N/A','C12':'N/A','C18':'N/A','X6':'N/A','X12':'N/A','X18':'N/A'}
        if opp:
            if pid in s6 and opp in s6: 
                leads['C6'] = s6[pid]['cs']-s6[opp]['cs']; leads['X6'] = s6[pid]['xp']-s6[opp]['xp']
            if pid in s12 and opp in s12: 
                leads['C12'] = s12[pid]['cs']-s12[opp]['cs']; leads['X12'] = s12[pid]['xp']-s12[opp]['xp']
            if pid in s18 and opp in s18: 
                leads['C18'] = s18[pid]['cs']-s18[opp]['cs']; leads['X18'] = s18[pid]['xp']-s18[opp]['xp']

        reset = first_resets.get(pid)
        r_str = f"{int(reset//60)}:{int(reset%60):02d}" if reset else "-"
        
        res.append({
            'Player': p['riotIdGameName'], 'Role': p.get('individualPosition'),
            'K/D/A': f"{k}/{d}/{a}", 'DPM': dpm, 'Dmg': dmg, '% Dmg': f"{dmg/team_totals[tid]['dmg']*100:.1f}%",
            'Gold': gold, '% Gold': f"{gold/team_totals[tid]['gold']*100:.1f}%", 'KP%': f"{kp:.1f}%",
            'FB KP': "Yes" if pid in fb_parts else "No", 'First Reset': r_str,
            'Pink B/P/K': f"{p.get('visionWardsBoughtInGame')}/{p.get('detectorWardsPlaced')}/{p.get('wardsKilled')}",
            'CSD@6': leads['C6'], 'CSD@12': leads['C12'], 'CSD@18': leads['C18'],
            'XPD@6': leads['X6'], 'XPD@12': leads['X12'], 'XPD@18': leads['X18']
        })
        
    df = pd.DataFrame(res)
    cols_comb = ['Player', 'Role', 'K/D/A', 'DPM', 'Dmg', '% Dmg', 'Gold', '% Gold', 'KP%', 'FB KP', 'First Reset']
    cols_lane = ['Player', 'Role', 'Pink B/P/K', 'CSD@6', 'CSD@12', 'CSD@18', 'XPD@6', 'XPD@12', 'XPD@18']
    return df[cols_comb], df[cols_lane]

# =============================================================================
# 4. ANALYSE SOLOKILLS (SPLIT TABLES)
# =============================================================================

def analyze_solo_kills_formatted(data):
    participants = data['match']['info']['participants']
    pid_map = {p['participantId']: {'name': p['riotIdGameName'], 'role': p.get('individualPosition'), 'team': p['teamId']} for p in participants}
    
    opponents = {}
    for pid, p in pid_map.items():
        for oid, op in pid_map.items():
            if p['team'] != op['team'] and p['role'] == op['role']: opponents[pid] = oid
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    
    list_ahead = []
    list_behind = []

    def get_gold_xp(pid, time):
        idx = int(time // 60000)
        if idx >= len(frames): idx = len(frames) - 1
        d = frames[idx]['participantFrames'][str(pid)]
        return d['totalGold'], d['xp']

    for frame in frames:
        for event in frame['events']:
            if event.get('type') == 'CHAMPION_KILL':
                killer = event.get('killerId')
                assists = event.get('assistingParticipantIds', [])
                victim = event.get('victimId')
                
                if killer in pid_map and not assists and killer in opponents:
                    opp = opponents[killer]
                    time = event['timestamp']
                    kg, kx = get_gold_xp(killer, time)
                    og, ox = get_gold_xp(opp, time)
                    diff_g = kg - og
                    diff_x = kx - ox
                    time_str = f"{int(time/60000)}:{int(time/1000%60):02d}"
                    
                    row = {'Tueur': pid_map[killer]['name'], 'Rôle': pid_map[killer]['role'], 'Temps': time_str, 'Victime': pid_map[victim]['name'], 'Avantage Or': diff_g, 'Avantage XP': diff_x}
                    if diff_g > 0: list_ahead.append(row)
                    else: list_behind.append(row)
    
    return pd.DataFrame(list_ahead), pd.DataFrame(list_behind)

# =============================================================================
# 5. ANALYSE RATIOS & STRUCTURES
# =============================================================================

def analyze_ratios_kda(data):
    participants = data['match']['info']['participants']
    res_d, res_k, res_a = [], [], []
    
    for p in participants:
        name = p['riotIdGameName']
        k, d, a = p['kills'], p['deaths'], p['assists']
        hs = p.get('totalHealsOnTeammates', 0) + p.get('totalDamageShieldedOnTeammates', 0)
        dt = p['totalDamageTaken']
        g = p['goldEarned']
        
        den_d = d if d > 0 else 1
        den_k = k if k > 0 else 1
        den_a = a if a > 0 else 1
        
        base = {'Joueur': name}
        res_d.append({**base, 'Heal/D': int(hs/den_d), 'Tank/D': int(dt/den_d), 'Gold/D': int(g/den_d)})
        res_k.append({**base, 'Heal/K': int(hs/den_k), 'Tank/K': int(dt/den_k), 'Gold/K': int(g/den_k)})
        res_a.append({**base, 'Heal/A': int(hs/den_a), 'Tank/A': int(dt/den_a), 'Gold/A': int(g/den_a)})
        
    return pd.DataFrame(res_d), pd.DataFrame(res_k), pd.DataFrame(res_a)

def analyze_structures(data):
    participants = data['match']['info']['participants']
    players = {p['participantId']: {'name': p['riotIdGameName'], 'plates': 0, 'towers': 0, 'plate_times': [], 'tower_times': []} for p in participants}
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    
    for frame in frames:
        for event in frame['events']:
            if event.get('type') == 'TURRET_PLATE_DESTROYED':
                kid = event.get('killerId')
                if kid in players:
                    players[kid]['plates'] += 1
                    players[kid]['plate_times'].append(f"{int(event['timestamp']/1000//60)}:{int(event['timestamp']/1000%60):02d}")
            if event.get('type') == 'BUILDING_KILL' and event.get('buildingType') == 'TOWER_BUILDING':
                kid = event.get('killerId')
                if kid in players: 
                    players[kid]['towers'] += 1
                    players[kid]['tower_times'].append(f"{int(event['timestamp']/1000//60)}:{int(event['timestamp']/1000%60):02d}")
                
    res = []
    for p in players.values():
        res.append({'Joueur': p['name'], 'Plates': p['plates'], 'Timings Plates': ", ".join(p['plate_times']), 'Tours': p['towers'], 'Timings Tours': ", ".join(p['tower_times'])})
    return pd.DataFrame(res).sort_values(by=['Tours', 'Plates'], ascending=False)

# =============================================================================
# 6. ANALYSE SPELLS (ULT / DASH / CC / MULTIKILL)
# =============================================================================

def analyze_spells_impact_global(data):
    ULTS = {
        'Akali': ['akalir'], 'AurelionSol': ['aurelionsolr'], 'Bard': ['bardr'], 'Briar': ['briarr'],
        'Camille': ['camiller'], 'Leona': ['leonasolarflare'], 'MissFortune': ['missfortunebullettime'],
        'MonkeyKing': ['monkeykingspintowin'], 'Mordekaiser': ['mordekaiserr'], 'Smolder': ['smolderr']
    }
    DASHS = {
        'Akali': ['akalie'], 'AurelionSol': ['aurelionsolw'], 'Briar': ['briarq'], 'Camille': ['camillee'],
        'Leona': ['leonazenithblade'], 'MonkeyKing': ['monkeykingnimbus'], 'Smolder': ['smoldere']
    }
    CC = {
        'Akali': ['akalir'], 'AurelionSol': ['aurelionsolq'], 'Bard': ['bardq'], 'Briar': ['briare'],
        'Camille': ['camillee'], 'Leona': ['leonazenithblade', 'leonashieldofdaybreak'], 'MonkeyKing': ['monkeykingspintowin']
    }

    participants = data['match']['info']['participants']
    players = {p['participantId']: {'name': p['riotIdGameName'], 'champ': p['championName'], 
               'k':0, 'k_r':0, 'k_d':0, 'k_cc':0, 'd':0, 'd_r':0, 'd_d':0, 'd_cc':0, 'a':0, 'a_r':0} for p in participants}
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']

    for frame in frames:
        for event in frame['events']:
            if event.get('type') == 'CHAMPION_KILL':
                killer = event.get('killerId')
                victim = event.get('victimId')
                assists = event.get('assistingParticipantIds', [])
                
                def check_spell(dmg_list, champ, spell_dict):
                    spells = spell_dict.get(champ, []) + [f"{champ.lower()}r"]
                    for dmg in dmg_list:
                        if dmg.get('name') == champ and dmg.get('spellName', '').lower() in spells:
                            return True
                    return False

                if killer in players:
                    p = players[killer]
                    p['k'] += 1
                    dmg = event.get('victimDamageReceived', [])
                    if check_spell(dmg, p['champ'], ULTS): p['k_r'] += 1
                    if check_spell(dmg, p['champ'], DASHS): p['k_d'] += 1
                    if check_spell(dmg, p['champ'], CC): p['k_cc'] += 1

                if victim in players:
                    p = players[victim]
                    p['d'] += 1
                    dmg = event.get('victimDamageDealt', [])
                    if check_spell(dmg, p['champ'], ULTS): p['d_r'] += 1
                    if check_spell(dmg, p['champ'], DASHS): p['d_d'] += 1
                    if check_spell(dmg, p['champ'], CC): p['d_cc'] += 1
                
                dmg_recv = event.get('victimDamageReceived', [])
                for aid in assists:
                    if aid in players:
                        p = players[aid]
                        p['a'] += 1
                        if check_spell(dmg_recv, p['champ'], ULTS): p['a_r'] += 1

    res = []
    for p in players.values():
        res.append({
            'Joueur': p['name'], 'Champ': p['champ'],
            '% Kill Ult': f"{(p['k_r']/p['k']*100 if p['k'] else 0):.0f}%",
            '% Death Ult': f"{(p['d_r']/p['d']*100 if p['d'] else 0):.0f}%",
            '% Assist Ult': f"{(p['a_r']/p['a']*100 if p['a'] else 0):.0f}%",
            '% Kill Dash': f"{(p['k_d']/p['k']*100 if p['k'] else 0):.0f}%",
            '% Kill CC': f"{(p['k_cc']/p['k']*100 if p['k'] else 0):.0f}%"
        })
    return pd.DataFrame(res)

def analyze_multikill_ult_impact(data):
    # Dico Ults
    ULTIMATE_KEYS = {
        'Akali': ['akalir', 'akalirb'], 'AurelionSol': ['aurelionsolr'], 'Bard': ['bardr'], 'Briar': ['briarr'],
        'Camille': ['camiller'], 'Leona': ['leonasolarflare'], 'MissFortune': ['missfortunebullettime'],
        'MonkeyKing': ['monkeykingspintowin'], 'Mordekaiser': ['mordekaiserr'], 'Smolder': ['smolderr']
    }
    participants = data['match']['info']['participants']
    players = {p['participantId']: {'name': p['riotIdGameName'], 'champion': p['championName'], 'kills': [], 'assists': []} for p in participants}
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']

    for frame in frames:
        for event in frame['events']:
            if event.get('type') == 'CHAMPION_KILL':
                time = event['timestamp']
                killer = event.get('killerId')
                assists = event.get('assistingParticipantIds', [])
                dmg_recv = event.get('victimDamageReceived', [])

                if killer in players:
                    p = players[killer]
                    ul = ULTIMATE_KEYS.get(p['champion'], []) + [f"{p['champion'].lower()}r"]
                    used = any(d.get('name') == p['champion'] and d.get('spellName', '').lower() in ul for d in dmg_recv)
                    p['kills'].append({'time': time, 'used': used})
                
                for aid in assists:
                    if aid in players:
                        p = players[aid]
                        ul = ULTIMATE_KEYS.get(p['champion'], []) + [f"{p['champion'].lower()}r"]
                        used = any(d.get('name') == p['champion'] and d.get('spellName', '').lower() in ul for d in dmg_recv)
                        p['assists'].append({'time': time, 'used': used})

    summary = []
    for p in players.values():
        def get_chains(evs):
            if not evs: return {'2':0,'3':0,'4+':0}
            evs.sort(key=lambda x: x['time'])
            chains = []
            curr = [evs[0]]
            for i in range(1, len(evs)):
                if evs[i]['time'] - evs[i-1]['time'] <= 10000: curr.append(evs[i])
                else: chains.append(curr); curr = [evs[i]]
            chains.append(curr)
            stats = {'2':0,'3':0,'4+':0}
            for c in chains:
                if len(c) >= 2 and any(e['used'] for e in c):
                    if len(c)==2: stats['2']+=1
                    elif len(c)==3: stats['3']+=1
                    else: stats['4+']+=1
            return stats

        k_stats = get_chains(p['kills'])
        a_stats = get_chains(p['assists'])
        summary.append({
            'Joueur': p['name'], 'Champ': p['champion'],
            'Double K (Ult)': k_stats['2'], 'Triple K (Ult)': k_stats['3'], 'Quadra+ K (Ult)': k_stats['4+'],
            'Double A (Ult)': a_stats['2'], 'Triple A (Ult)': a_stats['3'], 'Quadra+ A (Ult)': a_stats['4+']
        })
    return pd.DataFrame(summary).sort_values(by='Double K (Ult)', ascending=False)

# =============================================================================
# 7. ANALYSE SYNERGIE & CONTEXTE (BOT, JGL/SUP, HOARDING, SHUTDOWNS, CATCHES)
# =============================================================================

def analyze_botlane_synergy(data, pid_map, teams):
    duos = []
    for tid, roles in teams.items():
        if roles['ADC'] and roles['SUP']:
            duos.append({
                'Team': tid, 'ADC_ID': roles['ADC'], 'SUP_ID': roles['SUP'],
                'ADC_Name': pid_map[roles['ADC']]['name'], 'SUP_Name': pid_map[roles['SUP']]['name']
            })
    if not duos: return pd.DataFrame()

    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    stats = []

    for duo in duos:
        adc_id, sup_id = str(duo['ADC_ID']), str(duo['SUP_ID'])
        dist_sum, frames_together, total_frames = 0, 0, 0
        
        for frame in frames:
            if adc_id in frame['participantFrames'] and sup_id in frame['participantFrames']:
                p_adc = frame['participantFrames'][adc_id]['position']
                p_sup = frame['participantFrames'][sup_id]['position']
                dist = get_distance(p_adc, p_sup)
                dist_sum += dist
                if dist < 1200: frames_together += 1
                total_frames += 1
        
        # Combat Synergy
        adc_kills, sup_assists = 0, 0
        adc_buys, sup_buys = [], []
        
        for frame in frames:
            for event in frame['events']:
                if event.get('type') == 'CHAMPION_KILL':
                    if event.get('killerId') == int(adc_id):
                        adc_kills += 1
                        if int(sup_id) in event.get('assistingParticipantIds', []): sup_assists += 1
                if event.get('type') == 'ITEM_PURCHASED' and event['timestamp'] > 60000:
                    if event['participantId'] == int(adc_id): adc_buys.append(event['timestamp'])
                    if event['participantId'] == int(sup_id): sup_buys.append(event['timestamp'])
                    
        # Sync Backs
        synced = 0
        for t_a in adc_buys:
            if any(abs(t_a - t_s) <= 15000 for t_s in sup_buys): synced += 1

        stats.append({
            'ADC': duo['ADC_Name'], 'Support': duo['SUP_Name'],
            'Dist Moyenne': int(dist_sum/total_frames) if total_frames else 0,
            '% Temps Ensemble': f"{frames_together/total_frames*100:.1f}%" if total_frames else "0%",
            'Synergie Combat': f"{sup_assists/adc_kills*100:.1f}%" if adc_kills else "0%",
            'Synchro Backs': synced
        })
    return pd.DataFrame(stats)

def analyze_jungle_support_synergy(data, pid_map, teams):
    duos = []
    for tid, roles in teams.items():
        if roles['JGL'] and roles['SUP']:
            duos.append({
                'Team': tid, 'JGL_ID': roles['JGL'], 'SUP_ID': roles['SUP'],
                'JGL_Name': pid_map[roles['JGL']]['name'], 'SUP_Name': pid_map[roles['SUP']]['name']
            })
    if not duos: return pd.DataFrame()

    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    stats = []

    for duo in duos:
        jid, sid = str(duo['JGL_ID']), str(duo['SUP_ID'])
        prox_frames, total_frames = 0, 0
        
        # Proximité > 8min
        for frame in frames:
            if frame['timestamp'] < 480000: continue
            if jid in frame['participantFrames'] and sid in frame['participantFrames']:
                d = get_distance(frame['participantFrames'][jid]['position'], frame['participantFrames'][sid]['position'])
                if d < 2000: prox_frames += 1
                total_frames += 1
        
        # Objectifs & Combat
        obj_tot, obj_sup = 0, 0
        kill_tot, kill_sup = 0, 0
        
        for frame in frames:
            for event in frame['events']:
                assists = event.get('assistingParticipantIds', [])
                if event.get('type') == 'ELITE_MONSTER_KILL' and event.get('killerId') == int(jid):
                    obj_tot += 1
                    if int(sid) in assists: obj_sup += 1
                if event.get('type') == 'CHAMPION_KILL' and event.get('killerId') == int(jid):
                    kill_tot += 1
                    if int(sid) in assists: kill_sup += 1
                    
        stats.append({
            'Jungle': duo['JGL_Name'], 'Support': duo['SUP_Name'],
            '% Temps Ensemble': f"{prox_frames/total_frames*100:.1f}%" if total_frames else "0%",
            'Synergie Obj': f"{obj_sup/obj_tot*100:.1f}%" if obj_tot else "0%",
            'Synergie Combat': f"{kill_sup/kill_tot*100:.1f}%" if kill_tot else "0%"
        })
    return pd.DataFrame(stats)

def analyze_hoarding_deaths(data, pid_map, gold_threshold=2000):
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    res = []
    
    for frame in frames:
        gold_snap = {int(pid): d['currentGold'] for pid, d in frame['participantFrames'].items()}
        shopped = set()
        
        events = sorted(frame['events'], key=lambda x: x['timestamp'])
        for ev in events:
            if ev.get('type') == 'ITEM_PURCHASED': shopped.add(ev['participantId'])
            if ev.get('type') == 'CHAMPION_KILL':
                vic = ev['victimId']
                if vic not in shopped:
                    gold = gold_snap.get(vic, 0)
                    if gold >= gold_threshold:
                        res.append({'Joueur': pid_map[vic]['name'], 'Gold': gold, 'Time': f"{int(ev['timestamp']/60000)}m"})
    
    df = pd.DataFrame(res)
    if not df.empty:
        df = df.sort_values(by='Gold', ascending=False)
    return df

def analyze_shutdown_deaths(data, pid_map):
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    res = []
    
    for frame in frames:
        for ev in frame['events']:
            if ev.get('type') == 'CHAMPION_KILL':
                bounty = ev.get('bounty', 300)
                shutdown = bounty - 300
                if shutdown > 300:
                    res.append({
                        'Victime': pid_map[ev['victimId']]['name'], 
                        'Tueur': pid_map[ev.get('killerId', 0)]['name'] if ev.get('killerId', 0) in pid_map else 'Minion',
                        'Prime': shutdown,
                        'Time': f"{int(ev['timestamp']/60000)}m"
                    })
    
    df = pd.DataFrame(res)
    if not df.empty:
        df = df.sort_values(by='Prime', ascending=False)
    return df

def analyze_catches(data, pid_map):
    """
    Identifie les kills effectués en supériorité numérique (Catch).
    C'est un indicateur de bonne macro, de vision et de prise d'opportunité.
    """
    try:
        timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
        if not timeline: return pd.DataFrame()
        frames = timeline['info']['frames']
        
        catches = []
        FIGHT_RADIUS = 2000  # Rayon pour considérer qu'un allié de la victime est "présent"

        for frame in frames:
            # Snapshot des positions au début de la minute (utilisé si pas de pos exacte dans l'event)
            positions = {int(pid): d['position'] for pid, d in frame['participantFrames'].items()}
            
            for event in frame['events']:
                if event.get('type') == 'CHAMPION_KILL':
                    killer_id = event.get('killerId')
                    victim_id = event.get('victimId')
                    assists = event.get('assistingParticipantIds', [])
                    
                    if killer_id == 0 and not assists:
                        continue

                    attackers_count = len(assists) + (1 if killer_id != 0 else 0)
                    
                    kill_pos = event.get('position') or positions.get(victim_id)
                    
                    if kill_pos and victim_id in pid_map:
                        victim_team = pid_map[victim_id]['team']
                        defenders_count = sum(1 for pid, info in pid_map.items() if info['team'] == victim_team and pid in positions and get_distance(kill_pos, positions[pid]) <= FIGHT_RADIUS)
                        
                        if attackers_count > defenders_count:
                            time_ms = event['timestamp']
                            killer_name = pid_map.get(killer_id, {'name': 'Minion/Tour'})['name']
                            
                            catches.append({
                                'Tueur (Leader)': killer_name,
                                'Temps': f"{int(time_ms/1000//60)}:{int(time_ms/1000%60):02d}",
                                'Scénario': f"{attackers_count}v{defenders_count}",
                                'Victime': pid_map[victim_id]['name'],
                                'Différentiel': f"+{attackers_count - defenders_count}"
                            })

        df = pd.DataFrame(catches)
        if not df.empty:
            df.sort_values(by='Scénario', ascending=False, inplace=True)
        return df
    except Exception as e:
        print(f"Erreur dans analyze_catches : {e}")
        return pd.DataFrame()

def analyze_outnumbered_deaths(data, pid_map):
    """
    Identifie les joueurs qui meurent en étant en infériorité numérique.
    C'est un indicateur de mauvaise prise de décision ou de positionnement.
    """
    try:
        timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
        if not timeline: return pd.DataFrame()
        frames = timeline['info']['frames']
        
        outnumbered_deaths = []
        FIGHT_RADIUS = 2000

        for frame in frames:
            positions = {int(pid): d['position'] for pid, d in frame['participantFrames'].items()}
            
            for event in frame['events']:
                if event.get('type') == 'CHAMPION_KILL':
                    victim_id = event.get('victimId')
                    if victim_id not in pid_map: continue

                    attackers_count = len(event.get('assistingParticipantIds', [])) + (1 if event.get('killerId', 0) != 0 else 0)
                    
                    vic_pos = event.get('position') or positions.get(victim_id)
                    
                    if vic_pos:
                        victim_team = pid_map[victim_id]['team']
                        allies_nearby = sum(1 for pid, info in pid_map.items() if info['team'] == victim_team and pid in positions and get_distance(vic_pos, positions[pid]) <= FIGHT_RADIUS)
                        
                        if attackers_count > allies_nearby:
                            time_ms = event['timestamp']
                            outnumbered_deaths.append({
                                'Joueur (Victime)': pid_map[victim_id]['name'],
                                'Rôle': pid_map[victim_id]['role'],
                                'Temps': f"{int(time_ms/1000//60)}:{int(time_ms/1000%60):02d}",
                                'Scénario': f"{allies_nearby}v{attackers_count}"
                            })
        return pd.DataFrame(outnumbered_deaths).sort_values(by='Temps') if not pd.DataFrame(outnumbered_deaths).empty else pd.DataFrame()
    except Exception as e:
        print(f"Erreur dans analyze_outnumbered_deaths : {e}")
        return pd.DataFrame()

def analyze_gold_spikes(file_path, threshold_ratio=2.0, min_gold=600):
    """
    Détecte les pics de revenus anormaux sur une minute et identifie la source.
    
    :param file_path: Chemin du fichier JSON.
    :param threshold_ratio: Le gain doit être X fois supérieur à la moyenne des autres joueurs pour être détecté.
    :param min_gold: Le gain brut minimum (en PO) pour être considéré (évite les faux positifs en début de game).
    :return: Un DataFrame Pandas avec le timing, le joueur et la cause du gain.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return f"Erreur : {e}"

    # --- 1. Initialisation ---
    participants = data['match']['info']['participants']
    pid_map = {p['participantId']: {'name': p['riotIdGameName'], 'role': p.get('individualPosition')} for p in participants}
    
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    if not timeline: return "Timeline introuvable"
    frames = timeline['info']['frames']
    
    spikes = []

    # --- 2. Analyse Minute par Minute ---
    for i in range(len(frames) - 1):
        # On compare la frame actuelle (i) et la suivante (i+1) pour avoir le gain sur la minute
        current_frame = frames[i]
        next_frame = frames[i+1]
        minute = int(next_frame['timestamp'] / 60000)
        
        # Calculer les gains d'or de TOUS les joueurs sur cet intervalle
        gains = {}
        for pid_str, p_next in next_frame['participantFrames'].items():
            pid = int(pid_str)
            gold_next = p_next['totalGold']
            gold_curr = current_frame['participantFrames'][pid_str]['totalGold']
            gains[pid] = gold_next - gold_curr
            
        # Moyenne des gains de la minute (Benchmark)
        avg_gain = np.mean(list(gains.values()))
        
        # Identifier les Joueurs au-dessus du lot
        for pid, gain in gains.items():
            # Critère : Gain explosif (&gt; 2x la moyenne ET &gt; 600 PO)
            if gain > (avg_gain * threshold_ratio) and gain >= min_gold:
                
                # --- 3. Enquête sur la Cause (Analyse des Événements) ---
                causes = []
                # Les événements qui ont causé ce gain sont dans la frame suivante (ou l'intervalle)
                events = next_frame['events']
                
                for ev in events:
                    # Cause 1 : Kills & Assists
                    if ev.get('type') == 'CHAMPION_KILL':
                        if ev.get('killerId') == pid:
                            victim_name = pid_map.get(ev.get('victimId'), {}).get('name', 'Ennemi')
                            bounty = ev.get('bounty', 300)
                            causes.append(f"Kill ({victim_name}) +{bounty}g")
                        elif pid in ev.get('assistingParticipantIds', []):
                            causes.append("Assist")
                            
                    # Cause 2 : Structures (Plates & Tours)
                    if ev.get('type') == 'TURRET_PLATE_DESTROYED':
                        if ev.get('killerId') == pid or pid in ev.get('assistingParticipantIds', []):
                            causes.append("Plate")
                    
                    if ev.get('type') == 'BUILDING_KILL':
                        if ev.get('killerId') == pid or pid in ev.get('assistingParticipantIds', []):
                            b_type = ev.get('buildingType', 'Structure').replace('_BUILDING', '')
                            causes.append(f"Destruction {b_type}")

                    # Cause 3 : Objectifs Neutres (Baron, Dragon)
                    if ev.get('type') == 'ELITE_MONSTER_KILL':
                        # Si le joueur (ou son équipe) tue un Baron/Elder, c'est un gros gain global
                        if ev.get('killerId') == pid:
                            causes.append(f"{ev.get('monsterType')}")

                # Si aucune cause majeure trouvée, c'est du farming pur ou un passif (ex: Draven/TF/GP)
                cause_str = ", ".join(causes) if causes else "Farming intensif / Wave clear"
                
                spikes.append({
                    'Minute': minute,
                    'Joueur': pid_map[pid]['name'],
                    'Rôle': pid_map[pid]['role'],
                    'Gain (1min)': gain,
                    'Moyenne des autres': int(avg_gain),
                    'Ratio': f"{gain/avg_gain:.1f}x",
                    'Cause Probable': cause_str
                })

    # --- 4. Résultat ---
    df = pd.DataFrame(spikes)
    if not df.empty:
        df.sort_values(by='Gain (1min)', ascending=False, inplace=True)
        
    return df

# =============================================================================
# 8. ANALYSE TACTIQUE AVANCÉE (SPLITPUSH, JUNGLE BIAS, VISION SETUP)
# =============================================================================

def analyze_splitpush_isolation(data, pid_map):
    """
    Calcule le % de temps passé 'isolé' (loin de tout allié) après 15 minutes.
    Indicateur clé pour les Toplaners (Splitpush) vs les joueurs qui groupent trop.
    """
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    
    # Configuration
    ISOLATION_DIST = 2500 # Distance pour être considéré "Seul"
    START_TIME = 15 # On analyse après 15 minutes
    
    stats = {pid: {'frames_iso': 0, 'frames_tot': 0} for pid in pid_map}
    participant_teams = {p: pid_map[p]['team'] for p in pid_map}

    for frame in frames:
        if frame['timestamp'] < START_TIME * 60 * 1000: continue
        
        positions = {int(pid): d['position'] for pid, d in frame['participantFrames'].items()}
        
        for pid, pos in positions.items():
            stats[pid]['frames_tot'] += 1
            team = participant_teams[pid]
            
            # Chercher l'allié le plus proche
            min_dist_ally = float('inf')
            for ally_pid, ally_pos in positions.items():
                if ally_pid != pid and participant_teams[ally_pid] == team:
                    d = get_distance(pos, ally_pos)
                    if d < min_dist_ally: min_dist_ally = d
            
            # Si l'allié le plus proche est loin, on est isolé (Splitpush)
            if min_dist_ally > ISOLATION_DIST:
                stats[pid]['frames_iso'] += 1

    res = []
    for pid, info in stats.items():
        pct = (info['frames_iso'] / info['frames_tot'] * 100) if info['frames_tot'] else 0
        res.append({
            'Joueur': pid_map[pid]['name'],
            'Rôle': pid_map[pid]['role'],
            '% Temps Isolé (>15min)': f"{pct:.1f}%"
        })
    
    # On trie pour voir les gros splitpushers en premier (souvent Top/Mid)
    return pd.DataFrame(res).sort_values(by='% Temps Isolé (>15min)', ascending=False)

def analyze_jungle_lane_bias(data, pid_map):
    """
    Détermine sur quelle lane le jungler a passé le plus de temps durant l'Early Game (0-14 min).
    Utile pour savoir : "Est-ce que mon jungler a joué pour ma lane ?"
    """
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    
    LANES = {'TOP': {'x': 1500, 'y': 13500}, 'BOT': {'x': 13500, 'y': 1500}, 'MID': {'x': 7500, 'y': 7500}}
    junglers = [pid for pid, info in pid_map.items() if info['role'] == 'JUNGLE']
    jungle_stats = {pid: {'TOP': 0, 'MID': 0, 'BOT': 0, 'JUNGLE/RIVER': 0} for pid in junglers}

    for frame in frames:
        if frame['timestamp'] > 14 * 60 * 1000: break
        for jid in junglers:
            if str(jid) in frame['participantFrames']:
                pos = frame['participantFrames'][str(jid)]['position']
                found_lane = False
                for lane_name, lane_pos in LANES.items():
                    if get_distance(pos, lane_pos) < 2500:
                        jungle_stats[jid][lane_name] += 1
                        found_lane = True; break
                if not found_lane: jungle_stats[jid]['JUNGLE/RIVER'] += 1
                    
    res = []
    for jid, counts in jungle_stats.items():
        total = sum(counts.values())
        if total == 0: continue
        res.append({'Jungler': pid_map[jid]['name'], 'Top Présence': f"{counts['TOP']/total*100:.1f}%", 'Mid Présence': f"{counts['MID']/total*100:.1f}%", 'Bot Présence': f"{counts['BOT']/total*100:.1f}%", 'Farm/Roam': f"{counts['JUNGLE/RIVER']/total*100:.1f}%"})
    return pd.DataFrame(res)

def analyze_vision_setup_objectives(data, pid_map):
    """
    Vérifie si de la vision (Ward/Pink) a été posée autour des objectifs (Dragon/Baron)
    dans la minute PRÉCÉDANT leur mort. C'est le "Vision Setup".
    """
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    events = [e for f in timeline['info']['frames'] for e in f['events']]
    objectives = [e for e in events if e.get('type') == 'ELITE_MONSTER_KILL']
    LOCS = {'DRAGON': {'x': 9866, 'y': 4414}, 'BARON_NASHOR': {'x': 5007, 'y': 10471}, 'RIFTHERALD': {'x': 5007, 'y': 10471}, 'HORDE': {'x': 5007, 'y': 10471}}
    setup_scores = {pid: 0 for pid in pid_map}
    
    for obj in objectives:
        obj_type = obj.get('monsterType')
        if obj_type not in LOCS: continue
        death_time = obj['timestamp']
        for w in events:
            if w.get('type') == 'WARD_PLACED' and w.get('creatorId') in setup_scores:
                if 0 < (death_time - w['timestamp']) <= 60000: setup_scores[w['creatorId']] += 1

    res = []
    for pid, score in setup_scores.items():
        role = pid_map[pid]['role']
        if score > 0 or role in ['UTILITY', 'JUNGLE']:
            res.append({'Joueur': pid_map[pid]['name'], 'Rôle': role, 'Wards "Setup" (1min avant obj)': score})
    return pd.DataFrame(res).sort_values(by='Wards "Setup" (1min avant obj)', ascending=False)

def analyze_coaching_metrics_v2(data):
    participants = data['match']['info']['participants']
    timeline = data.get('timeline', data.get('match', {}).get('timeline', {}))
    frames = timeline['info']['frames']
    
    # 1. Stats Mécaniques & Conversion
    mech_stats = []
    for p in participants:
        challenges = p.get('challenges', {})
        gold = p['goldEarned']
        dmg = p['totalDamageDealtToChampions']
        efficiency = (dmg / gold) if gold > 0 else 0
        
        mech_stats.append({
            'Joueur': p['riotIdGameName'],
            'Rôle': p.get('individualPosition', 'NONE'),
            'Dégâts/Gold': f"{efficiency:.2f}",
            'Skillshots Touchés': challenges.get('skillshotsHit', 0),
            'Skillshots Esquivés': challenges.get('skillshotsDodged', 0)
        })

    # 2. Dominance Early Game (@14 min)
    idx_14 = min(14, len(frames) - 1)
    frame_14 = frames[idx_14]
    
    early_stats = []
    # Mapping rapide ID->Nom pour cette frame
    id_map = {p['participantId']: {'n': p['riotIdGameName'], 'r': p.get('individualPosition')} for p in participants}

    for pid_str, p_data in frame_14['participantFrames'].items():
        pid = int(pid_str)
        info = id_map.get(pid, {'n': '?', 'r': '?'})
        
        early_stats.append({
            'Joueur': info['n'],
            'Rôle': info['r'],
            'Dégâts @14m': p_data['damageStats']['totalDamageDoneToChampions'],
            'Gold @14m': p_data['totalGold'],
            'CS @14m': p_data['minionsKilled'] + p_data['jungleMinionsKilled'],
            'XP @14m': p_data['xp']
        })

    df_mech = pd.DataFrame(mech_stats).sort_values(by='Dégâts/Gold', ascending=False)
    df_early = pd.DataFrame(early_stats).sort_values(by='Dégâts @14m', ascending=False)
    
    return df_mech, df_early
# =============================================================================
#                           MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # --- CONFIGURATION DU FICHIER ---
    # Utilisez un chemin relatif simple : le fichier JSON doit être dans le même dossier
    json_file_path = 'test_data.json'
    
    print("\n" + "="*60)
    print(f"   ANALYSE COMPLÈTE DE LA PARTIE : {json_file_path}")
    print("="*60 + "\n")
    
    data = load_data(json_file_path)
    
    if data:
        # Prépare les infos participants
        participants_info = data['match']['info']['participants']
        pid_map, teams = get_participant_info(participants_info)

        # 1. JUNGLE
        print(">>> 1. JUNGLE PATHING & FARMING <<<")
        df_j_sum, df_j_obj, df_j_det = analyze_jungle_complete(data)
        if df_j_sum is not None: 
            print(df_j_sum.to_string(index=False))
            print("\n--- Objectifs Neutres ---")
            print(df_j_obj.to_string(index=False))
            print("\n--- Détail Morts Alliées (AFK Farm?) ---")
            print(df_j_det.to_string(index=False))
        
        # 2. TIMING & SHOP
        print("\n" + "="*60 + "\n")
        print(">>> 2. ANALYSE TIMINGS (SHOP & BACKS) <<<")
        df_bad, df_spike = analyze_shop_timings(data)
        print("\n--- Bad Backs (Morts < 45s après achat) ---")
        print(df_bad.to_string(index=False))
        print("\n--- Power Spikes (Kills < 45s après achat) ---")
        print(df_spike.to_string(index=False))
        
        print("\n--- Gold Différentiel (5min après 1er back) ---")
        print(analyze_gold_diff_5min(data).to_string(index=False))

        # 3. STATS AVANCEES
        print("\n" + "="*60 + "\n")
        print(">>> 3. STATISTIQUES AVANCÉES & LEADS <<<")
        df_comb, df_lan = analyze_advanced_split(data)
        if df_comb is not None:
            print("--- Combat & Économie ---")
            print(df_comb.to_string(index=False))
            print("\n--- Laning & Vision ---")
            print(df_lan.to_string(index=False))
            
        # 4. SOLOKILLS
        print("\n" + "="*60 + "\n")
        print(">>> 4. SOLOKILLS (AVANCE vs RETARD) <<<")
        df_ahead, df_behind = analyze_solo_kills_formatted(data)
        
        if not df_ahead.empty:
            print("\n--- SNOWBALL (Kills en étant en avance Gold) ---")
            print(df_ahead.to_string(index=False))
        else:
            print("\n--- SNOWBALL : Aucun kill en avance détecté ---")

        if not df_behind.empty:
            print("\n--- OUTPLAYS / COMEBACK (Kills en étant en retard/égalité) ---")
            print(df_behind.to_string(index=False))
        else:
            print("\n--- OUTPLAYS : Aucun kill en retard détecté ---")

        # 5. RATIOS
        print("\n" + "="*60 + "\n")
        print(">>> 5. RATIOS D'EFFICACITÉ (PER DEATH/KILL/ASSIST) <<<")
        df_rd, df_rk, df_ra = analyze_ratios_kda(data)
        print("--- Par Mort (Rentabilité) ---")
        print(df_rd.to_string(index=False))
        print("\n--- Par Assist (Utilité) ---")
        print(df_ra.to_string(index=False))

        # 6. SPELLS & STRUCTURES
        print("\n" + "="*60 + "\n")
        print(">>> 6. IMPACT SORTS & STRUCTURES <<<")
        print("--- Impact des Sorts (Ult / Dash / CC) ---")
        print(analyze_spells_impact_global(data).to_string(index=False))
        print("\n--- Multikills avec Ultime ---")
        print(analyze_multikill_ult_impact(data).to_string(index=False))
        print("\n--- Destruction de Structures (Avec Timings) ---")
        print(analyze_structures(data).to_string(index=False))

        # 7. SYNERGIE & CONTEXTE
        print("\n" + "="*60 + "\n")
        print(">>> 7. SYNERGIES & CONTEXTE <<<")
        print("--- Synergie Botlane ---")
        print(analyze_botlane_synergy(data, pid_map, teams).to_string(index=False))
        print("\n--- Synergie Jungle/Support ---")
        print(analyze_jungle_support_synergy(data, pid_map, teams).to_string(index=False))
        
        print("\n--- Greed (Morts avec > 2000 golds) ---")
        df_hoard = analyze_hoarding_deaths(data, pid_map)
        if not df_hoard.empty: print(df_hoard.to_string(index=False))
        else: print("Aucun")
        
        print("\n--- Shutdowns Donnés (> 300g) ---")
        df_shut = analyze_shutdown_deaths(data, pid_map)
        if not df_shut.empty: print(df_shut.to_string(index=False))
        else: print("Aucun")
        
        print("\n--- CATCHES (Kills en supériorité numérique) ---")
        df_catch = analyze_catches(data, pid_map)
        if not df_catch.empty: print(df_catch.to_string(index=False))
        else: print("Aucun")
        
        print("\n--- MORTS EN INFÉRIORITÉ NUMÉRIQUE (Caught) ---")
        df_outnumbered = analyze_outnumbered_deaths(data, pid_map)
        if not df_outnumbered.empty: print(df_outnumbered.to_string(index=False))
        else: print("Aucun")

        # --- Analyse des pics de revenus ---
        print("\n--- ANALYSE DES PICS DE REVENUS (GOLD SPIKES) ---")
        df_gold_spikes = analyze_gold_spikes(json_file_path) # Appel corrigé
        if isinstance(df_gold_spikes, pd.DataFrame):
            pd.set_option('display.max_colwidth', None)
            print(df_gold_spikes.to_string(index=False))
        else:
            print(df_gold_spikes)

        # 8. ANALYSE TACTIQUE AVANCÉE
        print("\n" + "="*60 + "\n")
        print(">>> 8. ANALYSE TACTIQUE AVANCÉE (SPLITPUSH, JUNGLE BIAS, VISION SETUP) <<<")
        
        print("\n--- % Temps Isolé (Splitpush > 15min) ---")
        df_split = analyze_splitpush_isolation(data, pid_map)
        if not df_split.empty:
            print(df_split.to_string(index=False))
        else:
            print("Analyse du splitpush non disponible.")

        print("\n--- Bias de Présence des Junglers (0-14min) ---")
        df_bias = analyze_jungle_lane_bias(data, pid_map)
        if not df_bias.empty:
            print(df_bias.to_string(index=False))
        else:
            print("Analyse du bias jungler non disponible.")

        print("\n--- Vision Setup avant Objectifs Majeurs ---")
        df_vision = analyze_vision_setup_objectives(data, pid_map)
        if not df_vision.empty:
            print(df_vision.to_string(index=False))
        else:
            print("Analyse de la vision non disponible.")

        # 9. MÉTRIQUES DE COACHING
        print("\n" + "="*60 + "\n")
        print(">>> 9. MÉTRIQUES DE COACHING V2 <<<")
        df_mech, df_early = analyze_coaching_metrics_v2(data)
        
        print("\n--- Efficacité Mécanique & Conversion Gold/Dégâts ---")
        if not df_mech.empty:
            print(df_mech.to_string(index=False))

        print("\n--- Dominance Early Game (@14 min) ---")
        if not df_early.empty:
            print(df_early.to_string(index=False))

        print("\n" + "="*60)
        print("   FIN DE L'ANALYSE   ")
        print("="*60 + "\n")