from urllib.request import urlopen
import numpy as np
import re

class PlayerStats():
    def __init__(self, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, bb, cc, dd, ee):
        self.id = a
        self.name = b
        self.position = c
        self.age = d
        self.team = e
        self.games_played = f
        self.games_started = g
        self.minutes = h
        self.field_goals_made = i
        self.field_goals_attempted = j
        self.field_goals_percentage = k
        self.three_pointers_made = l
        self.three_pointers_attempted = m
        self.three_point_percentage = n
        self.two_pointers_made = o
        self.two_pointers_attempted = p
        self.two_point_percentage = q
        self.effective_field_goal_percentage = r
        self.free_throws_made = s
        self.free_throws_attempted = t
        self.free_throws_percentage = u
        self.offensive_rebounds = v
        self.defensive_rebounds = w
        self.total_rebounds = x
        self.assists =y
        self.steals = z
        self.blocks = aa
        self.turnovers = bb
        self.personal_fouls = cc
        self.points = dd
        self.year = ee
        self.defensive_win_shares = 'N/A'
        self.defensive_win_share_normalized = None
        self.win_shares = 'N/A'
        self.defensive_plus_minus = 'N/A'
        self.fg_add = 'N/A'
        self.ts_add = 'N/A'
        self.usage_percentage = 'N/A'
        self.assist_rate = 'N/A'
        self.steal_rate = 'N/A'
        self.block_rate = 'N/A'
        self.rebound_rate = 'N/A'
        self.turnover_rate = 'N/A'
        self.assist_share = 'N/A'
        self.normal_ts = None
        self.shot_volume = None
        self.normal_assist = None
        self.calc_assist_rates = None
        self.normal_assist_volume = None
        self.normal_rebounds = None
        self.normal_rebounds_volume = None
        self.scoring = -999999999999999999999999999999999999999
        self.rebounding = -999999999999999999999999999999999999999
        self.playmaking = -999999999999999999999999999999999999999
        self.defense = -999999999999999999999999999999999999999
        self.vorp = -999999999999999999999999999999999999999
    
    def calcScoring(self):
        if (self.normal_ts == None or self.normal_shot_volume == None):
            return None
        curr_stat = self.normal_ts + (1.5 * self.normal_shot_volume)
        self.scoring = curr_stat
        return curr_stat

    def calcPlaymaking(self):
        if (self.normal_assist == None or self.normal_assist_volume == None):
            return None
        curr_stat = self.normal_assist + (1.5 * self.normal_assist_volume)
        self.playmaking = curr_stat
        return curr_stat

    def calcRebounding(self):
        if (self.normal_rebounds == None):
            return None
        curr_stat = self.normal_rebounds + (1.5 * self.normal_rebounds_volume)
        self.rebounding = curr_stat
        return curr_stat

    def calcDefense(self, offset, avg_blocks, avg_steals, avg_fouls, avg_win_share, avg_srate, avg_brate):
        if (self.defensive_plus_minus == 'N/A' or self.defensive_win_shares == 'N/A'):
            return None
        block_num = (float(self.block_rate) / avg_brate) + (int(self.blocks) / avg_blocks)
        steals_num = (float(self.steal_rate) / avg_srate) + (int(self.steals) / avg_steals)
        fouls_num = int(self.personal_fouls) / avg_fouls
        if(fouls_num < 1):
            fouls_num = 1
        box_defense = (block_num + steals_num) / (fouls_num)

        advanced_defense = float(self.defensive_plus_minus) * int(self.games_played) / 10
        curr_stat = box_defense + advanced_defense - offset
        self.defense = curr_stat
        return curr_stat

    def calcVorp(self):
        if(self.defensive_win_share_normalized != None):
            self.vorp = (.5 * self.rebounding) + (2.5 * self.scoring) + self.playmaking + (2 * self.defensive_win_share_normalized)
class statHandler():
    def __init__(self, year) -> None:
        self.year = year
        self.players = []
    def getStats(self):
        ###url = "https://www.basketball-reference.com/leagues/NBA_"+ self.year + "_totals.html"
        ##page = urlopen(url)
        ##html_bytes = page.read()
        ##html = html_bytes.decode("utf-8")
        html = open("2023-24 NBA Player Stats_ Totals _ Basketball-Reference.com.html", "r").read()
        player_stats = html.split("</tr>")

        for i in player_stats:
            stats = i.split("</td>")
            if (len(stats) == 30):
                stat_list = []
                for j in stats:
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)
                    for z in y:
                        c = z[1:]
                        if (c != '\n'):
                            stat_list.append(c)
                if(len(stat_list) < 30):
                    continue
                self.players.append(PlayerStats(stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5], stat_list[6], stat_list[7], stat_list[8], 
                                        stat_list[9], stat_list[10], stat_list[11], stat_list[12], stat_list[13], stat_list[14], stat_list[15], 
                                            stat_list[16], stat_list[17], stat_list[18], stat_list[19], stat_list[20], stat_list[21], stat_list[22], stat_list[23], stat_list[24], 
                                            stat_list[25], stat_list[26], stat_list[27], stat_list[28], stat_list[29], self.year))
                
        ###url = "https://www.basketball-reference.com/leagues/NBA_" + self.year + "_adj_shooting.html"
        ###page = urlopen(url)
        ###html_bytes = page.read()
        ###html = html_bytes.decode("utf-8")
        html = open("2023-24 NBA Player Stats_ Adjusted Shooting _ Basketball-Reference.com.html", "r").read()
        player_stats = html.split("</tr>")
        for ii in range(0, len(player_stats)):
            i = player_stats[ii]
            stats = i.split("</td>")
            if(len(stats) == 28):
                stat_list = []
                for j in stats:
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)
                    if (len(x) > 0):
                        if (x[0][11:len(x[0]) - 1] == 'fg_pts_added' 
                            or x[0][11:len(x[0]) - 1] == 'ts_pts_added' 
                            or x[0][11:len(x[0]) - 1] == 'team'):
                            if(len(y) > 0):
                                stat_list.append(y[0][1:])
                        elif(x[0][11:len(x[0]) - 1] == 'ranker'):
                            if(len(y) > 1):
                                stat_list.append(y[1][1:])
                if(len(stat_list) == 4):
                    for p in self.players:
                        if(p.name == stat_list[0] and p.team == stat_list[1]):
                            p.ts_add = stat_list[3]
                            p.fg_add = stat_list[2]
        ###url = "https://www.basketball-reference.com/leagues/NBA_"+ self.year +"_advanced.html"
        ###page = urlopen(url)
        ###html_bytes = page.read()
        ###html = html_bytes.decode("utf-8")
        html = open("2023-24 NBA Player Stats_ Advanced _ Basketball-Reference.com.html", "r").read()
        player_stats = html.split("</tr>")
        for ii in range(0, len(player_stats)):
            i = player_stats[ii]
            stats = i.split("</td>")
            if(len(stats) == 29):
                stat_list = []
                for j in stats:
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)
                    if (len(x) > 0):
                        if (x[0][11:len(x[0]) - 1] == 'ast_pct'
                            or x[0][11:len(x[0]) - 1] == 'blk_pct'
                            or x[0][11:len(x[0]) - 1] == 'stl_pct'
                            or x[0][11:len(x[0]) - 1] == 'trb_pct'
                            or x[0][11:len(x[0]) - 1] == 'tov_pct'
                            or x[0][11:len(x[0]) - 1] == 'usg_pct'
                            or x[0][11:len(x[0]) - 1] == 'dbpm'
                            or x[0][11:len(x[0]) - 1] == 'dws'
                            or x[0][11:len(x[0]) - 1] == 'ws'
                            or x[0][11:len(x[0]) - 1] == 'team_id'):
                            if(len(y) > 0):
                                stat_list.append(y[0][1:])
                        elif(x[0][11:len(x[0]) - 1] == 'ranker'):
                            if(len(y) > 1):
                                stat_list.append(y[1][1:])
                if(len(stat_list) == 11):
                    for p in self.players:
                        if(p.name == stat_list[0] and p.team == stat_list[1]):
                            p.rebound_rate = stat_list[2]
                            p.assist_rate = stat_list[3]
                            p.steal_rate = stat_list[4]
                            p.block_rate = stat_list[5]
                            p.turnover_rate = stat_list[6]
                            p.usage_percentage = stat_list[7]
                            p.defensive_win_shares = stat_list[8]
                            p.win_shares = stat_list[9]
                            p.defensive_plus_minus = stat_list[10]
        countDict = {}
        for p in self.players:
            if (p.name in countDict):
                countDict[p.name] +=1
            else:
                countDict[p.name] = 1
        for c in countDict:
            if (countDict[c] > 1):
                for p in self.players:
                    if (p.name == c and p.team != 'TOT'):
                        self.players.remove(p)
                for p in self.players:
                    if (p.name == c and p.team != 'TOT'):
                        self.players.remove(p)


    def calculateTopScorers(self):
        ts_add_arr = []
        for p in self.players:
            if(p.ts_add != 'N/A'):
                ts_add_arr.append(float(p.ts_add))
        mean = np.mean(ts_add_arr)
        std = np.std(ts_add_arr)
        for p in self.players:
            if(p.ts_add != 'N/A'):
                p.normal_ts = (float(p.ts_add) - mean) / std
        shot_volume_arr = []
        for p in self.players:
            p.shot_volume = int(p.field_goals_attempted) + (0.44 * int(p.free_throws_attempted))
            shot_volume_arr.append(p.shot_volume)

        mean = np.mean(shot_volume_arr)
        std = np.std(shot_volume_arr)
        for p in self.players:
            p.normal_shot_volume = (float(p.shot_volume) - mean) / std
        for p in self.players:
            p.calcScoring()

        self.players.sort(key=lambda x: x.scoring, reverse=True)
        max_val = self.players[0].scoring
        for p in self.players:
            p.scoring /= max_val
        print("TOP 10 SCORERS")
        for i in range(0, 10):
            p = self.players[i]
            print(p.id, p.name, p.team, str(p.scoring))

    def calculateTopPlayMakers(self):
        rates_arr = []
        for p in self.players:
            if(p.assist_rate == 'N/A' or p.turnover_rate == 'N/A' or p.usage_percentage == 'N/A'):
                continue
            assist_rate = (float(p.assist_rate) - float(p.turnover_rate)) / float(p.usage_percentage)
            rates_arr.append(assist_rate)
            p.calc_assist_rates = assist_rate
        mean, std = np.mean(rates_arr), np.std(rates_arr)
        for p in self.players:
            if(p.calc_assist_rates == None):
                continue
            p.normal_assist = (p.calc_assist_rates - mean) / std

        for p in self.players:
            rates_arr.append(int(p.assists))
        mean, std = np.mean(rates_arr), np.std(rates_arr)

        for p in self.players:
            p.normal_assist_volume = (int(p.assists) - mean) / std
        for p in self.players:
            p.calcPlaymaking()
            
        self.players.sort(key=lambda x: x.playmaking, reverse=True)
        print()
        max_val = self.players[0].playmaking
        for p in self.players:
            p.playmaking /= max_val
        print("TOP 10 PLAYMAKERS")
        for i in range(0, 10):
            p = self.players[i]
            print(p.id, p.name, p.team, str(p.playmaking))

    def calculateTopDefenders(self):
        count = 0
        total = 0
        total_blocks, total_steals, total_fouls, total_shares, total_brate, total_srate = 0, 0, 0, 0, 0, 0
        for p in self.players:
            total_blocks += int(p.blocks)
            total_fouls += int(p.personal_fouls)
            total_steals += int(p.steals)
            if (p.defensive_win_shares != 'N/A' and p.defensive_plus_minus != 'N/A' and p.steal_rate != 'N/A' and p.block_rate != 'N/A') :
                count+=1
                total_shares += float(p.defensive_win_shares)
                total_srate += float(p.steal_rate)
                total_brate += float(p.block_rate)

        total_blocks /= len(self.players)
        total_fouls /= len(self.players)
        total_steals /= len(self.players)
        total_brate /= count
        total_srate /= count
        total_shares /= count

        count = 0
        for p in self.players:
            score = p.calcDefense(0, total_blocks, total_steals,total_fouls, total_shares, total_srate, total_brate)
            if (score != None):
                count+=1
                total+= score
        for p in self.players:
            p.calcDefense(-1 * total/count, total_blocks, total_steals,total_fouls, total_shares, total_srate, total_brate)
            


        self.players.sort(key=lambda x: x.defense, reverse=True)
        max_val = self.players[0].defense
        for p in self.players:
            p.defense /= max_val
        print()
        print("TOP 10 DEFENDERS")
        for i in range(0, 10):
            p = self.players[i]
            print(p.id, p.name, p.team, str(p.defense))

    def calculateTopRebounders(self):
        reb_rates = []
        for p in self.players:
            if (p.rebound_rate == 'N/A'):
                continue
            reb_rates.append(float(p.rebound_rate))
        mean, std = np.mean(reb_rates), np.std(reb_rates)

        for p in self.players:
            if (p.rebound_rate == 'N/A'):
                continue
            p.normal_rebounds = (float(p.rebound_rate) - mean) / std

        off_rebs, def_rebs = [], []
        for p in self.players:
            off_rebs.append(int(p.offensive_rebounds))
            def_rebs.append(int(p.defensive_rebounds))
        off_mean, off_std, def_mean, def_std = np.mean(off_rebs), np.std(off_rebs), np.mean(def_rebs), np.std(def_rebs)
        for p in self.players:
            p.normal_rebounds_volume = 0.5 * (((int(p.offensive_rebounds) - off_mean) / off_std) + ((int(p.defensive_rebounds) - def_mean) / def_std))
        for p in self.players:
            p.calcRebounding()
            


        self.players.sort(key=lambda x: x.rebounding, reverse=True)
        max_val = self.players[0].rebounding
        for p in self.players:
            p.rebounding /= max_val

        print()
        print("TOP 10 REBOUNDERS")
        for i in range(0, 10):
            p = self.players[i]
            print(p.id, p.name, p.team, str(p.rebounding))
    def calculateTopPlayers(self):
        self.getStats()
        self.calculateTopScorers()
        self.calculateTopDefenders()
        self.calculateTopPlayMakers()
        self.calculateTopRebounders()
        self.players.sort(key=lambda x: x.defensive_win_shares, reverse=True)
        max_val = None
        for p in self.players:
            if(p.defensive_win_shares != 'N/A'):
                max_val = float(p.defensive_win_shares)
                break
        for p in self.players:
            if(p.defensive_win_shares != 'N/A'):
                p.defensive_win_share_normalized = float(p.defensive_win_shares) / max_val
        for p in self.players:
            p.calcVorp()

        self.players.sort(key=lambda x: x.vorp, reverse=True)

        print()
        print("TOP 10 PLAYERS")
        for i in range(0, 10):
            p = self.players[i]
            print(p.id, p.name, p.team, str(p.vorp))
        return self.players
s = statHandler("2024")
s.calculateTopPlayers()



