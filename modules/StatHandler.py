from urllib.request import urlopen
import numpy as np
import re
import torch
import torch.onnx
import onnx
import torch.nn as nn
import torch.optim as optim
import optuna
import certifi
import ssl
import random
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup, Comment
import re
import sys
import matplotlib.pyplot as plt
import json
from PlayerStats import PlayerStats
class StatHandler():

    #initalize instance and set target season
    def __init__(self, year) -> None:
        self.year = year
        self.players = []

    #Get all necesary statistics from local files
    def getStats(self):
        #Get stats from the 'totals' sections (Season total stats)
        html = open("./year_stats/" + self.year+"total.html", "r").read()
        soup = BeautifulSoup(html,"html.parser")
        table = soup.find("table", id="totals_stats")
        id_counter = 0
        for row in table.find_all("tr"):
            # Get each cell in the row
            cells = row.find_all("td")
            
            # If there are cells (to skip header rows, etc.)
            if cells:
                data = [cell.text.strip() for cell in cells]  # Get cell text
                stat_list = data
                for i in range(len(stat_list)):
                    if stat_list[i] == '':
                        stat_list[i] = None
                    elif stat_list[i][len(stat_list[i]) - 1] == '*':
                        stat_list[i] = stat_list[i][:len(stat_list[i])-1]
                if (stat_list[2] == None):
                    continue
                if (stat_list[2][1:] == "TM" or stat_list[2] == "TOT"):
                    stat_list[2] = "Multiple Teams"
                self.players.append(PlayerStats(id_counter, stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5], stat_list[6], stat_list[7], stat_list[8], 
                                        stat_list[9], stat_list[10], stat_list[11], stat_list[12], stat_list[13], stat_list[14], stat_list[15], 
                                            stat_list[16], stat_list[17], stat_list[18], stat_list[19], stat_list[20], stat_list[21], stat_list[22], stat_list[23], stat_list[24], 
                                            stat_list[25], stat_list[26], stat_list[27], stat_list[28], self.year))
                id_counter +=1
                
        #get the stats for precise shooting
        html = open("./year_stats/" + self.year+"shooting.html", "r", encoding="utf-8").read()
        # Remove unclosed comments
        html = re.sub("<!--\n", "\n", html)
        soup = BeautifulSoup(html,"html.parser")
        table = soup.find("table", id="adj-shooting")
        if table == None:
            table = soup.find("table", id="adj_shooting")
        team_index = 3
        if (self.year == '2025'):
            team_index = 2
        rows = table.find_all("tr")
        for row in rows:
            # Get each cell in the row
            cells = row.find_all("td")
            # If there are cells (to skip header rows, etc.)
            if cells:
                data = [cell.text.strip() for cell in cells]  # Get cell text
                stat_list = data
                for i in range(len(stat_list)):
                    if stat_list[i] == '':
                        stat_list[i] = None
                    elif stat_list[i][len(stat_list[i]) - 1] == '*':
                        stat_list[i] = stat_list[i][:len(stat_list[i])-1]
                if (stat_list[team_index] == None):
                    continue
                if (stat_list[team_index][1:] == "TM" or stat_list[team_index] == "TOT"):
                    stat_list[team_index] == 'Multiple Teams'
                for p in self.players:
                        if(p.name == stat_list[0] and p.team == stat_list[team_index]):
                            p.fg_add = stat_list[len(stat_list) - (2 + 3 - team_index)]
                            p.ts_add = stat_list[len(stat_list) - (1 + 3 - team_index)]
        
        #get advanced stats
        html = open("./year_stats/" + self.year+"advanced.html", "r").read()
        soup = BeautifulSoup(html,"html.parser")
        table = soup.find("table", id="advanced_stats")
        if not table:
            table = soup.find("table", id="advanced")
        rows = table.find_all("tr")
        id_counter = 0
        for row in rows:
            # Get each cell in the row
            cells = row.find_all("td")
            # If there are cells (to skip header rows, etc.)
            if cells:
                data = [cell.text.strip() for cell in cells]  # Get cell text
                stat_list = data
                for i in range(len(stat_list)):
                    if stat_list[i] == '':
                        stat_list[i] = None
                    elif stat_list[i][len(stat_list[i]) - 1] == '*':
                        stat_list[i] = stat_list[i][:len(stat_list[i])-1]
                if (stat_list[team_index] == None):
                    continue
                if (stat_list[team_index][1:] == "TM" or stat_list[team_index] == "TOT"):
                    stat_list[team_index] = "Multiple Teams"
                for p in self.players:
                        if(p.name == stat_list[0] and p.team == stat_list[team_index]):
                            p.rebound_rate = stat_list[12 + 3 - team_index]
                            p.assist_rate = stat_list[13 + 3 - team_index]
                            p.steal_rate = stat_list[14 + 3 - team_index]
                            p.block_rate = stat_list[15 + 3 - team_index]
                            p.turnover_rate = stat_list[16 + 3 - team_index]
                            p.usage_percentage = stat_list[17 + 3 - team_index]
                            p.defensive_win_shares = stat_list[20 + 3 - team_index]
                            p.win_shares = stat_list[21 + 3 - team_index]
                            p.defensive_plus_minus = stat_list[25 + 3 - team_index]
        
        return self.players


    #calculate the top scorers
    def calculateTopScorers(self, pr):
        ts_add_arr = []
        
        #transform the 'points added by true shooting above average' stat, into a z-score
        for p in self.players:
            if(p.ts_add != None):
                ts_add_arr.append(float(p.ts_add))

        #obtain the mean and standard deviation of the stat
        mean, std = np.mean(ts_add_arr), np.std(ts_add_arr)

        #normalize it
        for p in self.players:
            if(p.ts_add != None):
                p.normal_ts = (float(p.ts_add) - mean) / std

        #calculate the volume statistic for scoring
        shot_volume_arr = []
        for p in self.players:
            if (p.field_goals_attempted != None and p.free_throws_attempted != None):
                p.shot_volume = int(p.field_goals_attempted) + (0.44 * int(p.free_throws_attempted))
                shot_volume_arr.append(p.shot_volume)
            else:
                p.shot_volume = None

        #find the mean and standard deviation and then normalize
        mean, std = np.mean(shot_volume_arr), np.std(shot_volume_arr)
        for p in self.players:
            if (p.shot_volume != None):
                p.normal_shot_volume = (float(p.shot_volume) - mean) / std
            else:
                p.normal_shot_volume = None
        #calculate scoring
        score_arr = []
        for p in self.players:
            score_arr += [p.calcScoring()] if p.calcScoring() is not None else []
        mean, std = np.mean(score_arr), np.std(score_arr)

        for p in self.players:
            p.scoring = ((p.scoring - mean) / std)
        #sort players from best to worst scorers
        self.players.sort(key=lambda x: x.scoring, reverse=True)
        

        #print list out if necessary
        if(pr):
            print("TOP 10 SCORERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, str(p.scoring))

    #calculate top playmakers
    def calculateTopPlayMakers(self, pr):
        rates_arr = []

        #calculate assist rates
        for p in self.players:
            if(p.assist_rate == None or p.turnover_rate == None or p.usage_percentage == None):
                continue
            assist_rate = (float(p.assist_rate) - float(p.turnover_rate)) / float(p.usage_percentage)
            rates_arr.append(assist_rate)
            p.calc_assist_rates = assist_rate
        
        #find mean and standard deviation of the assist rates
        mean, std = np.mean(rates_arr), np.std(rates_arr)
        
        #normalize the assist rates
        for p in self.players:
            if(p.calc_assist_rates == None):
                continue
            p.normal_assist = (p.calc_assist_rates - mean) / std

        volume_arr = []

        #find mean and standard dev of the volume of assists
        for p in self.players:
            if (p.assists != None):
                volume_arr.append(int(p.assists))
        mean, std = np.mean(volume_arr), np.std(volume_arr)

        #normalize the assist volume
        for p in self.players:
            if (p.assists != None):
                p.normal_assist_volume = (int(p.assists) - mean) / std
            else:
                p.normal_assist_volume = None

        playmaking_arr = []
        #calculate playmaking score
        for p in self.players:
            playmaking_arr += [p.calcPlaymaking()] if p.calcPlaymaking() is not None else []
        mean, std = np.mean(playmaking_arr), np.std(playmaking_arr)
        for p in self.players:
            p.playmaking = ((p.playmaking - mean) / std)
        #sort players from best to worst playmakers
        self.players.sort(key=lambda x: x.playmaking, reverse=True)


        #print findings if specified
        if(pr):
            print()
            print("TOP 10 PLAYMAKERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, str(p.playmaking))

    #calculate top rebounders
    def calculateTopRebounders(self, pr):
        reb_rates = []

        for p in self.players:
            if (p.rebound_rate == None):
                continue
            reb_rates.append(float(p.rebound_rate))

        #find the mean and standard dev of rebound rates
        mean, std = np.mean(reb_rates), np.std(reb_rates)

        #normalize the rates
        for p in self.players:
            if (p.rebound_rate == None):
                continue
            p.normal_rebounds = (float(p.rebound_rate) - mean) / std

        off_rebs, def_rebs = [], []
        
        #find mean and standard dev of the two volume metrics
        for p in self.players:
            if (p.offensive_rebounds != None):
                    off_rebs.append(int(p.offensive_rebounds))
            if (p.defensive_rebounds != None):
                def_rebs.append(int(p.defensive_rebounds))
        off_mean, off_std, def_mean, def_std = np.mean(off_rebs), np.std(off_rebs), np.mean(def_rebs), np.std(def_rebs)

        #normalize the volume metrics and combine them
        for p in self.players:
            if (p.offensive_rebounds != None and p.defensive_rebounds != None):
                p.normal_rebounds_volume = 0.5 * (((int(p.offensive_rebounds) - off_mean) / off_std) + ((int(p.defensive_rebounds) - def_mean) / def_std))
        
        rebound_arr = []
        #calculate top rebounders
        for p in self.players:
            rebound_arr += [p.calcRebounding()] if p.calcRebounding() is not None else []
        mean, std = np.mean(rebound_arr), np.std(rebound_arr)
        for p in self.players:
            p.rebounding = ((p.rebounding - mean) / std)
        #sort best rebounder to worst
        self.players.sort(key=lambda x: x.rebounding, reverse=True)

        #print results if necessary
        if(pr):
            print()
            print("TOP 10 REBOUNDERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, str(p.rebounding))
    
    #calculate top defenders
    def calculateTopDefenders(self, pr):

        defense_arr = []
        for p in self.players:
            defense_arr += [float(p.defensive_win_shares)] if p.defensive_win_shares != None else []
        mean, std = np.mean(defense_arr), np.std(defense_arr)
        for p in self.players:
            p.defensive_win_share_normalized = ((float(p.defensive_win_shares) - mean) / std) if p.defensive_win_shares != None else -sys.maxsize - 1
        
        #sort by best to worst defender
        self.players.sort(key=lambda x: x.defensive_win_share_normalized, reverse=True)

        #print results if necessary
        if(pr):
            print()
            print("TOP 10 DEFENDERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, p.defensive_win_share_normalized)

    #calculate top players
    def calculateTopPlayers(self, pr):
        #obtain stats
        self.getStats()

        #calculate the necessary statistics
        self.calculateTopScorers(pr)
        self.calculateTopPlayMakers(pr)
        self.calculateTopRebounders(pr)
        self.calculateTopDefenders(pr)
        vorpArr = []
        #calculate VORP for each player
        for p in self.players:
            vorpArr.append(p.calcVorp())
        #filter out the blank stats
        vorpArr = [i for i in vorpArr if i is not None and i > -9000]

        #sort from best to worst player
        self.players.sort(key=lambda x: x.vorp, reverse=True)
        
        #obtain mean and standard dev for vorp
        mean, std = np.mean(vorpArr), np.std(vorpArr)

        #normalize vorp
        for p in self.players:
            p.normal_vorp = p.vorp - mean
            p.normal_vorp /= std
        for p in self.players:
            if (p.vorp == -sys.maxsize - 1 or p.defensive_win_share_normalized == -sys.maxsize - 1 
            or p.playmaking == -sys.maxsize - 1 or p.scoring == -sys.maxsize - 1 
            or p.rebounding == -sys.maxsize - 1):
                self.players.remove(p)
        #print results if necessary
        if(pr):
            print()
            print("TOP 10 PLAYERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, p.normal_vorp)

        #return full set of ranked player
        return self.players

    #save the data by pulling it from the web
    def saveData(year):
        context = ssl.create_default_context(cafile=certifi.where())
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_advanced.html", context=context)
        html_content = response.read()
        with open("./year_stats/" + year+"advanced.html", 'wb') as file:
            file.write(html_content)
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_ratings.html", context=context)
        html_content = response.read()
        with open("./year_stats/" + year+"teams.html", 'wb') as file:
            file.write(html_content)
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_adj_shooting.html", context=context)
        html_content = response.read()
        with open("./year_stats/" + year+"shooting.html", 'wb') as file:
            file.write(html_content)
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_totals.html", context=context)
        html_content = response.read()
        with open("./year_stats/" + year+"total.html", 'wb') as file:
            file.write(html_content)
    
    #get win/loss stat for each team from a given year
    def getYearStats(year):
        #read proper file
        html = open("./year_stats/" + year+"teams.html", "r").read()
        soup = BeautifulSoup(html,"html.parser")
        table = soup.find("table", id="ratings")
        nba_dict = {}
        nba_actual_wins = {}
        nba_games_played = {}
        for row in table.find_all("tr"):
            # Get each cell in the row
            cells = row.find_all("td")
            # If there are cells (to skip header rows, etc.)
            if cells:
                data = [cell.text.strip() for cell in cells]  # Get cell text
                stat_list = data
                for i in range(len(stat_list)):
                    if stat_list[i] == '':
                        stat_list[i] = None
                    elif stat_list[i][len(stat_list[i]) - 1] == '*':
                        stat_list[i] = stat_list[i][:len(stat_list[i])-1]
                nba_dict[stat_list[0]] = float(stat_list[5])
                nba_actual_wins[stat_list[0]] = int(stat_list[3])
                nba_games_played[stat_list[0]] = int(stat_list[3]) + int(stat_list[4])

        return nba_dict, nba_actual_wins, nba_games_played