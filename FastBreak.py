from urllib.request import urlopen
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
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

class PlayerStats():
    #initalize variables
    def __init__(self, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, bb, cc, dd, ee):
        self.id = a
        self.name = b
        self.age = c
        self.team = d
        self.position = e
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
        self.defensive_win_shares = None
        self.win_shares = None
        self.defensive_plus_minus = None
        self.fg_add = None
        self.ts_add = None
        self.usage_percentage = None
        self.assist_rate = None
        self.steal_rate = None
        self.block_rate = None
        self.rebound_rate = None
        self.turnover_rate = None
        self.assist_share = None
        self.normal_ts = None
        self.shot_volume = None
        self.normal_assist = None
        self.calc_assist_rates = None
        self.normal_assist_volume = None
        self.normal_rebounds = None
        self.normal_rebounds_volume = None
        self.defensive_win_share_normalized = -sys.maxsize - 1
        self.scoring = -sys.maxsize - 1
        self.rebounding = -sys.maxsize - 1
        self.playmaking = -sys.maxsize - 1
        self.defense = -sys.maxsize - 1
        self.vorp = -sys.maxsize - 1
        self.normal_vorp = -sys.maxsize - 1
        self.vol_modifier = 1.5
        self.rebound_modifier = .5
        self.scoring_modifier = 2.5
        self.playmaking_modifier = 1
        self.defense_modifier = 2
    
    #calculate and set scoring var
    def calcScoring(self):
        if (self.normal_ts == None or self.normal_shot_volume == None):
            return None
        curr_stat = self.normal_ts + (self.vol_modifier * self.normal_shot_volume)
        self.scoring = curr_stat
        return curr_stat

    #calculate and set playmaking var
    def calcPlaymaking(self):
        if (self.normal_assist == None or self.normal_assist_volume == None):
            return None
        curr_stat = self.normal_assist + (self.vol_modifier * self.normal_assist_volume)
        self.playmaking = curr_stat
        return curr_stat

    #calculate and set rebounding var
    def calcRebounding(self):
        if (self.normal_rebounds == None):
            return None
        curr_stat = self.normal_rebounds + (self.vol_modifier * self.normal_rebounds_volume)
        self.rebounding = curr_stat
        return curr_stat

    #calculate and set overall vorp
    def calcVorp(self):
        if(self.defensive_win_share_normalized != None):
            self.vorp = (self.rebound_modifier * self.rebounding) + (self.scoring_modifier * self.scoring) + (self.playmaking_modifier * self.playmaking) + (self.defense_modifier * self.defensive_win_share_normalized)
            return self.vorp

#Class to calculate all statistics and pull them
class statHandler():

    #initalize instance and set target season
    def __init__(self, year) -> None:
        self.year = year
        self.players = []

    #Get all necesary statistics from local files
    def getStats(self):
        #Get stats from the 'totals' sections (Season total stats)
        html = open(self.year+"total.html", "r").read()
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
                if (stat_list[2] == None or stat_list[2][1:] == 'TM'):
                    continue
                self.players.append(PlayerStats(id_counter, stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5], stat_list[6], stat_list[7], stat_list[8], 
                                        stat_list[9], stat_list[10], stat_list[11], stat_list[12], stat_list[13], stat_list[14], stat_list[15], 
                                            stat_list[16], stat_list[17], stat_list[18], stat_list[19], stat_list[20], stat_list[21], stat_list[22], stat_list[23], stat_list[24], 
                                            stat_list[25], stat_list[26], stat_list[27], stat_list[28], self.year))
                id_counter +=1
                
        #get the stats for precise shooting
        html = open(self.year+"shooting.html", "r", encoding="utf-8").read()
        # Remove unclosed comments
        html = re.sub("<!--\n", "\n", html)
        soup = BeautifulSoup(html,"html.parser")
        table = soup.find("table", id="adj-shooting")
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
                if (stat_list[3] == None or stat_list[3][1:] == 'TM'):
                    continue
                for p in self.players:
                        if(p.name == stat_list[0] and p.team == stat_list[3]):
                            p.fg_add = stat_list[len(stat_list) - 2]
                            p.ts_add = stat_list[len(stat_list) - 1]
        #get advanced stats
        html = open(self.year+"advanced.html", "r").read()
        soup = BeautifulSoup(html,"html.parser")
        table = soup.find("table", id="advanced_stats")
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
                if (stat_list[3] == None or stat_list[3][1:] == 'TM'):
                    continue
                for p in self.players:
                        if(p.name == stat_list[0] and p.team == stat_list[3]):
                            p.rebound_rate = stat_list[12]
                            p.assist_rate = stat_list[13]
                            p.steal_rate = stat_list[14]
                            p.block_rate = stat_list[15]
                            p.turnover_rate = stat_list[16]
                            p.usage_percentage = stat_list[17]
                            p.defensive_win_shares = stat_list[20]
                            p.win_shares = stat_list[21]
                            p.defensive_plus_minus = stat_list[25]
        
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
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_advanced.html")
        html_content = response.read()
        with open(year+"advanced.html", 'wb') as file:
            file.write(html_content)
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_ratings.html")
        html_content = response.read()
        with open(year+"teams.html", 'wb') as file:
            file.write(html_content)
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_adj_shooting.html")
        html_content = response.read()
        with open(year+"shooting.html", 'wb') as file:
            file.write(html_content)
        response = urlopen( "https://www.basketball-reference.com/leagues/NBA_"+ year +"_totals.html")
        html_content = response.read()
        with open(year+"total.html", 'wb') as file:
            file.write(html_content)
    
    #get win/loss stat for each team from a given year
    def getYearStats(year):
        #read proper file
        html = open(year+"teams.html", "r").read()
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
                
        
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Since we have 8 vectors of 4 elements each, input dimension will be 8 * 4 = 32
        self.input_dim = 8 * 5
        self.hidden_dim1 = 80
        self.hidden_dim2 = 40
        self.output_dim = 1 

        # Define the layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.output_dim)

    def forward(self, x):
        x = x.view(-1, 8 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class Performer():
    def __init__(self, year):
        # Dictionary mapping 3-letter team codes to team names
        self.nba_team_dict = {
            'ATL': 'Atlanta Hawks',
            'BOS': 'Boston Celtics',
            'BKN': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets',
            'CHO': 'Charlotte Hornets',
            'CHH': 'Charlotte Hornets',
            'CHI': 'Chicago Bulls',
            'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks',
            'DEN': 'Denver Nuggets',
            'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors',
            'HOU': 'Houston Rockets',
            'IND': 'Indiana Pacers',
            'LAC': 'Los Angeles Clippers',
            'SDC': 'San Diego Clippers',
            'LAL': 'Los Angeles Lakers',
            'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat',
            'MIL': 'Milwaukee Bucks',
            'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans',
            'NYK': 'New York Knicks',
            'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic',
            'PHI': 'Philadelphia 76ers',
            'PHX': 'Phoenix Suns',
            'PHO': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers',
            'SAC': 'Sacramento Kings',
            'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors',
            'UTA': 'Utah Jazz',
            'WAS': 'Washington Wizards',
            'WSB': 'Washington Bullets',
            'SEA': 'Seattle SuperSonics',
            'NJN': 'New Jersey Nets',
            'KCK': 'Kansas City Kings',
            'BRK': 'Brooklyn Nets',
            'VAN': 'Vancouver Grizzlies',
            'CHA2005': 'Charlotte Bobcats',
            'NOH': 'New Orleans Hornets',
            'NOK': 'New Orleans/Oklahoma City Hornets',
        }

        #full list of 3 letter codes
        self.nba_team_codes = [
            'ATL', 'BOS', 'BKN', 'CHA', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS', 
            'WSB', 'SEA', 'NJN', 'PHO', 'SDC', 'KCK', 'BRK', 'CHH', 'VAN', 'CHA2005', 'NOH'
            'NOK'
        ]

        # Initialize the dictionary with team codes as keys and 0 as values
        self.nba_team_arrs = {code: [] for code in self.nba_team_codes}
        self.nba_team_wins, self.nba_team_wins_count, self.nba_games_played = statHandler.getYearStats(year)
        self.year = year


    def performModel(self):
        s = statHandler(self.year)
        players = s.calculateTopPlayers(False)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        if Path("model.pth").is_file():
            checkpoint = torch.load('model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 7, 4)
        
        #set up example input by assinging players to the necessary team

        example_input = {}
        for p in players:
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8):
                example_input[a] = self.nba_team_arrs[a]
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            # Example target (for training purposes)
            key = i
            if (key == 'CHA' and int(self.year) <= 2014):
                key = 'CHA2005'
            example_target = torch.full((1, 1), self.nba_team_wins[self.nba_team_dict[key]])  # Example target labels for classification
            # Compute loss
            loss = criterion(output, example_target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #export model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model.pth')
    
    #show performance of the model without training it further
    def performModelNoUpdate(self):
        s = statHandler(self.year)
        players = s.calculateTopPlayers(False)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 7, 4)
        
        #set up example input

        example_input = {}
        for p in players:
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8):
                example_input[a] = self.nba_team_arrs[a]
        total_acc = 0
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            # Example target (for training purposes)
            key = i
            if (key == 'CHA' and int(self.year) <= 2014):
                key = 'CHA2005'
            example_target = torch.full((1, 1), self.nba_team_wins[self.nba_team_dict[key]])  # Example target labels for classification
            # Compute loss
            total_acc += 1 - abs((output.detach().numpy()[0][0] - self.nba_team_wins[self.nba_team_dict[key]]) / self.nba_team_wins[self.nba_team_dict[key]])
        total_acc /= len(example_input)
        return total_acc

    def performModelForTeam(self, team):
        s = statHandler(self.year)
        players = s.calculateTopPlayers(True)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 7, 4)
        
        #set up example input
        example_input = {}
        for p in players:
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8 and a == team):
                example_input[a] = self.nba_team_arrs[a]
        print('RESULTS FOR ' + team + ': ', self.year, )
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            # Example target (for training purposes)
            key = i
            if (key == 'CHA' and int(self.year) <= 2014):
                key = 'CHA2005'
            print(self.nba_team_dict[i], output.detach().numpy()[0][0], self.nba_team_wins[self.nba_team_dict[key]])
            example_target = torch.full((1, 1), self.nba_team_wins[self.nba_team_dict[key]])  # Example target labels for classification

    def showCase(self):
        s = statHandler(self.year)
        players = s.calculateTopPlayers(True)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 8, 4)
        
        #set up example input
        example_input = {}
        for p in players:
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8):
                example_input[a] = self.nba_team_arrs[a]
        total_loss = 0
        print('RESULTS: ', self.year)
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            # Example target (for training purposes)
            key = i
            if (key == 'CHA' and int(self.year) <= 2014):
                key = 'CHA2005'
            example_target = torch.full((1, 1), self.nba_team_wins[self.nba_team_dict[key]])  # Example target labels for classification
            # Compute loss
            print(self.nba_team_dict[i], output.detach().numpy()[0][0], self.nba_team_wins[self.nba_team_dict[key]])
            loss = criterion(output, example_target)
            total_loss += loss.item()
            print(f'Loss: {loss.item()}')

        #plot maps for stats
        for i in range(0, len(players)):
            plt.plot(i, players[i].normal_vorp, 'bo')
        plt.title("Top Players of " + self.year)
        plt.xlabel("Players")
        plt.ylabel("VORP")    
        plt.savefig("vorpChart"+ self.year+".png")
        plt.clf()

        players.sort(key=lambda x: x.scoring, reverse=True)
        for i in range(0, len(players)):
            plt.plot(i, players[i].scoring, 'bo')
        plt.title("Top Scorers of " + self.year)
        plt.xlabel("Players")
        plt.ylabel("Scoring")    
        plt.savefig("scoringChart"+ self.year+".png")
        plt.clf()


        players.sort(key=lambda x: x.playmaking, reverse=True)
        for i in range(0, len(players)):
            plt.plot(i, players[i].playmaking, 'bo')
        plt.title("Top Playmakers of " + self.year)
        plt.xlabel("Players")
        plt.ylabel("Playmaking")    
        plt.savefig("playmakingChart"+ self.year+".png")
        plt.clf()

        players.sort(key=lambda x: x.rebounding, reverse=True)
        for i in range(0, len(players)):
            plt.plot(i, players[i].rebounding, 'bo')
        plt.title("Top Rebounders of " + self.year)
        plt.xlabel("Players")
        plt.ylabel("Rebounding")    
        plt.savefig("reboundingChart"+ self.year+".png")
        plt.clf()

        players.sort(key=lambda x: x.defensive_win_share_normalized, reverse=True)
        for i in range(0, len(players)):
            plt.plot(i, players[i].defensive_win_share_normalized, 'bo')
        plt.title("Top Defenders of " + self.year)
        plt.xlabel("Players")
        plt.ylabel("Defense")    
        plt.savefig("defenseChart"+ self.year+".png")
        plt.clf()

        total_loss /= len(example_input)
        return total_loss
    def getYearPreds(self):
        s = statHandler(self.year)
        players = s.calculateTopPlayers(False)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 8, 4)
        
        #set up example input
        example_input = {}
        for p in players:
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                print(self.nba_games_played)
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8):
                example_input[a] = self.nba_team_arrs[a]
        data_arr = []
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            # Example target (for training purposes)
            key = i
            if (key == 'CHA' and int(self.year) <= 2014):
                key = 'CHA2005'
            # Compute loss
            print(self.nba_team_dict[i], output.detach().numpy()[0][0], self.nba_team_wins[self.nba_team_dict[key]])
            data = {
                "team": self.nba_team_dict[key],
                "predicted_win_rate": str(round(output.detach().numpy()[0][0], 3)),
                "actual_win_rate": str(round(self.nba_team_wins[self.nba_team_dict[key]], 3)),
                "predicted_loss_rate": str(round(1 - output.detach().numpy()[0][0], 3)),
                "actual_loss_rate": str(round(1 - self.nba_team_wins[self.nba_team_dict[key]], 3)),
                "predicted_wins": str(round(round(82 * output.detach().numpy()[0][0], 3))),
                "predicted_losses": str(round(round((82 * (1 - output.detach().numpy()[0][0]))), 3)),
                "actual_wins": str(self.nba_team_wins_count[self.nba_team_dict[key]]),
                "actual_losses": str(self.nba_games_played[self.nba_team_dict[key]] - self.nba_team_wins_count[self.nba_team_dict[key]]),
                "year": str(self.year)
            }
            data_arr.append(data)
        return data_arr
    #train model for x epochs
    def trainForEpochs(epochs):
        for e in range(0, epochs):
            for i in range(2000, 2024):
                if (i % 10 != 3 and i % 10 != 4):
                    p = Performer(str(i))
                    p.performModel()
            if (e % int(epochs / 20) == 0):
                val = int(5 * e / (epochs / 20))
                if (val > 0):
                    print(str(val)+ '%...')
                    test_years = ["2024", "2023", "2014", "2013"]
                    index = random.randint(0, len(test_years) - 1)
                    p = Performer(test_years[index])
                    print(test_years[index], "average accuracy:", p.performModelNoUpdate())
            
#Performer.trainForEpochs(50)
#p = Performer("2024")
#p.showCase()
progress = 0
start_year = 1980
end_year = 2026
player_data = []
#for i in range(2025, 2026):
#    statHandler.saveData(str(i))

for i in range(start_year, end_year):
    s = statHandler(str(i))
    perf = Performer(str(i))
    players = s.calculateTopPlayers(False)
    for p in players:
        key = p.team
        if (key == 'CHA' and i <= 2014):
                key = 'CHA2005'
        data = {
            "name": p.name,
            "team": perf.nba_team_dict[key],
            "scoring": round(p.scoring, 4),
            "playmaking": round(p.playmaking, 4),
            "rebounding": round(p.rebounding, 4),
            "defense": round(p.defensive_win_share_normalized, 4),
            "vorp": round(p.vorp, 4),
            "n_vorp": round(p.normal_vorp, 4),
            "games_played": p.games_played,
            "year": str(i)
        }
        player_data.append(data)
    progress += 1.0

    if (progress % 5 == 0):
        print(f'{round(100 * (progress / (end_year - start_year)), 3)}% complete...')
with open("players.json", "w") as f:
    json.dump(player_data, f)
teams = []
for i in range(2020, end_year):
    perf = Performer(str(i))
    teams.extend(perf.getYearPreds())
with open("teams.json", "w") as f:
    json.dump(teams, f)

