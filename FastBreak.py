from urllib.request import urlopen
import numpy as np
import re
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

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
    
    def calcScoring(self):
        if (self.normal_ts == None or self.normal_shot_volume == None):
            return None
        curr_stat = self.normal_ts + (self.vol_modifier * self.normal_shot_volume)
        self.scoring = curr_stat
        return curr_stat

    def calcPlaymaking(self):
        if (self.normal_assist == None or self.normal_assist_volume == None):
            return None
        curr_stat = self.normal_assist + (self.vol_modifier * self.normal_assist_volume)
        self.playmaking = curr_stat
        return curr_stat

    def calcRebounding(self):
        if (self.normal_rebounds == None):
            return None
        curr_stat = self.normal_rebounds + (self.vol_modifier * self.normal_rebounds_volume)
        self.rebounding = curr_stat
        return curr_stat

    def calcVorp(self):
        if(self.defensive_win_share_normalized != None):
            self.vorp = (self.rebound_modifier * self.rebounding) + (self.scoring_modifier * self.scoring) + (self.playmaking_modifier * self.playmaking) + (self.defense_modifier * self.defensive_win_share_normalized)
            return self.vorp
class statHandler():
    def __init__(self, year) -> None:
        self.year = year
        self.players = []
    def getStats(self):
        ###url = "https://www.basketball-reference.com/leagues/NBA_"+ self.year + "_totals.html"
        ##page = urlopen(url)
        ##html_bytes = page.read()
        ##html = html_bytes.decode("utf-8")
        html = open(self.year+"total.html", "r").read()
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
                if(len(stat_list) < 30 or stat_list[4] == 'TOT'):
                    continue
                self.players.append(PlayerStats(stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5], stat_list[6], stat_list[7], stat_list[8], 
                                        stat_list[9], stat_list[10], stat_list[11], stat_list[12], stat_list[13], stat_list[14], stat_list[15], 
                                            stat_list[16], stat_list[17], stat_list[18], stat_list[19], stat_list[20], stat_list[21], stat_list[22], stat_list[23], stat_list[24], 
                                            stat_list[25], stat_list[26], stat_list[27], stat_list[28], stat_list[29], self.year))
                
        ###url = "https://www.basketball-reference.com/leagues/NBA_" + self.year + "_adj_shooting.html"
        ###page = urlopen(url)
        ###html_bytes = page.read()
        ###html = html_bytes.decode("utf-8")
        html = open(self.year+"shooting.html", "r").read()
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
        html = open(self.year+"advanced.html", "r").read()
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



    def calculateTopScorers(self, pr):
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
        if(pr):
            print("TOP 10 SCORERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, str(p.scoring))

    def calculateTopPlayMakers(self, pr):
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
        max_val = self.players[0].playmaking
        for p in self.players:
            p.playmaking /= max_val
        if(pr):
            print()
            print("TOP 10 PLAYMAKERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, str(p.playmaking))

    def calculateTopRebounders(self, pr):
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
        if(pr):
            print()
            print("TOP 10 REBOUNDERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, str(p.rebounding))
    def calculateTopDefenders(self, pr):
        max_val = None
        for p in self.players:
            if (p.defensive_win_shares != 'N/A' and (max_val == None or float(p.defensive_win_shares) > max_val)):
                max_val = float(p.defensive_win_shares)
        for p in self.players:
            if(p.defensive_win_shares != 'N/A'):
                p.defensive_win_share_normalized = float(p.defensive_win_shares) / max_val
        self.players.sort(key=lambda x: x.defensive_win_share_normalized, reverse=True)
        if(pr):
            print()
            print("TOP 10 DEFENDERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, p.defensive_win_share_normalized)
    def calculateTopPlayers(self, pr):
        self.getStats()
        self.calculateTopScorers(pr)
        self.calculateTopPlayMakers(pr)
        self.calculateTopRebounders(pr)
        self.calculateTopDefenders(pr)
        vorpArr = []
        for p in self.players:
            vorpArr.append(p.calcVorp())
        vorpArr = [i for i in vorpArr if i is not None]
        self.players.sort(key=lambda x: x.vorp, reverse=True)
        mean, std = np.mean(vorpArr), np.std(vorpArr)
        for p in self.players:
            p.normal_vorp = p.vorp - mean
            p.normal_vorp /= std
        self.players.sort(key=lambda x: x.vorp, reverse=True)
        if(pr):
            print()
            print("TOP 10 PLAYERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, p.normal_vorp)
        return self.players

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
    def getYearStats(year):
        html = open(year+"teams.html", "r").read()
        player_stats = html.split("</tr>")
        nba_dict = {}
        for i in player_stats:
            stats = i.split("</td>")
            if(len(stats) == 15):
                stat_list = []
                for j in stats:
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)
                    if (len(x) > 0):
                        if (x[0][11:len(x[0]) - 1] == 'win_loss_pct'):
                            if(len(y) > 0):
                                stat_list.append(y[0][1:])
                        elif(len(x) > 1 and x[1][11:len(x[1]) - 1] == 'team_name'):
                            if(len(y) > 0):
                                stat_list.append(y[len(y) - 1][1:])
                if(len(stat_list) == 2):
                    nba_dict[stat_list[0]] = float(stat_list[1])
        return nba_dict
                
        # Save the HTML content to a file
        
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Since we have 7 vectors of 4 elements each, input dimension will be 7 * 4 = 28
        self.input_dim = 8 * 4
        self.hidden_dim1 = 64
        self.hidden_dim2 = 32
        self.output_dim = 1  # Adjust based on your specific problem (e.g., number of classes for classification)

        # Define the layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, self.output_dim)

    def forward(self, x):
        x = x.view(-1, 8 * 4)
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
            'BRK': 'Brooklyn Nets'
        }

        self.nba_team_codes = [
            'ATL', 'BOS', 'BKN', 'CHA', 'CHO', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
            'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 
            'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS', 
            'WSB', 'SEA', 'NJN', 'PHO', 'SDC', 'KCK', 'BRK', 'CHH'
        ]

        # Initialize the dictionary with team codes as keys and 0 as values
        self.nba_team_arrs = {code: [] for code in self.nba_team_codes}
        self.nba_team_wins = statHandler.getYearStats(year)
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
        
        example_input = {}
        for p in players:
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8):
                example_input[a] = self.nba_team_arrs[a]
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            # Example target (for training purposes)
            example_target = torch.full((1, 1), self.nba_team_wins[self.nba_team_dict[i]])  # Example target labels for classification
            # Compute loss
            loss = criterion(output, example_target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model.pth')
    def performModelNoUpdate(self):
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
        
        example_input = {}
        for p in players:
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8):
                example_input[a] = self.nba_team_arrs[a]
        print('RESULTS: ', self.year)
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            print(self.nba_team_dict[i], output)
            # Example target (for training purposes)
            example_target = torch.full((1, 1), self.nba_team_wins[self.nba_team_dict[i]])  # Example target labels for classification
            # Compute loss
            loss = criterion(output, example_target)
            print(f'Loss: {loss.item()}')
            print()
    def trainForEpochs(epochs):
        for e in range(0, epochs):
            for i in range(1984, 1993):
                p = Performer(str(i))
                p.performModel()
            if (e % int(epochs / 10) == 0):
                print(str(int(10 * e / (epochs / 10)))+ '%...')
Performer.trainForEpochs(10)
p = Performer("2024")
p.performModelNoUpdate()
for i in range(2022, 2024):
    statHandler.saveData(str(i))

