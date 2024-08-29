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
import sys
import matplotlib.pyplot as plt

class PlayerStats():
    #initalize variables
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

        #split table by columns and rows to get elements
        player_stats = html.split("</tr>")
        for i in player_stats:
            stats = i.split("</td>")

            #make sure entry is from the necessary table
            if (len(stats) == 30):
                stat_list = []
                for j in stats:

                    #isolate stat name (x), and stat value (y)
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)

                    #filter values that are invalid
                    for z in y:
                        c = z[1:]
                        if (c != '\n'):
                            stat_list.append(c)

                #if all stats are found, and they have a team (not total stat), add it to list
                if(len(stat_list) < 30 or stat_list[4] == 'TOT'):
                    continue
                #
                self.players.append(PlayerStats(stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5], stat_list[6], stat_list[7], stat_list[8], 
                                        stat_list[9], stat_list[10], stat_list[11], stat_list[12], stat_list[13], stat_list[14], stat_list[15], 
                                            stat_list[16], stat_list[17], stat_list[18], stat_list[19], stat_list[20], stat_list[21], stat_list[22], stat_list[23], stat_list[24], 
                                            stat_list[25], stat_list[26], stat_list[27], stat_list[28], stat_list[29], self.year))
                
        #get the stats for precise shooting
        html = open(self.year+"shooting.html", "r").read()

        #split table by column and row to get elements
        player_stats = html.split("</tr>")
        for ii in range(0, len(player_stats)):
            i = player_stats[ii]
            stats = i.split("</td>")

            #check if the table is correct
            if(len(stats) == 28):
                stat_list = []
                for j in stats:

                    #get stat name (x), and stat value (y)
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)

                    #check if the stat is one we need
                    if (len(x) > 0):
                        if (x[0][11:len(x[0]) - 1] == 'fg_pts_added' 
                            or x[0][11:len(x[0]) - 1] == 'ts_pts_added' 
                            or x[0][11:len(x[0]) - 1] == 'team'):
                            if(len(y) > 0):
                                stat_list.append(y[0][1:])
                        elif(x[0][11:len(x[0]) - 1] == 'ranker'):
                            if(len(y) > 1):
                                stat_list.append(y[1][1:])
                    
                #if all stats are present add it to the correct player
                if(len(stat_list) == 4):
                    for p in self.players:
                        if(p.name == stat_list[0] and p.team == stat_list[1]):
                            p.ts_add = stat_list[3]
                            p.fg_add = stat_list[2]
        
        #get advanced stats
        html = open(self.year+"advanced.html", "r").read()

        #split table by columns and rows into elements
        player_stats = html.split("</tr>")
        for ii in range(0, len(player_stats)):
            i = player_stats[ii]
            stats = i.split("</td>")

            #check if we are in the correct table
            if(len(stats) == 29):
                stat_list = []

                for j in stats:

                    #isolate the stat name (x) and stat value (y)
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)

                    #if the stat is one we need, add it to the list
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

                #if the all stats are found, add it to the appropirate player
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


    #calculate the top scorers
    def calculateTopScorers(self, pr):
        ts_add_arr = []
        
        #transform the 'points added by true shooting above average' stat, into a z-score
        for p in self.players:
            if(p.ts_add != 'N/A'):
                ts_add_arr.append(float(p.ts_add))

        #obtain the mean and standard deviation of the stat
        mean, std = np.mean(ts_add_arr), np.std(ts_add_arr)

        #normalize it
        for p in self.players:
            if(p.ts_add != 'N/A'):
                p.normal_ts = (float(p.ts_add) - mean) / std

        #calculate the volume statistic for scoring
        shot_volume_arr = []
        for p in self.players:
            p.shot_volume = int(p.field_goals_attempted) + (0.44 * int(p.free_throws_attempted))
            shot_volume_arr.append(p.shot_volume)

        #find the mean and standard deviation and then normalize
        mean, std = np.mean(shot_volume_arr), np.std(shot_volume_arr)
        for p in self.players:
            p.normal_shot_volume = (float(p.shot_volume) - mean) / std
        
        #calculate scoring
        for p in self.players:
            p.calcScoring()

        #sort players from best to worst scorers
        self.players.sort(key=lambda x: x.scoring, reverse=True)

        #set value so that top player has score of 1, and rest are percentages
        max_val = self.players[0].scoring
        for p in self.players:
            p.scoring /= max_val

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
            if(p.assist_rate == 'N/A' or p.turnover_rate == 'N/A' or p.usage_percentage == 'N/A'):
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
            volume_arr.append(int(p.assists))
        mean, std = np.mean(volume_arr), np.std(volume_arr)

        #normalize the assist volume
        for p in self.players:
            p.normal_assist_volume = (int(p.assists) - mean) / std
        
        #calculate playmaking score
        for p in self.players:
            p.calcPlaymaking()
        
        #sort players from best to worst playmakers
        self.players.sort(key=lambda x: x.playmaking, reverse=True)

        #set best playmaker to 1, and rest to percentages of the best player
        max_val = self.players[0].playmaking
        for p in self.players:
            p.playmaking /= max_val

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
            if (p.rebound_rate == 'N/A'):
                continue
            reb_rates.append(float(p.rebound_rate))

        #find the mean and standard dev of rebound rates
        mean, std = np.mean(reb_rates), np.std(reb_rates)

        #normalize the rates
        for p in self.players:
            if (p.rebound_rate == 'N/A'):
                continue
            p.normal_rebounds = (float(p.rebound_rate) - mean) / std

        off_rebs, def_rebs = [], []

        #find mean and standard dev of the two volume metrics
        for p in self.players:
            off_rebs.append(int(p.offensive_rebounds))
            def_rebs.append(int(p.defensive_rebounds))
        off_mean, off_std, def_mean, def_std = np.mean(off_rebs), np.std(off_rebs), np.mean(def_rebs), np.std(def_rebs)

        #normalize the volume metrics and combine them
        for p in self.players:
            p.normal_rebounds_volume = 0.5 * (((int(p.offensive_rebounds) - off_mean) / off_std) + ((int(p.defensive_rebounds) - def_mean) / def_std))
        
        #calculate top rebounders
        for p in self.players:
            p.calcRebounding()
            

        #sort best rebounder to worst
        self.players.sort(key=lambda x: x.rebounding, reverse=True)

        #set top player to a score of 1, and the rest to a percentage
        max_val = self.players[0].rebounding
        for p in self.players:
            p.rebounding /= max_val

        #print results if necessary
        if(pr):
            print()
            print("TOP 10 REBOUNDERS")
            for i in range(0, 10):
                p = self.players[i]
                print(p.id, p.name, p.team, str(p.rebounding))
    
    #calculate top defenders
    def calculateTopDefenders(self, pr):
        max_val = None
        #set the defensive win share stat so best player is 1, and the rest are percentages
        for p in self.players:
            if (p.defensive_win_shares != 'N/A' and (max_val == None or float(p.defensive_win_shares) > max_val)):
                max_val = float(p.defensive_win_shares)
        
        for p in self.players:
            if(p.defensive_win_shares != 'N/A'):
                p.defensive_win_share_normalized = float(p.defensive_win_shares) / max_val
        
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
        vorpArr = [i for i in vorpArr if i is not None]

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
        nba_dict = {}

        #split table by rows and columns
        player_stats = html.split("</tr>")
        for i in player_stats:
            stats = i.split("</td>")

            #make sure we are in the right table
            if(len(stats) == 15):
                stat_list = []

                for j in stats:
                    #isolate stat names(x) and stat value (y)
                    x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
                    y = re.findall(">[a-z.A-Z0-9\s'-]+", j)

                    #if the stat is win_loss_pct add it to list
                    if (len(x) > 0):
                        if (x[0][11:len(x[0]) - 1] == 'win_loss_pct'):
                            if(len(y) > 0):
                                stat_list.append(y[0][1:])
                        elif(len(x) > 1 and x[1][11:len(x[1]) - 1] == 'team_name'):
                            if(len(y) > 0):
                                stat_list.append(y[len(y) - 1][1:])
                
                #set dictionary entry if stat was found
                if(len(stat_list) == 2):
                    nba_dict[stat_list[0]] = float(stat_list[1])
        return nba_dict
                
        
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Since we have 8 vectors of 4 elements each, input dimension will be 8 * 4 = 32
        self.input_dim = 8 * 4
        self.hidden_dim1 = 64
        self.hidden_dim2 = 32
        self.output_dim = 1 

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
            'BRK': 'Brooklyn Nets',
            'VAN': 'Vancouver Grizzlies',
            'CHA2005': 'Charlotte Bobcats',
            'NOH': 'New Orleans Hornets',
            'NOK': 'New Orleans',
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
        
        #set up example input by assinging players to the necessary team
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
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32))
        # Forward pass
        for a in self.nba_team_arrs:
            if(len(self.nba_team_arrs[a]) == 8):
                example_input[a] = self.nba_team_arrs[a]
        total_loss = 0
        for i in example_input:
            output = model(torch.tensor(example_input[i]).unsqueeze(0))
            # Example target (for training purposes)
            key = i
            if (key == 'CHA' and int(self.year) <= 2014):
                key = 'CHA2005'
            example_target = torch.full((1, 1), self.nba_team_wins[self.nba_team_dict[key]])  # Example target labels for classification
            # Compute loss
            loss = criterion(output, example_target)
            total_loss += loss.item()
        total_loss /= len(example_input)
        return total_loss

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
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
                self.nba_team_arrs[p.team].append(np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized], dtype=np.float32))
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
        # Each input tensor has the shape (batch_size, 7, 4)
        
        #set up example input
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
                    print("average loss:", p.performModelNoUpdate())
#Performer.trainForEpochs(20)
for i in range(2010, 2024):
    p = Performer(str(i))
    p.showCase()


