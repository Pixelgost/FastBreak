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
import math
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

from StatHandler import StatHandler
from PlayerStats import PlayerStats
from NeuralNetwork import SimpleNN

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
        self.nba_team_wins, self.nba_team_wins_count, self.nba_games_played = StatHandler.getYearStats(year)
        self.year = year


    def performModel(self):
        s = StatHandler(self.year)
        players = s.calculateTopPlayers(False)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        if Path("./models/model.pth").is_file():
            checkpoint = torch.load('./models/model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 7, 4)
        
        #set up example input by assinging players to the necessary team

        example_input = {}
        for p in players:
            if(p.team == "Multiple Teams"):
                continue
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
        }, './models/model.pth')
    
    #show performance of the model without training it further
    def performModelNoUpdate(self):
        s = StatHandler(self.year)
        players = s.calculateTopPlayers(False)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('./models/model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 7, 4)
        
        #set up example input

        example_input = {}
        for p in players:
            if(p.team == "Multiple Teams"):
                continue
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
        s = StatHandler(self.year)
        players = s.calculateTopPlayers(True)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('./models/model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 7, 4)
        
        #set up example input
        example_input = {}
        for p in players:
            if(p.team == "Multiple Teams"):
                continue
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
        s = StatHandler(self.year)
        players = s.calculateTopPlayers(True)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # For classification problems
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('./models/model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 8, 4)
        
        #set up example input
        example_input = {}
        for p in players:
            if(p.team == "Multiple Teams"):
                continue
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
        s = StatHandler(self.year)
        players = s.calculateTopPlayers(False)
        # Instantiate the model
        model = SimpleNN()

        # Define loss function and optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        #load trained model
        checkpoint = torch.load('./models/model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Example input (batch size of 2 for demonstration)
        # Each input tensor has the shape (batch_size, 8, 4)
        
        #set up example input
        example_input = {}
        for p in players:
            if(p.team == "Multiple Teams"):
                continue
            if (p.team not in self.nba_team_arrs):
                self.nba_team_arrs[p.team] = [np.array([p.scoring, p.playmaking, p.rebounding, p.defensive_win_share_normalized, int(p.games_played) / self.nba_games_played[self.nba_team_dict[p.team]]], dtype=np.float32)]
            elif(len(self.nba_team_arrs[p.team]) < 8):
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
        dummy_input = torch.randn(1, 40)  # Adjust this to your model's input size
        torch.onnx.export(
            model, 
            dummy_input, 
            "./fast-break/public/model.onnx",
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
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
            