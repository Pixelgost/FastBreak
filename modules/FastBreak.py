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

from StatHandler import StatHandler
from PlayerStats import PlayerStats
from NeuralNetwork import SimpleNN
from Performer import Performer
#Class to calculate all statistics and pull them

                
        


#Performer.trainForEpochs(50)
#p = Performer("2024")
#p.showCase()
progress = 0
start_year = 1980
end_year = 2026
player_data = []
for i in range(2025, 2026):
    StatHandler.saveData(str(i))
for i in range(start_year, end_year):
    s = StatHandler(str(i))
    perf = Performer(str(i))
    players = s.calculateTopPlayers(False)
    for p in players:
        key = p.team
        if (key == 'CHA' and i <= 2014):
                key = 'CHA2005'
        team = key
        if (team != "Multiple Teams"):
            team = perf.nba_team_dict[key]
        data = {
            "name": p.name,
            "team": team,
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
        print(f'{round(100 * (progress / (end_year - start_year)), 2)}% complete...')
with open("./fast-break/public/players.json", "w") as f:
    json.dump(player_data, f)
teams = []
for i in range(2016, end_year):
    perf = Performer(str(i))
    teams.extend(perf.getYearPreds())
with open("./fast-break/public/teams.json", "w") as f:
    json.dump(teams, f)

