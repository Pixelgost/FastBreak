from urllib.request import urlopen
import re

class PlayerStats():
    def __init__(self, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, aa, bb, cc, dd):
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
        self.fg_add = 'N/A'
        self.ts_add = 'N/A'
        self.scoring = -999999999999999999999999999999999999999

url = "https://www.basketball-reference.com/leagues/NBA_2024_totals.html"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
player_stats = html.split("</tr>")
players = []
excluded = []
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
            excluded = stat_list[0]
            continue
        players.append(PlayerStats(stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5], stat_list[6], stat_list[7], stat_list[8], 
                                   stat_list[9], stat_list[10], stat_list[11], stat_list[12], stat_list[13], stat_list[14], stat_list[15], 
                                    stat_list[16], stat_list[17], stat_list[18], stat_list[19], stat_list[20], stat_list[21], stat_list[22], stat_list[23], stat_list[24], 
                                    stat_list[25], stat_list[26], stat_list[27], stat_list[28], stat_list[29]))
          
url = "https://www.basketball-reference.com/leagues/NBA_2024_adj_shooting.html"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
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
            for p in players:
                if(p.name == stat_list[0] and p.team == stat_list[1]):
                    p.ts_add = stat_list[3]
                    p.fg_add = stat_list[2]

max = None
max_stat = 0
count = 0
total = 0
countDict = {}
for p in players:
    if (p.name in countDict):
        countDict[p.name] +=1
    else:
        countDict[p.name] = 1
for c in countDict:
    if (countDict[c] > 1):
        for p in players:
            if (p.name == c and p.team != 'TOT'):
                players.remove(p)
        for p in players:
            if (p.name == c and p.team != 'TOT'):
                players.remove(p)
def mergeSort(arr):
    if (len(arr) > 1):
        mid = int(len(arr) / 2)
        a = mergeSort(arr[:mid])
        b = mergeSort(arr[mid:])
        aPointer = 0
        bPointer = 0
        fin = []
        for i in range(0, len(arr)):
            if (aPointer >= len(a)):
                fin.append(b[bPointer])
                bPointer+=1
            elif (bPointer >= len(b)):
                fin.append(a[aPointer])
                aPointer+=1
            elif(a[aPointer].scoring > b[bPointer].scoring):
                fin.append(a[aPointer])
                aPointer+=1
            else:
                fin.append(b[bPointer])
                bPointer +=1
        return fin
    else:
        return arr

def calcScoring(p):
    curr_stat = float(p.ts_add) + (.25 * (int(p.field_goals_attempted) + (.5 * int(p.free_throws_attempted)))) + 0.5 * float(p.fg_add)
    curr_stat /= int(p.games_played)
    return curr_stat

for p in players:
    if (p.ts_add == 'N/A' or p.fg_add == 'N/A'):
        continue
    curr_stat = calcScoring(p)
    count+=1
    total+= curr_stat
offset = -1 * total / count
count = 0
total = 0


for p in players:
    if (p.ts_add == 'N/A' or p.fg_add == 'N/A'):
        continue
    curr_stat = calcScoring(p)
    curr_stat += offset
    p.scoring = curr_stat
sortedScoring = mergeSort(players)
for i in range(0, 10):
    p = sortedScoring[i]
    print(p.id + ' '+ p.name + ' ' + p.team + ' ' + str(p.scoring))
    



