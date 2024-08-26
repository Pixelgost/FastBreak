from urllib.request import urlopen
import re

class PlayerStats():
    def __init__(self, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z):
        self.id = a
        self.name = b
        self.position = c
        self.age = d
        self.team = e
        self.games_played = f
        self.per = g
        self.ts_pct = h
        self.fg3a_per_fga_pct = i
        self.fta_per_fga_pct = j
        self.orb_pct  = k
        self.drb_pct = l
        self.trb_pct = m
        self.ast_pct = n
        self.stl_pct = o
        self.blk_pct = p
        self.tov_pct = q
        self.usg_pct = r
        self.ows = s
        self.dws = t
        self.ws = u
        self.ws_per_48 = v
        self.obpm = w
        self.dbpm = x
        self.bpm =y
        self.vorp = z
        self.fg_add = 'N/A'
        self.ts_add = 'N/A'
        self.scoring = -999999999999999999999999999999999999999

url = "https://www.basketball-reference.com/leagues/NBA_2024_advanced.html"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
player_stats = html.split("</tr>")
players = []
excluded = []
for i in player_stats:
    stats = i.split("</td>")
    if (len(stats) == 29):
        stat_list = []
        for j in stats:
            x = re.findall("data-stat=\"[a-zA-Z0-9_]+\"", j)
            y = re.findall(">[a-z.A-Z0-9\s'-]+", j)
            for z in y:
                c = z[1:]
                if (c != '\n'):
                    stat_list.append(c)
        if(len(stat_list) < 26):
            excluded = stat_list[0]
            continue
        players.append(PlayerStats(stat_list[0], stat_list[1], stat_list[2], stat_list[3], stat_list[4], stat_list[5], stat_list[6], stat_list[7], stat_list[8], 
                                   stat_list[9], stat_list[10], stat_list[11], stat_list[12], stat_list[13], stat_list[14], stat_list[15], 
                                    stat_list[16], stat_list[17], stat_list[18], stat_list[19], stat_list[20], stat_list[21], stat_list[22], stat_list[23], stat_list[24], stat_list[25]))
          
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
for p in players:
    if (p.ts_add == 'N/A' or p.fg_add == 'N/A'):
        continue
    curr_stat = (0.5 * float(p.ts_add)) + float(p.fg_add)
    curr_stat /= int(p.games_played)
    count+=1
    total+= curr_stat
offset = -1 * total / count
count = 0
total = 0

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

for p in players:
    if (p.ts_add == 'N/A' or p.fg_add == 'N/A'):
        continue
    curr_stat = (0.5 * float(p.ts_add)) + float(p.fg_add)
    curr_stat /= int(p.games_played)
    curr_stat += offset
    p.scoring = curr_stat
sortedScoring = mergeSort(players)
for p in sortedScoring:
    print(p.id + ' '+ p.name + ' ' + p.team + ' ' + str(p.scoring))
    



