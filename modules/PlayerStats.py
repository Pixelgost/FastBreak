import sys
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
