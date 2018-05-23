class Team:

    def __init__(self, _id):
        self._id = _id
        self.name = None
        self.numGamesPlayed = 0
        self.POSS = 0.0
        self.dEff = 0.0
        self.oEff = 0.0
        self.EFG = 0.0
        self.ORB = 0.0
        self.DRB = 0.0
        self.TO = 0.0
        self.TOF = 0.0
        self.FTP = 0.0
        self.FTA = 0.0
        self.MOV = 0.0
        self.MOL = 0.0
        self.winsVsTourney = 0
        self.confTournWins = 0
        self.RPI = None

    def updateStats(self, game, won):
        self.numGamesPlayed += 1
        self.POSS = (self.POSS*(self.numGamesPlayed - 1) + self.getPossessions(game, won)) / self.numGamesPlayed
        if won:
            curOEff = 100.0*game["WScore"] / self.POSS
            self.oEff = (self.oEff*(self.numGamesPlayed - 1) + curOEff) / self.numGamesPlayed
            curDEff = 100.0*game["LScore"] / self.POSS
            self.dEff = (self.dEff*(self.numGamesPlayed - 1) + curDEff) / self.numGamesPlayed
            self.EFG = (game["WFGM"] + (0.5 + game["WFGM3"])) / game["WFGA"]
            self.ORB = (self.ORB*(self.numGamesPlayed - 1) + game["WOR"]) / self.numGamesPlayed
            self.DRB = (self.DRB*(self.numGamesPlayed - 1) + game["WDR"]) / self.numGamesPlayed
            self.TO = (self.TO*(self.numGamesPlayed - 1) + game["WTO"]) / self.numGamesPlayed
            self.TOF = (self.TOF*(self.numGamesPlayed - 1) + game["LTO"]) / self.numGamesPlayed
            curFTP = 1.0*game["WFTM"]/game["WFTA"] if game["WFTA"] != 0 else 0 
            self.FTP = (self.FTP*(self.numGamesPlayed - 1) + curFTP) / self.numGamesPlayed 
            self.FTA = (self.FTA*(self.numGamesPlayed - 1) + game["WFTA"]) / self.numGamesPlayed
            self.MOV = (self.MOV*(self.numGamesPlayed - 1) + abs(game["WScore"] - game["LScore"])) / self.numGamesPlayed
            if game["DayNum"] > 122:
                self.confTournWins += 1

        else:
            curOEff = 100.0*game["LScore"] / self.POSS
            self.oEff = (self.oEff*(self.numGamesPlayed - 1) + curOEff) / self.numGamesPlayed
            curDEff = 100.0*game["WScore"] / self.POSS
            self.dEff = (self.dEff*(self.numGamesPlayed - 1) + curDEff) / self.numGamesPlayed
            self.EFG = (game["LFGM"] + (0.5 + game["LFGM3"])) / game["LFGA"]
            self.ORB = (self.ORB*(self.numGamesPlayed - 1) + game["LOR"]) / self.numGamesPlayed
            self.DRB = (self.DRB*(self.numGamesPlayed - 1) + game["LDR"]) / self.numGamesPlayed
            self.TO = (self.TO*(self.numGamesPlayed - 1) + game["LTO"]) / self.numGamesPlayed
            self.TOF = (self.TOF*(self.numGamesPlayed - 1) + game["WTO"]) / self.numGamesPlayed
            curFTP = 1.0*game["LFTM"]/game["LFTA"] if game["LFTA"] != 0 else 0
            self.FTP = (self.FTP*(self.numGamesPlayed - 1) + curFTP) / self.numGamesPlayed
            self.FTA = (self.FTA*(self.numGamesPlayed - 1) + game["LFTA"]) / self.numGamesPlayed
            self.MOL = (self.MOL*(self.numGamesPlayed - 1) + abs(game["WScore"] - game["LScore"])) / self.numGamesPlayed


    def getPossessions(self, game, won):
        if won:
            return game["WFGA"] - game["WOR"] + game["WTO"] + (0.4 * game["WFTA"])
        else:
            return game["LFGA"] - game["LOR"] + game["LTO"] + (0.4 * game["LFTA"])

    def objToDict(self):
        return self.__dict__