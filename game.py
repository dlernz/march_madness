class Game:

    def __init__(self, game):
        print game
        self.season = game[0]
        self.dayNum = game[1]
        self.wTeamId = game[2]
        self.wScore = game[3]
        self.lTeamId = game[4]
        self.lScore = game[5]
        self.wLoc = game[6]
        self.wFGM = game[8]
        self.wFGA = game[9]
        self.wFGM3 = game[10]
        self.wFGA3 = game[11]
        self.wFTM = game[12]
        self.wFTA = game[13]
        self.wOR = game[14]
        self.wDR = game[15]
        self.wAst = game[16]
        self.wTO = game[17]
        self.wStl = game[18]
        self.wBlk = game[19]
        self.wPF = game[20]
        self.lFGM = game[21]
        self.lFGA = game[22]
        self.lFGM3 = game[23]
        self.lFGA3 = game[24]
        self.lFTM = game[25]
        self.lFTA = game[26]
        self.lOR = game[27]
        self.lDR = game[28]
        self.lAst = game[29]
        self.lTO = game[30]
        self.lStl = game[31]
        self.lBlk = game[32]
        self.lPF = game[33]