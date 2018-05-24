"""
- Conference tournament success (# of wins) 
    - Source: ConferenceTourneyGames.csv
    - Method to count number of wins for each team
- SOS
    - Source: MasseyOrdinals_Prelim2018.csv
    - RPI column -- day closest to 132
- Pace/# of possessions 
    - Source: RegularSeasonDetailedResults.csv
    - Possessions = FGA-OR+TO+.475*FTA
- Defensive Efficiency 
    - Source: RegularSeasonDetailedResults.csv
    - 100*Pts Allowed_a / Possessions_a + 100*Pts Allowed_b / Possessions_b ... / n_games
- Offensive Efficiency 
    - Source: RegularSeasonDetailedResults.csv
    - See above 
- 3P EFG%
    - Source: RegularSeasonDetailedResults.csv
- EFG%
    - Source: RegularSeasonDetailedResults.csv
- Off Rebounding
    - Source: RegularSeasonDetailedResults.csv
- Def Rebounding
    - Source: RegularSeasonDetailedResults.csv
- Turnovers Given
    - Source: RegularSeasonDetailedResults.csv
- Turnovers Forced
    - Source: RegularSeasonDetailedResults.csv
- Free throw percentage
    - Source: RegularSeasonDetailedResults.csv
- Free throws attempted 
    - Source: RegularSeasonDetailedResults.csv
- Margin of Victory
    - Source: RegularSeasonDetailedResults.csv
    - WScore - LScore
- Margin of Loss
    - Source: RegularSeasonDetailedResults.csv
    - WScore - LScore
- # of Wins against Tournament Teams
    - Source: RegularSeasonDetailedResults.csvgit
    - Needs to be calculated --- team plays games past day 132
- Repeated for Opponent^^^

[Matchup 1, M2, M3....] --> Matchup 1: [Team1, Team2]
Get season stats for team1, team2 --> access stored stats for all teams in a season (Doc S)
S has end of season stats listed above for each team for multiple seasons

- Get percentage of win/loss for each possible matchup using random forest with tests data (2018 games)
- Training data is tourney data from previous years 

Winner: Team or Opponent What were tryna classify 
Sample Row Team1Wins, Team1Stats, Team2Stats

### Model Creation
Training Data: team stats at end of season/matchups in tournament (2003-2016)
EndOfSeasonStats: TeamId, Stat1SeasonAvg, Stat2SeasonAvg...StatNSeasonAvg
Matchups: WinTeamId, LTeamId, WScore, LScore Source: "NCAATourneyCompactResults.csv"
Test Data: matchups in 2017 season

Baseline: Choosing matchup based on RPI
"""
import pandas as pd 
import numpy as np
import team, game as g
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

def populateNCAATourneyTeams():
    ncaaTourneyTeams = {}
    ncaaTournResults = pd.read_csv("data/NCAATourneyCompactResults.csv")
    for index, row in ncaaTournResults.iterrows():
        season = row["Season"]
        dayNum = row["DayNum"]
        wTeamId = row["WTeamID"]
        lTeamId = row["LTeamID"]
        customWId = str(wTeamId) + "_" + str(season)
        customLId = str(lTeamId) + "_" + str(season)

        if customWId not in ncaaTourneyTeams:
            ncaaTourneyTeams[customWId] = 1
        if customLId not in ncaaTourneyTeams:
            ncaaTourneyTeams[customLId] = 1
    return ncaaTourneyTeams

def getSeasonStats(ncaaTourneyTeams):
    teams = {}
    names = getTeamNames()
    unfiltRanks = pd.read_csv("data/MasseyOrdinals_Prelim2018.csv")
    rankings = unfiltRanks[(unfiltRanks["SystemName"] == "RPI") & (unfiltRanks["RankingDayNum"] == 133)]
    regSeasonResults = pd.read_csv("data/RegularSeasonDetailedResults.csv")
    for index, row in regSeasonResults.iterrows():
        season = row["Season"]
        dayNum = row["DayNum"]
        wTeamId = row["WTeamID"]
        lTeamId = row["LTeamID"]
        customWId = str(wTeamId) + "_" + str(season)
        customLId = str(lTeamId) + "_" + str(season)
        wRPI = None
        lRPI = None
        try:
            wRPI = rankings[(rankings["Season"] == season) & (rankings["TeamID"] == wTeamId)].iloc[0]["OrdinalRank"]
            lRPI = rankings[(rankings["Season"] == season) & (rankings["TeamID"] == lTeamId)].iloc[0]["OrdinalRank"]
        except Exception as e:
            pass
            # print str(lTeamId) + " " + str(season) + " not found"
        
        if customWId not in teams:
            teams[customWId] = team.Team(customWId)
        if customLId not in teams:
            teams[customLId] = team.Team(customLId)
        wTeam = teams[customWId]
        wTeam.RPI = wRPI
        wTeam.name = names[wTeamId]
        wTeam.updateStats(row, True)
        if customLId in ncaaTourneyTeams:
            wTeam.winsVsTourney += 1
        lTeam = teams[customLId]
        lTeam.name = names[lTeamId]
        lTeam.RPI = lRPI
        lTeam.updateStats(row, False)
    return teams

def getMatchups(teams):
    matchups = []
    ncaaTournResults = pd.read_csv("data/NCAATourneyCompactResults.csv")
    for index, row in ncaaTournResults.iterrows():
        season = row["Season"]
        dayNum = row["DayNum"]
        wTeamId = row["WTeamID"]
        lTeamId = row["LTeamID"]
        customWId = str(wTeamId) + "_" + str(season)
        customLId = str(lTeamId) + "_" + str(season)

        if customWId in teams and customLId in teams:
            wTeamData = teams[customWId].objToDict().copy()
            for key in wTeamData.keys():
                wTeamData["w" + key] = wTeamData[key]
                del wTeamData[key]
            lTeamData = teams[customLId].objToDict().copy()
            for key in lTeamData.keys():
                lTeamData["l" + key] = lTeamData[key]
                del lTeamData[key]
            matchupData = wTeamData.copy()
            matchupData.update(lTeamData)
            matchupData.update(winnerCol)
            matchups.append(matchupData)

    df = pd.DataFrame.from_dict(matchups)
    return df

def getMatchupData():
    try:
        matchups = pd.read_csv("Data/output/matchups.csv")
        return matchups
    except Exception as e:
        ncaaTourneyTeams = populateNCAATourneyTeams()
        teamObjs = getSeasonStats(ncaaTourneyTeams)
        matchups = getMatchups(teamObjs)
        matchups.to_csv("Data/output/matchups.csv", index=False)
        return matchups

def getTeamNames():
    names = {}
    teams = pd.read_csv("Data/Teams.csv")
    for index, row in teams.iterrows():
        teamId = row["TeamID"]
        name = row["TeamName"]
        names[teamId] = name
    return names

### Utilize historical matchup data to build RF model. Output accuracy of model
### for input year
def getPredictions(year):
    matchups = getMatchupData()
    matchups["baseline"] = matchups["wRPI"] < matchups["lRPI"]

    cols = list(matchups.columns)
    train = matchups[~matchups["w_id"].str.contains(year)]
    test = matchups[matchups["w_id"].str.contains(year)]
    baselineAcc = 1.0*sum(test["baseline"]) / test.shape[0]

    trainLabels = np.array(train["baseline"])
    testLabels = np.array(test["baseline"])
    testNames = np.column_stack((test["lname"], test["l_id"], test["wname"], test["w_id"]))
    # Drop qualitative & output columns
    train = train.drop(["w_id", "l_id", "baseline", "wname", "lname"], axis = 1)
    test = test.drop(["w_id", "l_id", "baseline", "wname", "lname"], axis = 1)
    feature_names = train.columns
    trainFeatures = np.array(train)
    testFeatures = np.array(test)

    rf = RandomForestClassifier(n_estimators = 1000, random_state=32, oob_score=True)
    rf.fit(trainFeatures, trainLabels)
    ### Draw sample classification tree
    # dot_data = StringIO()
    # export_graphviz(rf.estimators_[0], out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_names)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf("tree.pdf")

    predictions = rf.predict(testFeatures)
    predictProbs = rf.predict_proba(testFeatures)
    modelAcc = 1.0*sum(~(predictions ^ testLabels)) / predictions.shape[0]
    # stack = np.column_stack((predictions.T, testLabels.T, testNames[:,0], testNames[:,1], testNames[:,2], testNames[:,3], predictProbs[:,0], predictProbs[:,1]))
    # print stack[stack[:,0].argsort()]
    # return stack[stack[:,0].argsort()]
    return baselineAcc, modelAcc

# indPredicts = [["Predict", "Actual", "L Name", "L ID", "W Name", "W ID", "Prob For", "Prob Against"]]
baseAccs = []
modelAccs = []
for i in range(2003, 2018):
    output = getPredictions(str(i))
    baseAccs.append(output[0])
    modelAccs.append(output[1])
    # for row in output:
    #     indPredicts.append(row.tolist())
# pd.DataFrame(indPredicts).to_csv("data/output/testResults.csv", index=False)
print baseAccs
print modelAccs
