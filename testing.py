import pandas as pd 
import numpy as np
import team, game as g
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

def getTeamNames():
    """
    Return dictionary where key is team ID and value is team name
    """
    names = {}
    teams = pd.read_csv("Data/Teams.csv")
    for index, row in teams.iterrows():
        teamId = row["TeamID"]
        name = row["TeamName"]
        names[teamId] = name
    return names

def getSeasonStats(ncaaTourneyTeams):
    """
    Use regular season results and RPI rankings to create a dictionary where key is the team's ID and the value is a Team object. Team objects contain yearly avg stats for each team in various categories
    """
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
    """
    Use NCAA Tournament results to return data frame of matchups where each row contains data for one matchup between two teams, including their yearly avg totals in statistical categories, RPI, and game result.
    """
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
            matchupData["dayNum"] = dayNum
            matchupData["season"] = season
            matchups.append(matchupData)

    df = pd.DataFrame.from_dict(matchups)
    return df

def getMatchupData():
    """
    Returns data frame of historical matchups in NCAA tournament.
    Reads in existing CSV if available. Otherwise, produces data frame by creating Team objects, calculating yearly avg totals for each team, and joining with historical NCAA tourney matchup data
    """
    try:
        matchups = pd.read_csv("Data/output/matchups.csv")
        return matchups
    except Exception as e:
        ncaaTourneyTeams = populateNCAATourneyTeams()
        teamObjs = getSeasonStats(ncaaTourneyTeams)
        matchups = getMatchups(teamObjs)
        matchups.to_csv("Data/output/matchups.csv", index=False)
        return matchups

def findChampionshipMatches():
    """
    Read in NCAA tourney matchups and return data frame containing additional column denoting (True/False) if that matchup was a championship game. 
    """
    matchups = getMatchupData()
    ## group by season and with resulting groupby obj, find whether each row equals the dayNum max for each group
    ## store result as column in matchups defining whether championship played that day
    ## able to pass in functions to transform to perform calculations for each group
    matchups["chipGame"] = matchups.groupby(['season'])['dayNum'].transform(max) == matchups['dayNum']
    return matchups

def getPredictionsChips():
    """
    Output predictions for all championship games from 2003-2017 using a Random Forest classifier. Baseline model takes team with lower RPI as winner. 
    Returns a tuple consisting of a data frame containing the model's prediction for every matchup in our test dataset, the baseline model's accuracy, our model's accuracy
    """
    matchups = findChampionshipMatches()
    matchups["baseline"] = matchups["wRPI"] < matchups["lRPI"]
    cols = list(matchups.columns)

    train = matchups[matchups["chipGame"] == False]
    test = matchups[matchups["chipGame"] == True]
    baselineAcc = 1.0*sum(test["baseline"]) / test.shape[0]
    
    trainLabels = np.array(train["baseline"])
    testLabels = np.array(test["baseline"])
    testNames = np.column_stack((test["lname"], test["l_id"], test["wname"], test["w_id"]))
    # Drop qualitative & output columns
    train = train.drop(["w_id", "l_id", "baseline", "wname", "lname", "season", "dayNum", "chipGame"], axis = 1)
    test = test.drop(["w_id", "l_id", "baseline", "wname", "lname", "season", "dayNum", "chipGame"], axis = 1)
    feature_names = train.columns
    trainFeatures = np.array(train)
    testFeatures = np.array(test)
    maxFeatures = int(len(feature_names)**0.5)

    rf = RandomForestClassifier(n_estimators = 1000, random_state=42, oob_score=True, max_features=maxFeatures)
    rf.fit(trainFeatures, trainLabels)
    ## Draw sample classification tree
    # drawTree(rf, "sampleTree")

    predictions = rf.predict(testFeatures)
    predictProbs = rf.predict_proba(testFeatures)
    modelAcc = 1.0*sum(~(predictions ^ testLabels)) / predictions.shape[0]
    stack = np.column_stack((predictions.T, testLabels.T, testNames[:,0], testNames[:,1], testNames[:,2], testNames[:,3], predictProbs[:,0], predictProbs[:,1]))
    return stack[stack[:,0].argsort()], baselineAcc, modelAcc

### Utilize historical matchup data to build RF model. 
def getPredictions(year, train=None, test=None):
    """
        Output predictions for games from test data set using a Random Forest classifier. Baseline model takes team with lower RPI as winner. 
    Returns a tuple consisting of a data frame containing the model's prediction for every matchup in our test dataset, the baseline model's accuracy, our model's accuracy
    """
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
    train = train.drop(["w_id", "l_id", "baseline", "wname", "lname", "season", "dayNum"], axis = 1)
    test = test.drop(["w_id", "l_id", "baseline", "wname", "lname", "season", "dayNum"], axis = 1)
    feature_names = train.columns
    trainFeatures = np.array(train)
    testFeatures = np.array(test)
    maxFeatures = int(len(feature_names)**0.5)

    rf = RandomForestClassifier(n_estimators = 1000, random_state=42, oob_score=True, max_features=maxFeatures)
    rf.fit(trainFeatures, trainLabels)
    ## Draw sample classification tree
    # drawTree(rf, "sampleTree")

    predictions = rf.predict(testFeatures)
    predictProbs = rf.predict_proba(testFeatures)
    modelAcc = 1.0*sum(~(predictions ^ testLabels)) / predictions.shape[0]
    stack = np.column_stack((predictions.T, testLabels.T, testNames[:,0], testNames[:,1], testNames[:,2], testNames[:,3], predictProbs[:,0], predictProbs[:,1]))
    return stack[stack[:,0].argsort()], baselineAcc, modelAcc

def drawTree(rf, treeName):
    """
    Draws a visual representation of random forest from input classifier and saves as PDF
    """
    dot_data = StringIO()
    export_graphviz(rf.estimators_[0], out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("{}.pdf".format(treeName))


indPredicts = [["Predict", "Actual", "L Name", "L ID", "W Name", "W ID", "Prob For", "Prob Against"]]
baseAccs = []
modelAccs = []
# for i in range(2003, 2005):
#     output, baselineAcc, modelAcc = getPredictions(str(i))
#     baseAccs.append(baselineAcc)
#     modelAccs.append(modelAcc)
#     for row in output:
#         indPredicts.append(row.tolist())
# pd.DataFrame(indPredicts).to_csv("data/output/testResults.csv", index=False, header=False)
# print baseAccs
# print modelAccs

# Chip testing
output, baselineAcc, modelAcc = getPredictionsChips()
print baselineAcc, modelAcc
for row in output:
    indPredicts.append(row.tolist())
pd.DataFrame(indPredicts).to_csv("data/output/chipTestResults.csv", index=False, header=False)
## ran model with 30 percent random rows from matchups: still got 100%, need to look into overfitting issues?