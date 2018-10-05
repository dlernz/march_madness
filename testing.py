import pandas as pd 
import numpy as np
import team, game as g
from sklearn.ensemble import RandomForestClassifier
# Used for developing visual of Random Forest if desired
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
    Use regular season results and RPI rankings to create a 
    dictionary where key is the team's ID and the value is a 
    Team object. Team objects contain yearly avg stats for each 
    team in various categories.
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

def populateNCAATourneyTeams():
    """
    Create an ID for each team using a combination of its id and the season the team played in. 
    Output a dictionary with an entry for each team whose key is its newly created id
    """
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
            wTeamDataCopy = wTeamData.copy()
            for key in wTeamData.keys():
                wTeamDataCopy["w" + key] = wTeamData[key]
                del wTeamDataCopy[key]
            lTeamData = teams[customLId].objToDict().copy()
            lTeamDataCopy = lTeamData.copy()
            for key in lTeamData.keys():
                lTeamDataCopy["l" + key] = lTeamData[key]
                del lTeamDataCopy[key]
            matchupData = wTeamDataCopy.copy()
            matchupData.update(lTeamDataCopy)
            matchupData["dayNum"] = dayNum
            matchupData["season"] = season
            matchups.append(matchupData)
    colOrder = ["dayNum", "season", "l_id", "lname", "w_id", "wname", "lDRB", "lEFG", "lFTA", "lFTP", "lMOL", "lMOV", "lORB", "lPOSS",
                "lRPI", "lTO", "lTOF", "lconfTournWins", "ldEff", "lnumGamesPlayed", "loEff", "lwinsVsTourney",
                "wDRB", "wEFG", "wFTA", "wFTP", "wMOL", "wMOV", "wORB", "wPOSS", "wRPI", "wTO", "wTOF", 
                "wconfTournWins", "wdEff", "wnumGamesPlayed", "woEff", "wwinsVsTourney"]
    df = pd.DataFrame.from_dict(matchups)
    df = df[colOrder]
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
    matchups = sortMatchups(matchups)
    ## group by season and with resulting groupby obj, find whether each row equals the dayNum max for each group
    ## store result as column in matchups defining whether championship played that day
    ## able to pass in functions to transform to perform calculations for each group
    matchups["chipGame"] = matchups.groupby(['season'])['dayNum'].transform(max) == matchups['dayNum']
    return matchups

def getPredictionsChips():
    """
    Outputs predictions for all championship games from 2003-2017 using a Random Forest classifier. Baseline model takes team with lower RPI as winner. 
    Returns a tuple consisting of a data frame containing the model's prediction for every matchup in our test dataset, the baseline model's accuracy, our model's accuracy
    """
    matchups = findChampionshipMatches()
    cols = list(matchups.columns)
    
    # Create training/test data sets
    train = matchups[matchups["chipGame"] == False]
    test = matchups[matchups["chipGame"] == True]
    baselineAcc = 1.0*sum(test["baseline"]) / test.shape[0]
    
    trainLabels = np.array(train["baseline"])
    trainLabels = trainLabels.astype(int)
    testLabels = np.array(test["baseline"])
    testLabels = testLabels.astype(int)
    testNames = np.column_stack((test["aname"], test["a_id"], test["bname"], test["b_id"]))
    # Drop qualitative & output columns
    train = train.drop(["b_id", "a_id", "baseline", "bname", "aname", "season", "dayNum", "chipGame"], axis = 1)
    test = test.drop(["b_id", "a_id", "baseline", "bname", "aname", "season", "dayNum", "chipGame"], axis = 1)
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
    modelAcc = 1.0*(predictions.shape[0] - sum(predictions ^ testLabels)) / predictions.shape[0]
    stack = np.column_stack((predictions.T, testLabels.T, testNames[:,0], testNames[:,1], testNames[:,2], testNames[:,3], predictProbs[:,0], predictProbs[:,1]))
    outputPreds = stack[stack[:,0].argsort()]
    return outputPreds, baselineAcc, modelAcc

### Utilize historical matchup data to build RF model. 
def getPredictions(year, train=None, test=None):
    """
    Outputs predictions for games from test data set using a Random Forest classifier. Baseline model takes team with lower RPI as winner. 
    Returns a tuple consisting of a data frame containing the model's prediction for every matchup in our test dataset, the baseline model's accuracy, our model's accuracy
    """
    matchups = getMatchupData()
    matchups = sortMatchups(matchups)
    cols = list(matchups.columns)
    train = matchups[~matchups["b_id"].str.contains(year)]
    test = matchups[matchups["b_id"].str.contains(year)]
    baselineAcc = 1.0*sum(test["baseline"]) / test.shape[0]
    
    # Create training/test data sets
    trainLabels = np.array(train["baseline"])
    trainLabels = trainLabels.astype(int)
    testLabels = np.array(test["baseline"])
    testLabels = testLabels.astype(int)
    testNames = np.column_stack((test["aname"], test["a_id"], test["bname"], test["b_id"]))
    # Drop qualitative & output columns
    train = train.drop(["b_id", "a_id", "baseline", "bname", "aname", "season", "dayNum"], axis = 1)
    test = test.drop(["b_id", "a_id", "baseline", "bname", "aname", "season", "dayNum"], axis = 1)
    
    train = train.drop(["aMOV", "aMOL", "aFTA", "aFTP", "anumGamesPlayed", "aRPI", "bMOV", "bMOL", "bFTA", "bFTP", "bnumGamesPlayed", "bRPI"], axis = 1)
    test = test.drop(["aMOV", "aMOL", "aFTA", "aFTP", "anumGamesPlayed", "aRPI", "bMOV", "bMOL", "bFTA", "bFTP", "bnumGamesPlayed", "bRPI"], axis = 1)
    feature_names = train.columns
    trainFeatures = np.array(train)
    testFeatures = np.array(test)
    maxFeatures = int(len(feature_names)**0.5)

    rf = RandomForestClassifier(n_estimators = 1000, random_state=42, oob_score=True, max_features=maxFeatures)
    rf.fit(trainFeatures, trainLabels)
    
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index = train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    
    print(feature_importances)

    predictions = rf.predict(testFeatures)
    predictProbs = rf.predict_proba(testFeatures)
    modelAcc = 1.0*(predictions.shape[0] - sum(predictions ^ testLabels)) / predictions.shape[0]
    stack = np.column_stack((predictions.T, testLabels.T, testNames[:,0], testNames[:,1], testNames[:,2], testNames[:,3], predictProbs[:,0], predictProbs[:,1]))
    outputPreds = stack[stack[:,0].argsort()]
    return outputPreds, baselineAcc, modelAcc

def drawTree(rf, treeName):
    dot_data = StringIO()
    export_graphviz(rf.estimators_[0], out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("{}.pdf".format(treeName))
    
    
def sortRowByRPI(matchup):
    if matchup["lRPI"] > matchup["wRPI"]:
        numCols = matchup.shape[0]
        newOrder = [0, 1, 4, 5, 2, 3] + list(range(22, numCols - 1)) + list(range(6,22)) + [numCols - 1]
        matchup = matchup[matchup.index[newOrder]]
        return list(matchup.values)
    return list(matchup.values)

def findDeltasForMatch(matchup):
    print(matchup.iloc[1:12])

def sortMatchups(matchups):
    matchups["baseline"] = matchups["wRPI"] < matchups["lRPI"]
    matchups["baseline"].replace(False, 0, inplace=True)
    matchups["baseline"].replace(True, 1, inplace=True)
    sortedMatchups = matchups.apply(sortRowByRPI, axis = 1, result_type = "broadcast")

    newColNames = []
    columns = sortedMatchups.columns
    for name in columns:
        if name[0] == 'l':
            newColNames.append("a" + name[1:])
        elif name[0] == "w":
            newColNames.append("b" + name[1:])
        else:
            newColNames.append(name)
    sortedMatchups.columns = newColNames
    return sortedMatchups
    
matchups = getMatchupData()
print (sortMatchups(matchups))