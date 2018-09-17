# march_madness

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