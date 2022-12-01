from numpy import uint8, uint16, float16, float32
from bokeh.palettes import Category20

position_map = {
    'ALL': 'Overall',
    'GK': 'Goalkeeper',
    'CB': 'Centre-Back',
    'FB': 'Full-Back',
    'DM': 'Defensive Midfield',
    'CM': 'Central Midfield',
    'AM': 'Attacking Midfield',
    'WI': 'Winger',
    'FW': 'Forward'
}

positions = [
    'GK',
    'CB',
    'FB',
    'DM',
    'CM',
    'AM',
    'WI',
    'FW'
]

distributions = [
    'ALL',
    'GK',
    'CB',
    'FB',
    'DM',
    'CM',
    'AM',
    'WI',
    'FW'
]

colours = list(Category20[9])

distribution_colours = ['black'] + list(Category20[9])

category_map = {
    'Standard Stats': 'stats',
    'Goalkeeping': 'keepers',
    'Advanced Goalkeeping': 'keepersadv',
    'Shooting': 'shooting',
    'Passing': 'passing',
    'Pass Types': 'passing_types',
    'Goal and Shot Creation': 'gca',
    'Defensive Actions': 'defense',
    'Possession': 'possession',
    'Playing Time': 'playingtime',
    'Miscellaneous Stats': 'misc'
}

not_per_min = [
    'Min',
    'CS%',
    'Save%',
    'Save%.1',
    'PSxG/SoT',
    'CS%',
    'Cmp%',
    'Cmp%.1',
    'Cmp%.2',
    'Cmp%.3',
    'Launch%',
    'Launch%.1',
    'Stp%', 'G/Sh',
    'G/Sh',
    'npxG/Sh',
    'Tkl%',
    '%',
    'Succ%',
    'Rec%',
    'Min%',
    'PPM',
    'Won%',
    'Age',
    'Value',
    'GkMin'
]

axis_maps = dict(
    stats={
        'Matches Played': 'MP',
        'Age': 'Age',
        'Starts': 'Starts',
        'Minutes': 'Min',
        'Value': 'Value',
        'Minutes / 90': '90s',
        'Goals': 'Gls',
        'Assists': 'Ast',
        'Goals - Penalties': 'G-PK',
        'Goals + Assists - Penalties': 'G+A-PK',
        'Penalties': 'PK',
        'Penalties Attempted': 'PKatt',
        'Yellow Cards': 'CrdY',
        'Red Cards': 'CrdR',
        'xG': 'xG',
        'npxG': 'npxG',
        'xAG': 'xAG',
        'npxG+xA': 'npxG+xAG'
    },
    keepers={
        'Goals Against': 'GA',
        'Goalkeeper Minutes': 'GkMin',
        'Shots on Target Against': 'SoTA',
        'Save %': 'Save%',
        'Clean Sheets': 'CS',
        'Clean Sheet %': 'CS%',
        'Penalties Conceded': 'PKA',
        'Penalties Saved': 'PKsv',
        'Penalties Wide': 'PKm',
        'Penalty Save %': 'Save%.1',
    },
    keepersadv={
        'Post Shot xG': 'PSxG',
        'Post Shot xG / SoT': 'PSxG/SoT',
        'Post Shot xG +/-': 'PSxG+/-',
        'Launched Passes Completed ': 'Cmp',
        'Launched Passes Attempted ': 'Att',
        'Launched Pass Success %': 'Cmp%',
        'Passes Attempted ': 'Att.1',
        'Throws Attempted': 'Thr',
        'Launched Pass Selection %': 'Launch%',
        'Pass Average Length': 'AvgLen',
        'Goal Kicks Attempted ': 'Att.2',
        'Goal Kicks Launched %': 'Launch%.1',
        'Goal Kick Average Length': 'AvgLen.1',
        'Opponent Crosses': 'Opp',
        'Opponent Crosses Stopped': 'Stp',
        'Opponent Cross Stop %': 'Stp%',
        'Defensive Actions Outside Penalty Area': '#OPA',
        'Average Distance of Defensive Actions': 'AvgDist'
    },
    shooting={
        'Goals': 'Gls',
        'Shots': 'Sh',
        'Shots on Target': 'SoT',
        'Shots on Target %': 'SoT%',
        'Goals / Shot': 'G/Sh',
        'Goals / Shot on Target': 'G/SoT',
        'Average Distance of Shot': 'Dist',
        'Free Kick Shots': 'FK',
        'Penalties Scored': 'PK',
        'Penalties Attempted': 'PKatt',
        'xG': 'xG',
        'npxG': 'npxG',
        'npxG / Shot': 'npxG/Sh',
        'Goals - xG': 'G-xG',
        'Non Penalty Goals - npxG': 'np:G-xG'
    },
    passing={
        'Passes Completed': 'Cmp',
        'Passes Attempted ': 'Att',
        'Pass Success %': 'Cmp%',
        'Total Distance from Passing': 'TotDist',
        'Progressive Distance from Passing': 'PrgDist',
        'Short Passes Completed': 'Cmp.1',
        'Short Passes Attempted': 'Att.1',
        'Short Pass Success %': 'Cmp%.1',
        'Medium Passes Completed': 'Cmp.2',
        'Medium Passes Attempted': 'Att.2',
        'Medium Pass Success %': 'Cmp%.2',
        'Long Passes Completed': 'Cmp.3',
        'Long Passes Attempted': 'Att.3',
        'Long Pass Success %': 'Cmp%.3',
        'Assists': 'Ast',
        'xA': 'xA',
        'xAG': 'xAG',
        'Assists - xA': 'A-xAG',
        'Key Passes': 'KP',
        'Final Third Passes Completed': '1/3',
        '18 Yard Box Passes Completed': 'PPA',
        '18 Yard Box Crosses Completed': 'CrsPA',
        'Progressive Passes Completed': 'Prog'
    },
    passing_types={
        'Live Passes': 'Live',
        'Dead Passes': 'Dead',
        'Free Kick Passes Attempted': 'FK',
        'Through Balls Completed': 'TB',
        'Switch Passes ': 'Sw',
        'Crosses': 'Crs',
        'Throw Ins': 'TI',
        'Corners': 'CK',
        'In Swinging Corners': 'In',
        'Out Swinging Corners': 'Out',
        'Straight Corners': 'Str',
        'Passes Completed': 'Cmp',
        'Passes Offside': 'Off',
        'Passes Blocked': 'Blocks'
    },
    gca={
        'Shot Creating Actions': 'SCA',
        'Live Balls Leading to Shot': 'PassLive',
        'Dead Balls Leading to Shot': 'PassDead',
        'Dribles Leading to Shot': 'Drib',
        'Shot Leading to Shot': 'Sh',
        'Fouled Leading to Shot': 'Fld',
        'Defensive Action Leading to Shot': 'Def',
        'Goal Creating Actions': 'GCA',
        'Live Balls Leading to Goal': 'PassLive.1',
        'Dead Balls Leading to Goal': 'PassDead.1',
        'Dribles Leading to Goal': 'Drib.1',
        'Shot Leading to Goal': 'Sh.1',
        'Fouled Leading to Goal': 'Fld.1',
        'Defensive Action Leading to Goal': 'Def.1'
    },
    defense={
        'Tackles Attempted': 'Tkl',
        'Tackles Won': 'TklW',
        'Defensive Third Tackles': 'Def 3rd',
        'Middle Third Tackles': 'Mid 3rd',
        'Attacking Third Tackles': 'Att 3rd',
        'Dribbler Tackles Completed': 'Tkl.1',
        'Dribbler Tackles Attempted': 'Att',
        'Dribbler Tackle Success %': 'Tkl%',
        'Times Dribbled Past': 'Past',
        'Blocks': 'Blocks',
        'Shots Blocked': 'Sh',
        'Passes Blocked': 'Pass',
        'Interceptions': 'Int',
        'Tackles + Interceptions': 'Tkl+Int',
        'Clearances': 'Clr',
        'Errors Leading to Shot': 'Err'},
    possession={
        'Total Touches': 'Touches',
        'Own Penalty Area Touches': 'Def Pen',
        'Defensive Third Touches': 'Def 3rd',
        'Middle Third Touches': 'Mid 3rd',
        'Attacking Third Touches': 'Att 3rd',
        'Opposition Penalty Area Touches': 'Att Pen',
        'Live Ball Touches': 'Live',
        'Dribbles Completed': 'Succ',
        'Dribbles Attempted': 'Att',
        'Dribble Success %': 'Succ%',
        'Miscontrols': 'Mis',
        'Dispossessed': 'Dis',
        'Passes Received': 'Rec',
        'Progressive Passes Received': 'Prog'
    },
    playingtime={
        'Matches Played': 'MP',
        'Minutes Played': 'Min%',
        'Matches Completed': 'Compl',
        'Games as Substitute': 'Subs',
        'Games as Unused Substitute': 'unSub',
        'Points Per Match': 'PPM',
        'Team Goals While on Pitch': 'onG',
        'Team Goals Against While on Pitch': 'onGA',
        'Team Goals +/- While on Pitch': '+/-',
        'Team xG While on Pitch': 'onxG',
        'Team xG Against While on Pitch': 'onxGA',
        'Team xG +/- While on Pitch': 'xG+/-'
    },
    misc={
        'Yellow Cards': 'CrdY',
        'Red Cards': 'CrdR',
        'Two Yellow Cards in a Game': '2CrdY',
        'Fouls': 'Fls',
        'Fouled': 'Fld',
        'Offsides': 'Off',
        'Crosses': 'Crs',
        'Interceptions': 'Int',
        'Tackles Won': 'TklW',
        'Penalties Won': 'PKwon',
        'Penalties Conceded': 'PKcon',
        'Aerials Won': 'Won',
        'Aerials Lost': 'Lost',
        'Aerial Success %': 'Won%'
    }
)

data_types = {
    ('stats', 'Player'): str,
    ('stats', 'Nation'): str,
    ('stats', 'Pos'): str,
    ('stats', 'Squad'): str,
    ('stats', 'Age'): uint8,
    ('stats', 'MP'): uint16,
    ('stats', 'Starts'): uint16,
    ('stats', 'Min'): uint16,
    ('stats', 'Value'): float16,
    ('stats', 'Gls'): uint16,
    ('stats', 'Ast'): uint16,
    ('stats', 'G-PK'): uint16,
    ('stats', 'G+A-PK'): float16,
    ('stats', 'PK'): uint16,
    ('stats', 'PKatt'): uint16,
    ('stats', 'CrdY'): uint16,
    ('stats', 'CrdR'): uint16,
    ('stats', 'xG'): float16,
    ('stats', 'npxG'): float16,
    ('stats', 'xAG'): float16,
    ('stats', 'npxG+xAG'): float16,
    ('stats', 'Season'): str,
    ('stats', 'Team'): str,
    ('stats', 'short'): str,
    ('keepers', 'GA'): uint16,
    ('keepers', 'SoTA'): uint16,
    ('keepers', 'Save%'): float16,
    ('keepers', 'CS'): uint16,
    ('keepers', 'CS%'): float16,
    ('keepers', 'PKA'): uint16,
    ('keepers', 'PKsv'): uint16,
    ('keepers', 'PKm'): uint16,
    ('keepers', 'Save%.1'): float16,
    ('keepers', 'GkMin'): float16,
    ('keepersadv', 'PSxG'): float16,
    ('keepersadv', 'PSxG/SoT'): float16,
    ('keepersadv', 'PSxG+/-'): float16,
    ('keepersadv', 'Cmp'): uint16,
    ('keepersadv', 'Att'): uint16,
    ('keepersadv', 'Cmp%'): float16,
    ('keepersadv', 'Att.1'): uint16,
    ('keepersadv', 'Thr'): uint16,
    ('keepersadv', 'Launch%'): float16,
    ('keepersadv', 'AvgLen'): float16,
    ('keepersadv', 'Att.2'): uint16,
    ('keepersadv', 'Launch%.1'): float16,
    ('keepersadv', 'AvgLen.1'): float16,
    ('keepersadv', 'Opp'): uint16,
    ('keepersadv', 'Stp'): uint16,
    ('keepersadv', 'Stp%'): float16,
    ('keepersadv', '#OPA'): uint16,
    ('keepersadv', 'AvgDist'): float16,
    ('shooting', 'Gls'): uint16,
    ('shooting', 'Sh'): uint16,
    ('shooting', 'SoT'): uint16,
    ('shooting', 'SoT%'): float16,
    ('shooting', 'G/Sh'): float16,
    ('shooting', 'G/SoT'): float16,
    ('shooting', 'Dist'): float16,
    ('shooting', 'FK'): uint16,
    ('shooting', 'PK'): uint16,
    ('shooting', 'PKatt'): uint16,
    ('shooting', 'xG'): float16,
    ('shooting', 'npxG'): float16,
    ('shooting', 'npxG/Sh'): float16,
    ('shooting', 'G-xG'): float16,
    ('shooting', 'np:G-xG'): float16,
    ('passing', 'Cmp'): uint16,
    ('passing', 'Att'): uint16,
    ('passing', 'Cmp%'): float16,
    ('passing', 'TotDist'): float32,
    ('passing', 'PrgDist'): float32,
    ('passing', 'Cmp.1'): uint16,
    ('passing', 'Att.1'): uint16,
    ('passing', 'Cmp%.1'): float16,
    ('passing', 'Cmp.2'): uint16,
    ('passing', 'Att.2'): uint16,
    ('passing', 'Cmp%.2'): float16,
    ('passing', 'Cmp.3'): uint16,
    ('passing', 'Att.3'): uint16,
    ('passing', 'Cmp%.3'): float16,
    ('passing', 'Ast'): uint16,
    ('passing', 'xA'): float16,
    ('passing', 'xAG'): float16,
    ('passing', 'A-xAG'): float16,
    ('passing', 'KP'): uint16,
    ('passing', '1/3'): uint16,
    ('passing', 'PPA'): uint16,
    ('passing', 'CrsPA'): uint16,
    ('passing', 'Prog'): uint16,
    ('passing_types', 'Live'): uint16,
    ('passing_types', 'Dead'): uint16,
    ('passing_types', 'FK'): uint16,
    ('passing_types', 'TB'): uint16,
    ('passing_types', 'Sw'): uint16,
    ('passing_types', 'Crs'): uint16,
    ('passing_types', 'TI'): uint16,
    ('passing_types', 'CK'): uint16,
    ('passing_types', 'In'): uint16,
    ('passing_types', 'Out'): uint16,
    ('passing_types', 'Str'): uint16,
    ('passing_types', 'Cmp'): uint16,
    ('passing_types', 'Off'): uint16,
    ('passing_types', 'Blocks'): uint16,
    ('gca', 'SCA'): uint16,
    ('gca', 'PassLive'): uint16,
    ('gca', 'PassDead'): uint16,
    ('gca', 'Drib'): uint16,
    ('gca', 'Sh'): uint16,
    ('gca', 'Fld'): uint16,
    ('gca', 'Def'): uint16,
    ('gca', 'GCA'): uint16,
    ('gca', 'PassLive.1'): uint16,
    ('gca', 'PassDead.1'): uint16,
    ('gca', 'Drib.1'): uint16,
    ('gca', 'Sh.1'): uint16,
    ('gca', 'Fld.1'): uint16,
    ('gca', 'Def.1'): uint16,
    ('defense', 'Tkl'): uint16,
    ('defense', 'TklW'): uint16,
    ('defense', 'Def 3rd'): uint16,
    ('defense', 'Mid 3rd'): uint16,
    ('defense', 'Att 3rd'): uint16,
    ('defense', 'Tkl.1'): uint16,
    ('defense', 'Att'): uint16,
    ('defense', 'Tkl%'): float16,
    ('defense', 'Past'): uint16,
    ('defense', 'Blocks'): uint16,
    ('defense', 'Sh'): uint16,
    ('defense', 'Pass'): uint16,
    ('defense', 'Int'): uint16,
    ('defense', 'Tkl+Int'): uint16,
    ('defense', 'Clr'): uint16,
    ('defense', 'Err'): uint16,
    ('possession', 'Touches'): uint16,
    ('possession', 'Def Pen'): uint16,
    ('possession', 'Def 3rd'): uint16,
    ('possession', 'Mid 3rd'): uint16,
    ('possession', 'Att 3rd'): uint16,
    ('possession', 'Att Pen'): uint16,
    ('possession', 'Live'): uint16,
    ('possession', 'Succ'): uint16,
    ('possession', 'Att'): uint16,
    ('possession', 'Succ%'): float16,
    ('possession', 'Mis'): uint16,
    ('possession', 'Dis'): uint16,
    ('possession', 'Rec'): uint16,
    ('possession', 'Prog'): uint16,
    ('playingtime', 'MP'): uint16,
    ('playingtime', 'Min%'): float16,
    ('playingtime', 'Starts'): uint16,
    ('playingtime', 'Compl'): uint16,
    ('playingtime', 'Subs'): uint16,
    ('playingtime', 'unSub'): uint16,
    ('playingtime', 'PPM'): float16,
    ('playingtime', 'onG'): uint16,
    ('playingtime', 'onGA'): uint16,
    ('playingtime', '+/-'): float16,
    ('playingtime', 'onxG'): uint16,
    ('playingtime', 'onxGA'): float16,
    ('playingtime', 'xG+/-'): float16,
    ('misc', 'CrdY'): uint16,
    ('misc', 'CrdR'): uint16,
    ('misc', '2CrdY'): uint16,
    ('misc', 'Fls'): uint16,
    ('misc', 'Fld'): uint16,
    ('misc', 'Off'): uint16,
    ('misc', 'Crs'): uint16,
    ('misc', 'Int'): uint16,
    ('misc', 'TklW'): uint16,
    ('misc', 'PKwon'): uint16,
    ('misc', 'PKcon'): uint16,
    ('misc', 'Won'): uint16,
    ('misc', 'Lost'): uint16,
    ('misc', 'Won%'): uint16,
}

percentile_data_types = {
    ('perc', 'stats', 'Age'): uint8,
    ('perc', 'stats', 'MP'): uint8,
    ('perc', 'stats', 'Starts'): uint8,
    ('perc', 'stats', 'Min'): uint8,
    ('perc', 'stats', 'Value'): uint8,
    ('perc', 'stats', 'Gls'): uint8,
    ('perc', 'stats', 'Ast'): uint8,
    ('perc', 'stats', 'G-PK'): uint8,
    ('perc', 'stats', 'G+A-PK'): uint8,
    ('perc', 'stats', 'PK'): uint8,
    ('perc', 'stats', 'PKatt'): uint8,
    ('perc', 'stats', 'CrdY'): uint8,
    ('perc', 'stats', 'CrdR'): uint8,
    ('perc', 'stats', 'xG'): uint8,
    ('perc', 'stats', 'npxG'): uint8,
    ('perc', 'stats', 'xAG'): uint8,
    ('perc', 'stats', 'npxG+xAG'): uint8,
    ('perc', 'stats', 'short'): uint8,
    ('perc', 'keepers', 'GA'): uint8,
    ('perc', 'keepers', 'SoTA'): uint8,
    ('perc', 'keepers', 'Save%'): uint8,
    ('perc', 'keepers', 'CS'): uint8,
    ('perc', 'keepers', 'CS%'): uint8,
    ('perc', 'keepers', 'PKA'): uint8,
    ('perc', 'keepers', 'PKsv'): uint8,
    ('perc', 'keepers', 'PKm'): uint8,
    ('perc', 'keepers', 'Save%.1'): uint8,
    ('perc', 'keepers', 'GkMin'): uint8,
    ('perc', 'keepersadv', 'PSxG'): uint8,
    ('perc', 'keepersadv', 'PSxG/SoT'): uint8,
    ('perc', 'keepersadv', 'PSxG+/-'): uint8,
    ('perc', 'keepersadv', 'Cmp'): uint8,
    ('perc', 'keepersadv', 'Att'): uint8,
    ('perc', 'keepersadv', 'Cmp%'): uint8,
    ('perc', 'keepersadv', 'Att.1'): uint8,
    ('perc', 'keepersadv', 'Thr'): uint8,
    ('perc', 'keepersadv', 'Launch%'): uint8,
    ('perc', 'keepersadv', 'AvgLen'): uint8,
    ('perc', 'keepersadv', 'Att.2'): uint8,
    ('perc', 'keepersadv', 'Launch%.1'): uint8,
    ('perc', 'keepersadv', 'AvgLen.1'): uint8,
    ('perc', 'keepersadv', 'Opp'): uint8,
    ('perc', 'keepersadv', 'Stp'): uint8,
    ('perc', 'keepersadv', 'Stp%'): uint8,
    ('perc', 'keepersadv', '#OPA'): uint8,
    ('perc', 'keepersadv', 'AvgDist'): uint8,
    ('perc', 'shooting', 'Gls'): uint8,
    ('perc', 'shooting', 'Sh'): uint8,
    ('perc', 'shooting', 'SoT'): uint8,
    ('perc', 'shooting', 'SoT%'): uint8,
    ('perc', 'shooting', 'G/Sh'): uint8,
    ('perc', 'shooting', 'G/SoT'): uint8,
    ('perc', 'shooting', 'Dist'): uint8,
    ('perc', 'shooting', 'FK'): uint8,
    ('perc', 'shooting', 'PK'): uint8,
    ('perc', 'shooting', 'PKatt'): uint8,
    ('perc', 'shooting', 'xG'): uint8,
    ('perc', 'shooting', 'npxG'): uint8,
    ('perc', 'shooting', 'npxG/Sh'): uint8,
    ('perc', 'shooting', 'G-xG'): uint8,
    ('perc', 'shooting', 'np:G-xG'): uint8,
    ('perc', 'passing', 'Cmp'): uint8,
    ('perc', 'passing', 'Att'): uint8,
    ('perc', 'passing', 'Cmp%'): uint8,
    ('perc', 'passing', 'TotDist'): uint8,
    ('perc', 'passing', 'PrgDist'): uint8,
    ('perc', 'passing', 'Cmp.1'): uint8,
    ('perc', 'passing', 'Att.1'): uint8,
    ('perc', 'passing', 'Cmp%.1'): uint8,
    ('perc', 'passing', 'Cmp.2'): uint8,
    ('perc', 'passing', 'Att.2'): uint8,
    ('perc', 'passing', 'Cmp%.2'): uint8,
    ('perc', 'passing', 'Cmp.3'): uint8,
    ('perc', 'passing', 'Att.3'): uint8,
    ('perc', 'passing', 'Cmp%.3'): uint8,
    ('perc', 'passing', 'Ast'): uint8,
    ('perc', 'passing', 'xA'): uint8,
    ('perc', 'passing', 'xAG'): uint8,
    ('perc', 'passing', 'A-xAG'): uint8,
    ('perc', 'passing', 'KP'): uint8,
    ('perc', 'passing', '1/3'): uint8,
    ('perc', 'passing', 'PPA'): uint8,
    ('perc', 'passing', 'CrsPA'): uint8,
    ('perc', 'passing', 'Prog'): uint8,
    ('perc', 'passing_types', 'Live'): uint8,
    ('perc', 'passing_types', 'Dead'): uint8,
    ('perc', 'passing_types', 'FK'): uint8,
    ('perc', 'passing_types', 'TB'): uint8,
    ('perc', 'passing_types', 'Sw'): uint8,
    ('perc', 'passing_types', 'Crs'): uint8,
    ('perc', 'passing_types', 'TI'): uint8,
    ('perc', 'passing_types', 'CK'): uint8,
    ('perc', 'passing_types', 'In'): uint8,
    ('perc', 'passing_types', 'Out'): uint8,
    ('perc', 'passing_types', 'Str'): uint8,
    ('perc', 'passing_types', 'Cmp'): uint8,
    ('perc', 'passing_types', 'Off'): uint8,
    ('perc', 'passing_types', 'Blocks'): uint8,
    ('perc', 'gca', 'SCA'): uint8,
    ('perc', 'gca', 'PassLive'): uint8,
    ('perc', 'gca', 'PassDead'): uint8,
    ('perc', 'gca', 'Drib'): uint8,
    ('perc', 'gca', 'Sh'): uint8,
    ('perc', 'gca', 'Fld'): uint8,
    ('perc', 'gca', 'Def'): uint8,
    ('perc', 'gca', 'GCA'): uint8,
    ('perc', 'gca', 'PassLive.1'): uint8,
    ('perc', 'gca', 'PassDead.1'): uint8,
    ('perc', 'gca', 'Drib.1'): uint8,
    ('perc', 'gca', 'Sh.1'): uint8,
    ('perc', 'gca', 'Fld.1'): uint8,
    ('perc', 'gca', 'Def.1'): uint8,
    ('perc', 'defense', 'Tkl'): uint8,
    ('perc', 'defense', 'TklW'): uint8,
    ('perc', 'defense', 'Def 3rd'): uint8,
    ('perc', 'defense', 'Mid 3rd'): uint8,
    ('perc', 'defense', 'Att 3rd'): uint8,
    ('perc', 'defense', 'Tkl.1'): uint8,
    ('perc', 'defense', 'Att'): uint8,
    ('perc', 'defense', 'Tkl%'): uint8,
    ('perc', 'defense', 'Past'): uint8,
    ('perc', 'defense', 'Blocks'): uint8,
    ('perc', 'defense', 'Sh'): uint8,
    ('perc', 'defense', 'Pass'): uint8,
    ('perc', 'defense', 'Int'): uint8,
    ('perc', 'defense', 'Tkl+Int'): uint8,
    ('perc', 'defense', 'Clr'): uint8,
    ('perc', 'defense', 'Err'): uint8,
    ('perc', 'possession', 'Touches'): uint8,
    ('perc', 'possession', 'Def Pen'): uint8,
    ('perc', 'possession', 'Def 3rd'): uint8,
    ('perc', 'possession', 'Mid 3rd'): uint8,
    ('perc', 'possession', 'Att 3rd'): uint8,
    ('perc', 'possession', 'Att Pen'): uint8,
    ('perc', 'possession', 'Live'): uint8,
    ('perc', 'possession', 'Succ'): uint8,
    ('perc', 'possession', 'Att'): uint8,
    ('perc', 'possession', 'Succ%'): uint8,
    ('perc', 'possession', 'Mis'): uint8,
    ('perc', 'possession', 'Dis'): uint8,
    ('perc', 'possession', 'Rec'): uint8,
    ('perc', 'possession', 'Prog'): uint8,
    ('perc', 'playingtime', 'MP'): uint8,
    ('perc', 'playingtime', 'Min%'): uint8,
    ('perc', 'playingtime', 'Starts'): uint8,
    ('perc', 'playingtime', 'Compl'): uint8,
    ('perc', 'playingtime', 'Subs'): uint8,
    ('perc', 'playingtime', 'unSub'): uint8,
    ('perc', 'playingtime', 'PPM'): uint8,
    ('perc', 'playingtime', 'onG'): uint8,
    ('perc', 'playingtime', 'onGA'): uint8,
    ('perc', 'playingtime', '+/-'): uint8,
    ('perc', 'playingtime', 'onxG'): uint8,
    ('perc', 'playingtime', 'onxGA'): uint8,
    ('perc', 'playingtime', 'xG+/-'): uint8,
    ('perc', 'misc', 'CrdY'): uint8,
    ('perc', 'misc', 'CrdR'): uint8,
    ('perc', 'misc', '2CrdY'): uint8,
    ('perc', 'misc', 'Fls'): uint8,
    ('perc', 'misc', 'Fld'): uint8,
    ('perc', 'misc', 'Off'): uint8,
    ('perc', 'misc', 'Crs'): uint8,
    ('perc', 'misc', 'Int'): uint8,
    ('perc', 'misc', 'TklW'): uint8,
    ('perc', 'misc', 'PKwon'): uint8,
    ('perc', 'misc', 'PKcon'): uint8,
    ('perc', 'misc', 'Won'): uint8,
    ('perc', 'misc', 'Lost'): uint8,
    ('perc', 'misc', 'Won%'): uint8,
    ('posp', 'stats', 'Age'): uint8,
    ('posp', 'stats', 'MP'): uint8,
    ('posp', 'stats', 'Starts'): uint8,
    ('posp', 'stats', 'Min'): uint8,
    ('posp', 'stats', 'Value'): uint8,
    ('posp', 'stats', 'Gls'): uint8,
    ('posp', 'stats', 'Ast'): uint8,
    ('posp', 'stats', 'G-PK'): uint8,
    ('posp', 'stats', 'G+A-PK'): uint8,
    ('posp', 'stats', 'PK'): uint8,
    ('posp', 'stats', 'PKatt'): uint8,
    ('posp', 'stats', 'CrdY'): uint8,
    ('posp', 'stats', 'CrdR'): uint8,
    ('posp', 'stats', 'xG'): uint8,
    ('posp', 'stats', 'npxG'): uint8,
    ('posp', 'stats', 'xAG'): uint8,
    ('posp', 'stats', 'npxG+xAG'): uint8,
    ('posp', 'keepers', 'GA'): uint8,
    ('posp', 'keepers', 'SoTA'): uint8,
    ('posp', 'keepers', 'Save%'): uint8,
    ('posp', 'keepers', 'CS'): uint8,
    ('posp', 'keepers', 'CS%'): uint8,
    ('posp', 'keepers', 'PKA'): uint8,
    ('posp', 'keepers', 'PKsv'): uint8,
    ('posp', 'keepers', 'PKm'): uint8,
    ('posp', 'keepers', 'Save%.1'): uint8,
    ('posp', 'keepers', 'GkMin'): uint8,
    ('posp', 'keepersadv', 'PSxG'): uint8,
    ('posp', 'keepersadv', 'PSxG/SoT'): uint8,
    ('posp', 'keepersadv', 'PSxG+/-'): uint8,
    ('posp', 'keepersadv', 'Cmp'): uint8,
    ('posp', 'keepersadv', 'Att'): uint8,
    ('posp', 'keepersadv', 'Cmp%'): uint8,
    ('posp', 'keepersadv', 'Att.1'): uint8,
    ('posp', 'keepersadv', 'Thr'): uint8,
    ('posp', 'keepersadv', 'Launch%'): uint8,
    ('posp', 'keepersadv', 'AvgLen'): uint8,
    ('posp', 'keepersadv', 'Att.2'): uint8,
    ('posp', 'keepersadv', 'Launch%.1'): uint8,
    ('posp', 'keepersadv', 'AvgLen.1'): uint8,
    ('posp', 'keepersadv', 'Opp'): uint8,
    ('posp', 'keepersadv', 'Stp'): uint8,
    ('posp', 'keepersadv', 'Stp%'): uint8,
    ('posp', 'keepersadv', '#OPA'): uint8,
    ('posp', 'keepersadv', 'AvgDist'): uint8,
    ('posp', 'shooting', 'Gls'): uint8,
    ('posp', 'shooting', 'Sh'): uint8,
    ('posp', 'shooting', 'SoT'): uint8,
    ('posp', 'shooting', 'SoT%'): uint8,
    ('posp', 'shooting', 'G/Sh'): uint8,
    ('posp', 'shooting', 'G/SoT'): uint8,
    ('posp', 'shooting', 'Dist'): uint8,
    ('posp', 'shooting', 'FK'): uint8,
    ('posp', 'shooting', 'PK'): uint8,
    ('posp', 'shooting', 'PKatt'): uint8,
    ('posp', 'shooting', 'xG'): uint8,
    ('posp', 'shooting', 'npxG'): uint8,
    ('posp', 'shooting', 'npxG/Sh'): uint8,
    ('posp', 'shooting', 'G-xG'): uint8,
    ('posp', 'shooting', 'np:G-xG'): uint8,
    ('posp', 'passing', 'Cmp'): uint8,
    ('posp', 'passing', 'Att'): uint8,
    ('posp', 'passing', 'Cmp%'): uint8,
    ('posp', 'passing', 'TotDist'): uint8,
    ('posp', 'passing', 'PrgDist'): uint8,
    ('posp', 'passing', 'Cmp.1'): uint8,
    ('posp', 'passing', 'Att.1'): uint8,
    ('posp', 'passing', 'Cmp%.1'): uint8,
    ('posp', 'passing', 'Cmp.2'): uint8,
    ('posp', 'passing', 'Att.2'): uint8,
    ('posp', 'passing', 'Cmp%.2'): uint8,
    ('posp', 'passing', 'Cmp.3'): uint8,
    ('posp', 'passing', 'Att.3'): uint8,
    ('posp', 'passing', 'Cmp%.3'): uint8,
    ('posp', 'passing', 'Ast'): uint8,
    ('posp', 'passing', 'xA'): uint8,
    ('posp', 'passing', 'xAG'): uint8,
    ('posp', 'passing', 'A-xAG'): uint8,
    ('posp', 'passing', 'KP'): uint8,
    ('posp', 'passing', '1/3'): uint8,
    ('posp', 'passing', 'PPA'): uint8,
    ('posp', 'passing', 'CrsPA'): uint8,
    ('posp', 'passing', 'Prog'): uint8,
    ('posp', 'passing_types', 'Live'): uint8,
    ('posp', 'passing_types', 'Dead'): uint8,
    ('posp', 'passing_types', 'FK'): uint8,
    ('posp', 'passing_types', 'TB'): uint8,
    ('posp', 'passing_types', 'Sw'): uint8,
    ('posp', 'passing_types', 'Crs'): uint8,
    ('posp', 'passing_types', 'TI'): uint8,
    ('posp', 'passing_types', 'CK'): uint8,
    ('posp', 'passing_types', 'In'): uint8,
    ('posp', 'passing_types', 'Out'): uint8,
    ('posp', 'passing_types', 'Str'): uint8,
    ('posp', 'passing_types', 'Cmp'): uint8,
    ('posp', 'passing_types', 'Off'): uint8,
    ('posp', 'passing_types', 'Blocks'): uint8,
    ('posp', 'gca', 'SCA'): uint8,
    ('posp', 'gca', 'PassLive'): uint8,
    ('posp', 'gca', 'PassDead'): uint8,
    ('posp', 'gca', 'Drib'): uint8,
    ('posp', 'gca', 'Sh'): uint8,
    ('posp', 'gca', 'Fld'): uint8,
    ('posp', 'gca', 'Def'): uint8,
    ('posp', 'gca', 'GCA'): uint8,
    ('posp', 'gca', 'PassLive.1'): uint8,
    ('posp', 'gca', 'PassDead.1'): uint8,
    ('posp', 'gca', 'Drib.1'): uint8,
    ('posp', 'gca', 'Sh.1'): uint8,
    ('posp', 'gca', 'Fld.1'): uint8,
    ('posp', 'gca', 'Def.1'): uint8,
    ('posp', 'defense', 'Tkl'): uint8,
    ('posp', 'defense', 'TklW'): uint8,
    ('posp', 'defense', 'Def 3rd'): uint8,
    ('posp', 'defense', 'Mid 3rd'): uint8,
    ('posp', 'defense', 'Att 3rd'): uint8,
    ('posp', 'defense', 'Tkl.1'): uint8,
    ('posp', 'defense', 'Att'): uint8,
    ('posp', 'defense', 'Tkl%'): uint8,
    ('posp', 'defense', 'Past'): uint8,
    ('posp', 'defense', 'Blocks'): uint8,
    ('posp', 'defense', 'Sh'): uint8,
    ('posp', 'defense', 'Pass'): uint8,
    ('posp', 'defense', 'Int'): uint8,
    ('posp', 'defense', 'Tkl+Int'): uint8,
    ('posp', 'defense', 'Clr'): uint8,
    ('posp', 'defense', 'Err'): uint8,
    ('posp', 'possession', 'Touches'): uint8,
    ('posp', 'possession', 'Def Pen'): uint8,
    ('posp', 'possession', 'Def 3rd'): uint8,
    ('posp', 'possession', 'Mid 3rd'): uint8,
    ('posp', 'possession', 'Att 3rd'): uint8,
    ('posp', 'possession', 'Att Pen'): uint8,
    ('posp', 'possession', 'Live'): uint8,
    ('posp', 'possession', 'Succ'): uint8,
    ('posp', 'possession', 'Att'): uint8,
    ('posp', 'possession', 'Succ%'): uint8,
    ('posp', 'possession', 'Mis'): uint8,
    ('posp', 'possession', 'Dis'): uint8,
    ('posp', 'possession', 'Rec'): uint8,
    ('posp', 'possession', 'Prog'): uint8,
    ('posp', 'playingtime', 'MP'): uint8,
    ('posp', 'playingtime', 'Min%'): uint8,
    ('posp', 'playingtime', 'Starts'): uint8,
    ('posp', 'playingtime', 'Compl'): uint8,
    ('posp', 'playingtime', 'Subs'): uint8,
    ('posp', 'playingtime', 'unSub'): uint8,
    ('posp', 'playingtime', 'PPM'): uint8,
    ('posp', 'playingtime', 'onG'): uint8,
    ('posp', 'playingtime', 'onGA'): uint8,
    ('posp', 'playingtime', '+/-'): uint8,
    ('posp', 'playingtime', 'onxG'): uint8,
    ('posp', 'playingtime', 'onxGA'): uint8,
    ('posp', 'playingtime', 'xG+/-'): uint8,
    ('posp', 'misc', 'CrdY'): uint8,
    ('posp', 'misc', 'CrdR'): uint8,
    ('posp', 'misc', '2CrdY'): uint8,
    ('posp', 'misc', 'Fls'): uint8,
    ('posp', 'misc', 'Fld'): uint8,
    ('posp', 'misc', 'Off'): uint8,
    ('posp', 'misc', 'Crs'): uint8,
    ('posp', 'misc', 'Int'): uint8,
    ('posp', 'misc', 'TklW'): uint8,
    ('posp', 'misc', 'PKwon'): uint8,
    ('posp', 'misc', 'PKcon'): uint8,
    ('posp', 'misc', 'Won'): uint8,
    ('posp', 'misc', 'Lost'): uint8,
    ('posp', 'misc', 'Won%'): uint8,
}
