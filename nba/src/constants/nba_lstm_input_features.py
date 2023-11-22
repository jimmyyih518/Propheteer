from dataclasses import dataclass, field
from typing import List


@dataclass
class NbaLstmInputFeaturesV2:
    data: List[str] = field(
        default_factory=lambda: [
            "pts_prev_season",
            "PREV_1G_AVG_minutes_played",
            "PREV_10G_AVG_PTS",
            "day_of_year",
            "age",
            "rest_days",
            "PREV_3G_AVG_FGA",
            "PREV_1G_AVG_FGA",
            "PREV_10G_AVG_PTS_opponent_team",
            "reb_prev_season",
            "PREV_10G_AVG_REB",
            "player_height",
            "player_weight",
            "PREV_10G_AVG_minutes_played",
            "PREV_10G_AVG_AST",
            "ast_prev_season",
            "PREV_3G_AVG_AST",
            "player_team_at_home",
            "PREV_1G_AVG_PLUS_MINUS",
            "PREV_10G_AVG_FG_PCT",
            "PREV_10G_AVG_STL",
            "draft_number",
            "draft_round",
            "PREV_10G_AVG_BLK",
            "PREV_10G_AVG_FG_PCT_opponent_team",
            "PREV_3G_AVG_BLK",
            "PREV_10G_AVG_FG3_PCT_opponent_team",
            "PREV_10G_AVG_FT_PCT_opponent_team",
        ]
    )

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


# Example usage
# player_stats = PlayerStats()
# print(player_stats[0])  # Accessing the first item
# print(len(player_stats))  # Getting the number of items
# for stat in player_stats:
#    print(stat)  # Iterating through all items
