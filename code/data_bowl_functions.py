import logging
import warnings
import datetime as dt
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

# Suppress the SettingWithCopyWarning
pd.set_option("mode.chained_assignment", None)

# Anchor all data and output paths to the repository root so the pipeline
# resolves correctly regardless of the working directory it is run from.
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configured once on import so any entry point that imports this module
# gets pipeline logging exported to the logs folder by default.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            LOGS_DIR / f"pipeline_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
        ),
        logging.StreamHandler(),
    ],
)


def get_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all weekly tracking data and supplementary play info.

    Reads all 18 weeks of pre-throw (input) and post-throw (output)
    player tracking data from the data folder and concatenates them,
    along with the play-level supplementary data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of
        (all_input_data, all_output_data, play_info), where
        all_input_data and all_output_data hold the concatenated weekly
        tracking data and play_info holds one row per play.
    """
    # Iterate through weeks and pull input/output data
    input_data_list = []
    output_data_list = []
    for i in range(18):
        if i >= 9:
            week = str(i + 1)
        else:
            week = "0" + str(i + 1)

        logging.info(f"Loading week {week} input/output tracking data")
        input_data = pd.read_csv(DATA_DIR / f"input_2023_w{week}.csv")
        output_data = pd.read_csv(DATA_DIR / f"output_2023_w{week}.csv")

        input_data_list.append(input_data)
        output_data_list.append(output_data)

    # Concat all input and output data together
    all_input_data = pd.concat(input_data_list)
    all_output_data = pd.concat(output_data_list)

    # Pull supplementary data as one csv
    play_info = pd.read_csv(DATA_DIR / "supplementary_data.csv")
    logging.info(f"Loaded {len(play_info)} plays of supplementary data")

    return all_input_data, all_output_data, play_info


def add_to_play_info(
    input_data: pd.DataFrame, output_data: pd.DataFrame, play_info: pd.DataFrame
) -> pd.DataFrame:
    """Enrich play_info with defender, receiver, and ball position data.

    Adds the closest man-coverage defender (def_id), the targeted
    receiver (off_id), the ball's release and landing position, the
    vertical ball speed, and play direction/yardline to play_info.
    Plays without tracking info for the ball landing spot are dropped.

    Args:
        input_data (pd.DataFrame): Pre-throw player tracking data.
        output_data (pd.DataFrame): Post-throw player tracking data.
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play.

    Returns:
        pd.DataFrame: play_info enriched with def_id, off_id,
        ball_start_x/y, ball_land_x/y, ball_speed_v, play_direction,
        and absolute_yardline_number.
    """
    # Find the target defender nfl_id and add it to play_info
    logging.info(f"Finding closest defender for {len(play_info)} plays")
    for row in play_info.itertuples():
        def_id = find_closest_defender(
            input_data, output_data, row.play_id, row.game_id
        )
        play_info.loc[row.Index, "def_id"] = def_id

    # Find target receiver nfl_id and add it to play_info
    target_receivers = input_data[
        input_data["player_role"] == "Targeted Receiver"
    ].copy()
    target_receivers = (
        target_receivers.groupby(["game_id", "play_id"])["nfl_id"].first().reset_index()
    )
    target_receivers.rename(columns={"nfl_id": "off_id"}, inplace=True)
    play_info = play_info.merge(target_receivers, on=["game_id", "play_id"], how="left")

    # Get ball_start / end position and add to play_info
    qb_data = input_data[input_data["player_position"] == "QB"].copy()
    ball_position = (
        qb_data.groupby(["game_id", "play_id"])["frame_id"].max().reset_index()
    )
    ball_position = ball_position.merge(
        qb_data[
            [
                "game_id",
                "play_id",
                "frame_id",
                "x",
                "y",
                "ball_land_x",
                "ball_land_y",
                "num_frames_output",
            ]
        ],
        on=["game_id", "play_id", "frame_id"],
        how="left",
    )
    ball_position.rename(
        columns={"x": "ball_start_x", "y": "ball_start_y"}, inplace=True
    )
    ball_position.drop(columns=["frame_id"], inplace=True)

    # Remove duplicates before merging
    ball_position = ball_position.drop_duplicates(subset=["game_id", "play_id"])
    play_info = play_info.merge(ball_position, on=["game_id", "play_id"], how="left")

    # Drop plays without tracking info
    play_info = play_info[~play_info["ball_land_x"].isnull()]

    # Add the vertical ball speed to play_info given throw height and catch
    # height of 3 yards. g (10.725) is gravitational acceleration in
    # yards/s^2.
    play_info["ball_speed_v"] = np.vectorize(find_ball_speed_v)(
        play_info["num_frames_output"], 10.725
    )

    # Add play direction and absolute yardline number to play_info
    direction_data = (
        input_data.groupby(["game_id", "play_id"])
        .agg({"play_direction": "first", "absolute_yardline_number": "first"})
        .reset_index()
    )
    play_info = play_info.merge(direction_data, on=["game_id", "play_id"], how="left")

    # Ensure num_frames_output is integer
    play_info["num_frames_output"] = play_info["num_frames_output"].astype(int)

    return play_info


def get_target_plays(
    input_data: pd.DataFrame, output_data: pd.DataFrame, play_info: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter tracking and play data down to the DIP analysis subset.

    Restricts plays to man coverage on short routes (SLANT, CROSS, IN,
    OUT, FLAT), which is the scope of the DIP duel model, then filters
    the tracking data to match.

    Args:
        input_data (pd.DataFrame): Pre-throw player tracking data.
        output_data (pd.DataFrame): Post-throw player tracking data.
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of
        (input_data, output_data, target_plays) filtered to the man
        coverage / short route subset.
    """
    # Filter info frame to man coverage scenarios with short routes
    target_plays = play_info[
        play_info["route_of_targeted_receiver"].isin(
            ["SLANT", "CROSS", "IN", "OUT", "FLAT"]
        )
        & (play_info["team_coverage_man_zone"] == "MAN_COVERAGE")
    ]

    # Filter input and output data to match target plays
    input_data = input_data.merge(
        target_plays[["game_id", "play_id"]], on=["game_id", "play_id"], how="inner"
    )
    output_data = output_data.merge(
        target_plays[["game_id", "play_id"]], on=["game_id", "play_id"], how="inner"
    )
    logging.info(
        f"Filtered to {len(target_plays)} man-coverage, short-route target plays"
    )
    return input_data, output_data, target_plays


def find_closest_defender(
    input_data: pd.DataFrame, output_data: pd.DataFrame, play_id: int, game_id: int
) -> int | None:
    """Find the defender with the smallest average distance to the receiver.

    Combines pre- and post-throw tracking data for the play and
    computes, for every defender, the average distance to the targeted
    receiver across all shared frames. The defender with the lowest
    average distance is treated as the man-coverage defender.

    Args:
        input_data (pd.DataFrame): Pre-throw player tracking data.
        output_data (pd.DataFrame): Post-throw player tracking data.
        play_id (int): The play identifier.
        game_id (int): The game identifier.

    Returns:
        int | None: The nfl_id of the closest defender, or None if the
        play has no targeted receiver or no defenders.
    """
    # Filter to play data
    play_output = output_data[
        (output_data["play_id"] == play_id) & (output_data["game_id"] == game_id)
    ]
    play_input = input_data[
        (input_data["play_id"] == play_id) & (input_data["game_id"] == game_id)
    ]

    # Concat play data together
    input_max_frame = play_input["frame_id"].max()
    play_output["frame_id"] = play_output["frame_id"] + input_max_frame

    # Filter for the given play_id
    play_data = pd.concat([play_input, play_output], ignore_index=True).reset_index(
        drop=True
    )

    # Find the targeted receiver
    receiver = play_data[play_data["player_role"] == "Targeted Receiver"]
    if receiver.empty:
        logging.warning(
            f"No targeted receiver found for game_id={game_id}, play_id={play_id}"
        )
        return None  # No targeted receiver found
    receiver = receiver[["x", "y", "frame_id"]]
    receiver.rename(columns={"x": "rx", "y": "ry"}, inplace=True)

    # Get all defenders in the same frame
    defenders = play_data[(play_data["player_side"] == "Defense")]
    if defenders.empty:
        logging.warning(
            f"No defenders found for game_id={game_id}, play_id={play_id}"
        )
        return None  # No defenders found

    # Compute distances
    defenders = defenders.merge(receiver, on="frame_id", how="left")
    defenders["distance"] = (
        (defenders["x"] - defenders["rx"]) ** 2
        + (defenders["y"] - defenders["ry"]) ** 2
    ) ** 0.5
    defender_avg_dis = defenders.groupby("nfl_id")["distance"].mean().reset_index()
    closest_def = defender_avg_dis.loc[defender_avg_dis["distance"].idxmin()]
    return int(closest_def["nfl_id"])


def find_ball_speed_v(num_frames: int, g: float) -> float:
    """Compute the initial vertical speed of a thrown ball.

    Assumes the ball leaves and lands at the same height, so the
    vertical speed decays linearly to zero at the midpoint of flight.

    Args:
        num_frames (int): Number of output frames the ball is in
            flight, at 10 frames per second.
        g (float): Gravitational acceleration, in yards per second
            squared.

    Returns:
        float: The initial vertical speed of the ball, in yards per
        second.
    """
    ball_speed_v = (g * (num_frames / 10)) / 2
    return ball_speed_v


def find_ball_height(ball_speed_v: float, frame: int, g: float) -> float:
    """Compute the ball's height at a given frame of its flight.

    Models the ball's vertical position as projectile motion released
    from a height of 2 yards.

    Args:
        ball_speed_v (float): Initial vertical speed of the ball, in
            yards per second.
        frame (int): Frame number since the ball was thrown, at 10
            frames per second.
        g (float): Gravitational acceleration, in yards per second
            squared.

    Returns:
        float: The ball's height, in yards.
    """
    time = frame / 10
    ball_height = 2 + (ball_speed_v * time) - (0.5 * g * time**2)
    return ball_height


def find_non_ca_proj_time(
    v_max: float, v_i: float, a_max: float, d_total: float
) -> float:
    """Find the time taken to accelerate from initial velocity to max
    velocity given max acceleration.

    Splits the trip into a constant-acceleration phase up to v_max and
    a constant-velocity phase at v_max for the remaining distance.

    Args:
        v_max (float): The player's maximum speed, in yards per
            second.
        v_i (float): The player's initial speed toward the target, in
            yards per second.
        a_max (float): The player's maximum acceleration, in yards per
            second squared.
        d_total (float): Total distance to cover, in yards.

    Returns:
        float: The projected travel time, in seconds.
    """
    t_accel = (v_max - v_i) / a_max
    t_const = (d_total / v_max) - ((v_max**2 - v_i**2) / (2 * a_max * v_max))
    t_proj = t_accel + t_const
    return t_proj


def find_ca_proj_time(v_i: float, a_max: float, d_total: float) -> float:
    """Find the time taken to cover a distance with constant
    acceleration from initial velocity.

    Used when a player reaches the target distance before accelerating
    up to their maximum speed.

    Args:
        v_i (float): The player's initial speed toward the target, in
            yards per second.
        a_max (float): The player's maximum acceleration, in yards per
            second squared.
        d_total (float): Total distance to cover, in yards.

    Returns:
        float: The projected travel time, in seconds.
    """
    t_proj = (-v_i + (v_i**2 + 2 * a_max * d_total) ** 0.5) / a_max
    return t_proj


def find_time_delta(
    t_total: float, v_max: float, v_i: float, a_max: float, d_total: float
) -> float:
    """Find how far ahead of (or behind) the ball a player would arrive.

    Projects the player's travel time to a distance using Spearman's
    reach model, then compares it to the time actually available.

    Args:
        t_total (float): Time available before the ball arrives, in
            seconds.
        v_max (float): The player's maximum speed, in yards per
            second.
        v_i (float): The player's initial speed toward the target, in
            yards per second.
        a_max (float): The player's maximum acceleration, in yards per
            second squared.
        d_total (float): Total distance to cover, in yards.

    Returns:
        float: The time delta (t_total minus projected travel time).
        Positive means the player is projected to arrive early.
    """
    # Find the final velocity given constant acceleration
    ca_v_f = (v_i**2 + 2 * a_max * d_total) ** 0.5

    # If the final velocity is less than the max velocity, use constant
    # acceleration formula
    if ca_v_f <= v_max:
        t_proj = find_ca_proj_time(v_i, a_max, d_total)

    # Else, use formula assuming that we reach max velocity
    else:
        t_proj = find_non_ca_proj_time(v_max, v_i, a_max, d_total)

    # Time delta is total time minus projected time
    t_delta = t_total - t_proj

    return t_delta


def logistic_scale_from_std(sigma: float) -> float:
    """Convert logistic std dev sigma to logistic scale s.

    Args:
        sigma (float): Standard deviation of the logistic distribution.

    Returns:
        float: The logistic scale parameter s.
    """
    return np.pi * sigma / np.sqrt(3.0)


def logistic_pdf_from_z(z: float, sigma: float) -> float:
    """Logistic PDF evaluated at z = T - t_reach.

    Models the probability density that a player reaches the ball at a
    given time difference z, softening the hard reach/no-reach cutoff
    with a logistic distribution of width sigma. z can be scalar or
    numpy array.

    Args:
        z (float): Time difference between ball arrival and player
            reach time. May be a scalar or a numpy array.
        sigma (float): Standard deviation of the logistic distribution,
            controlling temporal uncertainty.

    Returns:
        float: The logistic probability density at z. Matches the
        shape of z if z is an array.
    """
    s = logistic_scale_from_std(sigma)
    # compute exponent argument safely
    x = -z / s
    ex = np.exp(np.clip(x, -60, 60))  # clip to avoid overflow
    return ex / (s * (1.0 + ex) ** 2)


def get_distance(
    x1: float, x2: float, y1: float, y2: float
) -> float:  # Finds the the distance between (x1,y1) and (x2,y2)
    """Find the Euclidean distance between two points.

    Args:
        x1 (float): X coordinate of the first point.
        x2 (float): X coordinate of the second point.
        y1 (float): Y coordinate of the first point.
        y2 (float): Y coordinate of the second point.

    Returns:
        float: The distance between (x1, y1) and (x2, y2).
    """
    dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dis


def get_player_maxes(input_data: pd.DataFrame) -> pd.DataFrame:
    """Find each player's max acceleration and speed across all plays.

    Acceleration and speed are first averaged within one-second
    windows (10 frames) to smooth out single-frame noise before taking
    the max, so brief sensor spikes do not set a player's max.

    Args:
        input_data (pd.DataFrame): Pre-throw player tracking data.

    Returns:
        pd.DataFrame: One row per nfl_id with columns a_max and s_max.
    """
    input_data["frame_floor"] = np.floor(input_data["frame_id"] / 10).astype(int)
    max_frame = (
        input_data.groupby(["game_id", "play_id", "nfl_id", "frame_floor"])
        .agg({"a": "mean", "s": "mean"})
        .reset_index()
    )
    max_frame = (
        max_frame.groupby(["nfl_id"]).agg({"a": "max", "s": "max"}).reset_index()
    )
    max_frame.rename(columns={"a": "a_max", "s": "s_max"}, inplace=True)
    logging.info(f"Computed max acceleration/speed for {len(max_frame)} players")
    return max_frame


def get_augmented_output(
    input_data: pd.DataFrame, output_data: pd.DataFrame, play_row: NamedTuple
) -> pd.DataFrame:
    """Combine the play's defender/receiver tracking into one timeline.

    Anchors the post-throw output frames to the pre-throw input frames
    by prepending the last input frame as frame 0, restricted to the
    defender and targeted receiver only.

    Args:
        input_data (pd.DataFrame): Pre-throw player tracking data.
        output_data (pd.DataFrame): Post-throw player tracking data.
        play_row (NamedTuple): A play_info row from itertuples(), used
            for its game_id, play_id, def_id, and off_id.

    Returns:
        pd.DataFrame: Combined defender/receiver tracking data for the
        play, with frame_id 0 at the moment of the throw.
    """
    # Get the last frame of the input data and add to the output data
    filtered_input = input_data[
        (input_data["game_id"] == play_row.game_id)
        & (input_data["play_id"] == play_row.play_id)
        & (input_data["nfl_id"].isin([play_row.def_id, play_row.off_id]))
    ][["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "a", "s", "o", "dir"]]
    filtered_output = output_data[
        (output_data["game_id"] == play_row.game_id)
        & (output_data["play_id"] == play_row.play_id)
        & (output_data["nfl_id"].isin([play_row.def_id, play_row.off_id]))
    ]
    last_input_frame = filtered_input[
        filtered_input["frame_id"] == filtered_input["frame_id"].max()
    ]
    last_input_frame.loc[:, "frame_id"] = 0
    augmented_output = pd.concat([last_input_frame, filtered_output], ignore_index=True)
    return augmented_output


def get_ball_position_frame(
    augmented_output: pd.DataFrame, play_row: NamedTuple
) -> pd.DataFrame:
    """Build the ball's interpolated (x, y, z) position for every frame.

    Anchors the ball at its release position on frame 0 and its
    landing position on the final frame, linearly interpolates the
    horizontal position between them, and derives height from the
    projectile motion model.

    Args:
        augmented_output (pd.DataFrame): Combined defender/receiver
            tracking data for the play, as returned by
            get_augmented_output.
        play_row (NamedTuple): A play_info row from itertuples(), used
            for ball_start_x/y, ball_land_x/y, and ball_speed_v.

    Returns:
        pd.DataFrame: One row per frame_id with columns x_ball, y_ball,
        and z_ball.
    """
    # Find the ball x, y, z posiiton for the given play and return
    ball_position_frame = (
        augmented_output.groupby(["game_id", "play_id", "frame_id"])
        .count()
        .reset_index()[["game_id", "play_id", "frame_id"]]
    )
    ball_position_frame.loc[
        ball_position_frame["frame_id"] == 0, ["x_ball", "y_ball"]
    ] = play_row.ball_start_x, play_row.ball_start_y
    ball_position_frame.loc[
        ball_position_frame["frame_id"] == ball_position_frame["frame_id"].max(),
        ["x_ball", "y_ball"],
    ] = play_row.ball_land_x, play_row.ball_land_y
    ball_position_frame.sort_values(by="frame_id", ascending=True, inplace=True)
    ball_position_frame["x_ball"] = ball_position_frame["x_ball"].interpolate()
    ball_position_frame["y_ball"] = ball_position_frame["y_ball"].interpolate()
    ball_position_frame["z_ball"] = np.vectorize(find_ball_height)(
        play_row.ball_speed_v, ball_position_frame["frame_id"], 10.725
    )
    return ball_position_frame


def fill_player_movement_data(
    augmented_output: pd.DataFrame, nfl_id: int
) -> pd.DataFrame:
    """Use player tracking data to fill output movement data for the
    given player.

    Output (post-throw) tracking data does not include speed,
    direction, or acceleration, so they are derived here from
    frame-to-frame position deltas where missing.

    Args:
        augmented_output (pd.DataFrame): Combined defender/receiver
            tracking data for the play, as returned by
            get_augmented_output.
        nfl_id (int): The player to compute movement data for.

    Returns:
        pd.DataFrame: The player's tracking data, sorted by frame_id,
        with s, dir, and a filled in where missing.
    """
    player_data = augmented_output[augmented_output["nfl_id"] == nfl_id]
    player_data.sort_values(by="frame_id", ascending=True, inplace=True)
    player_data.reset_index(drop=True, inplace=True)
    player_data["x_dist"] = player_data["x"].diff().fillna(0)
    player_data["y_dist"] = player_data["y"].diff().fillna(0)
    player_data["total_dist"] = np.sqrt(
        player_data["x_dist"] ** 2 + player_data["y_dist"] ** 2
    )
    player_data.loc[player_data["s"].isnull(), "s"] = player_data["total_dist"] * 10
    player_data.loc[player_data["dir"].isnull(), "dir"] = np.degrees(
        np.arctan2(player_data["y_dist"], player_data["x_dist"])
    ).fillna(0)
    player_data.loc[player_data["a"].isnull(), "a"] = (
        player_data["s"].diff().fillna(0) * 10
    )
    player_data.loc[player_data["dir"] < 0, "dir"] += 360
    return player_data


def get_situation_data(
    player_data: pd.DataFrame,
    ball_row: pd.Series,
    max_frame: pd.DataFrame,
    nfl_id: int,
) -> pd.DataFrame | None:
    """Build a player's reach situation relative to one ball position.

    For every frame of player_data before the ball's frame, computes
    the distance and direction to the ball spot, the player's speed
    toward it, and the time delta (find_time_delta) versus the
    player's max speed/acceleration.

    Args:
        player_data (pd.DataFrame): A single player's tracking data,
            as returned by fill_player_movement_data.
        ball_row (pd.Series): One row of ball_position_frame giving the
            ball's x_ball/y_ball/frame_id for a single ball spot.
        max_frame (pd.DataFrame): Per-player a_max/s_max, as returned
            by get_player_maxes.
        nfl_id (int): The player nfl_id, used to look up max_frame.

    Returns:
        pd.DataFrame | None: The player's situation data relative to
        this ball spot, or None if no frames occur before the ball's
        frame.
    """
    # Build situation data for each ball
    situation_data = player_data.copy()
    situation_data["ball_x_dist"] = ball_row["x_ball"] - situation_data["x"]
    situation_data["ball_y_dist"] = ball_row["y_ball"] - situation_data["y"]
    situation_data["ball_total_dist"] = np.sqrt(
        situation_data["ball_x_dist"] ** 2 + situation_data["ball_y_dist"] ** 2
    )
    situation_data["time_to_ball"] = (
        ball_row["frame_id"] - situation_data["frame_id"]
    ) / 10.0
    situation_data = situation_data[situation_data["time_to_ball"] >= 0]
    if situation_data.empty:
        return None
    situation_data["ball_dir"] = np.degrees(
        np.arctan2(situation_data["ball_y_dist"], situation_data["ball_x_dist"])
    ).fillna(0)
    situation_data.loc[situation_data["ball_dir"] < 0, "ball_dir"] += 360
    situation_data["dir_diff"] = np.abs(
        situation_data["dir"] - situation_data["ball_dir"]
    )
    situation_data["v_to_ball"] = (
        np.cos(np.radians(situation_data["dir_diff"])) * situation_data["s"]
    )
    a_max = max_frame.loc[max_frame["nfl_id"] == nfl_id, "a_max"].values[0]
    s_max = max_frame.loc[max_frame["nfl_id"] == nfl_id, "s_max"].values[0]
    situation_data["t_delta"] = np.vectorize(find_time_delta)(
        situation_data["time_to_ball"],
        s_max,
        situation_data["v_to_ball"],
        a_max,
        situation_data["ball_total_dist"],
    )
    situation_data["ball_frame"] = ball_row["frame_id"]
    return situation_data


def get_all_situation_data(
    play_info: pd.DataFrame,
    input_data: pd.DataFrame,
    output_data: pd.DataFrame,
    max_frame: pd.DataFrame,
    animate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full DIP simulation across every play in play_info.

    For each play: builds the ball trajectory, fills in player
    movement data, computes each player's reach situation against
    every catchable ball spot, and derives the DIP probabilities
    (get_play_probabilities). This is the most expensive step in the
    pipeline; cache its output rather than re-running it unnecessarily.

    Args:
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play, enriched by add_to_play_info.
        input_data (pd.DataFrame): Pre-throw player tracking data.
        output_data (pd.DataFrame): Post-throw player tracking data.
        max_frame (pd.DataFrame): Per-player a_max/s_max, as returned
            by get_player_maxes.
        animate (bool): If True, also render and save a GIF animation
            for every play via create_play_gif. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple of
        (all_situations, all_probabilities, all_ball_positions)
        concatenated across every play.
    """
    # Add an output list to hold all output data
    full_output_list = []
    all_prob_list = []
    all_ball_list = []

    num_plays = len(play_info)
    logging.info(f"Starting DIP simulation for {num_plays} plays")

    # Iterate through each play and build cleaned output data with ball
    # trajectory
    for i, play_row in enumerate(play_info.itertuples(), start=1):
        try:
            # Get the last frame of the input data and add to the output data
            augmented_output = get_augmented_output(input_data, output_data, play_row)

            # Find the ball x, y, z posiiton and add to the output data
            ball_position_frame = get_ball_position_frame(augmented_output, play_row)

            play_situations_list = []

            for nfl_id in augmented_output["nfl_id"].unique():
                # Fill distance, speed and direction for each player
                player_data = fill_player_movement_data(augmented_output, nfl_id)

                for _, ball_row in ball_position_frame[
                    ball_position_frame["z_ball"] <= 3
                ].iterrows():
                    situation_data = get_situation_data(
                        player_data, ball_row, max_frame, nfl_id
                    )
                    if situation_data is not None:
                        play_situations_list.append(situation_data)

            play_situations = pd.concat(play_situations_list, ignore_index=True)
            play_situations["ball_frame"] = play_situations["ball_frame"].astype(int)

            play_probabilities = get_play_probabilities(
                play_situations, play_row, sigma=0.31, lambda_control=False
            )

            all_ball_list.append(ball_position_frame)
            full_output_list.append(play_situations)
            all_prob_list.append(play_probabilities)

            if animate:
                play_input = input_data[
                    (input_data["game_id"] == play_row.game_id)
                    & (input_data["play_id"] == play_row.play_id)
                ]
                play_output = output_data[
                    (output_data["game_id"] == play_row.game_id)
                    & (output_data["play_id"] == play_row.play_id)
                ]
                full_play = get_animation_data(
                    play_input,
                    play_output,
                    ball_position_frame,
                    play_probabilities,
                    play_row,
                )
                create_play_gif(full_play, play_row)

            if i % 100 == 0:
                logging.info(f"Processed {i}/{num_plays} plays")

        except Exception as e:
            logging.error(
                f"Failed to simulate game_id={play_row.game_id}, "
                + f"play_id={play_row.play_id}: {e}"
            )
            continue

    logging.info(f"Completed DIP simulation for {len(full_output_list)}/{num_plays} plays")

    # Concat all situations together
    all_situations = pd.concat(full_output_list, ignore_index=True)
    all_probabilities = pd.concat(all_prob_list, ignore_index=True)
    all_ball_positions = pd.concat(all_ball_list)

    return all_situations, all_probabilities, all_ball_positions


def get_play_probabilities(
    play_situations: pd.DataFrame,
    play_row: NamedTuple,
    sigma: float = 0.45,
    lambda_rate: float = 4.30,
    lambda_control: bool = True,
) -> pd.DataFrame:
    """Compute DIP for the defender and receiver at every frame.

    At each frame, allocates cumulative reach probability between the
    defender and targeted receiver using the logistic reach-time PDF
    (logistic_pdf_from_z), capping the total allocated probability at
    1 across the remaining flight. Frames with 3 or fewer situations
    (i.e. missing defender or receiver data) are skipped.

    Args:
        play_situations (pd.DataFrame): Per-player, per-frame reach
            situations for the play, as returned by
            get_all_situation_data.
        play_row (NamedTuple): A play_info row from itertuples(), used
            for its game_id, play_id, def_id, and off_id.
        sigma (float): Standard deviation of the reach-time logistic
            distribution, in seconds. Defaults to 0.45.
        lambda_rate (float): Rate parameter for the optional control
            probability term, in events per second. Defaults to 4.30.
        lambda_control (bool): If True, scale the incremental
            probability by 1 - e^(-lambda_rate * dt) to model the
            chance of maintaining ball control once reached. Defaults
            to True.

    Returns:
        pd.DataFrame: One row per frame_id with columns def_bal_prob
        and off_bal_prob giving the cumulative DIP for the defender and
        receiver.
    """
    play_probabilities_list = []
    for frame_id in play_situations["frame_id"].unique():
        frame_data = play_situations[play_situations["frame_id"] == frame_id]
        if len(frame_data) > 3:
            def_frame = frame_data[frame_data["nfl_id"] == play_row.def_id][
                ["time_to_ball", "t_delta"]
            ]
            def_frame = def_frame.rename(columns={"t_delta": "def_t_delta"})
            off_frame = frame_data[frame_data["nfl_id"] == play_row.off_id][
                ["time_to_ball", "t_delta"]
            ]
            off_frame = off_frame.rename(columns={"t_delta": "off_t_delta"})
            both_frame = def_frame.merge(off_frame, on="time_to_ball", how="left")
            both_frame.set_index("time_to_ball", inplace=True, drop=True)

            t_steps, n_players = both_frame.shape
            p_cumulative = np.zeros(n_players, dtype=float)
            p_timeseries = np.zeros((t_steps, n_players), dtype=float)

            p_int = logistic_pdf_from_z(both_frame.values, sigma)

            if lambda_control:
                p_ctrl = 1.0 - np.exp(-lambda_rate * 0.1)
            else:
                p_ctrl = 1.0

            for i in range(t_steps):
                remaining = 1.0 - p_cumulative.sum()

                p_timeseries[i:, :] = p_cumulative

                # instantaneous densities for this time step
                p_int_i = p_int[i, :]  # shape (N,)
                # incremental probability in this dt
                dp = remaining * p_int_i * p_ctrl
                total_dp = dp.sum()
                # guard against numerical overshoot
                if total_dp > remaining and total_dp > 0:
                    dp *= remaining / total_dp

                # update cumulative probabilities
                p_cumulative += dp
                p_timeseries[i, :] = p_cumulative

            probability_row = pd.DataFrame(
                {
                    "game_id": play_row.game_id,
                    "play_id": play_row.play_id,
                    "frame_id": frame_id,
                    "def_bal_prob": p_cumulative[0],
                    "off_bal_prob": p_cumulative[1],
                },
                index=[0],
            )
            play_probabilities_list.append(probability_row)

        else:
            continue
    play_probabilities = pd.concat(play_probabilities_list, ignore_index=True)
    return play_probabilities


def find_catch_distance(play_info: pd.DataFrame, all_situations: pd.DataFrame) -> float:
    """Find the median ball-to-receiver distance at the moment of catch.

    Restricted to completed passes, and outlier plays above the upper
    Tukey fence are excluded so a few broken-tracking plays do not
    skew the median.

    Args:
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play, including pass_result and off_id.
        all_situations (pd.DataFrame): Per-player, per-frame reach
            situations across all plays, as returned by
            get_all_situation_data.

    Returns:
        float: The median ball_total_dist at the receiver's last
        tracked frame, across completed passes.
    """
    # Finds the median distance at the time of catch for all completed
    # passes
    distance_frame = play_info[play_info["pass_result"] == "C"][
        ["game_id", "play_id", "off_id", "num_frames_output"]
    ].copy()
    landing_situations = (
        all_situations.groupby(["game_id", "play_id", "nfl_id"])
        .agg({"frame_id": "max", "ball_frame": "max"})
        .reset_index()
    )
    landing_situations = landing_situations.merge(
        all_situations[
            [
                "game_id",
                "play_id",
                "nfl_id",
                "frame_id",
                "ball_frame",
                "ball_total_dist",
            ]
        ],
        on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
    )
    distance_frame = distance_frame.merge(
        landing_situations,
        left_on=["game_id", "play_id", "off_id"],
        right_on=["game_id", "play_id", "nfl_id"],
        how="left",
    )

    percent_array = distance_frame["ball_total_dist"].describe()
    dist_q_one = percent_array["25%"]
    dist_q_three = percent_array["75%"]
    upper_outlier_bound = dist_q_three + 1.5 * (dist_q_three - dist_q_one)
    distance_frame = distance_frame[
        (distance_frame["ball_total_dist"] < upper_outlier_bound)
    ]
    return distance_frame["ball_total_dist"].median()


def get_outlier_bounds(array: pd.Series) -> Tuple[float, float]:
    """Find outlier bounds for a numeric series using the IQR.

    Args:
        array (pd.Series): The values to compute bounds for.

    Returns:
        Tuple[float, float]: A tuple of (upper_bound, lower_bound).
    """
    percent_array = array.describe()
    dist_q_one = percent_array["25%"]
    dist_q_three = percent_array["75%"]
    upper_bound = dist_q_three + 1.5 * (dist_q_three - dist_q_one)
    lower_bound = dist_q_one - 1.5 + (dist_q_three - dist_q_one)

    return upper_bound, lower_bound


def classify_movement_path(
    direction: str, ball_dir: float, def_dir: float, dir_tol: float = 5.0
) -> str:
    """Classify defender movement path relative to ball direction.

    Compares the defender's movement direction to the ball's direction
    of travel, accounting for the play's direction (left/right), to
    determine whether the defender broke toward the ball, stayed with
    the receiver (man coverage), or moved away from the play.

    Args:
        direction (str): The play direction, "left" or "right".
        ball_dir (float): The ball's direction of travel, in degrees
            (0-360).
        def_dir (float): The defender's movement direction, in degrees
            (0-360).
        dir_tol (float): Tolerance, in degrees, for treating the
            defender's direction as matching the ball's direction.
            Defaults to 5.0.

    Returns:
        str: One of "MAN" (defender tracked the receiver), "BALL"
        (defender broke toward the ball), "OUT" (defender moved away
        from the play), or "UNKNOWN" (direction was not "left" or
        "right").
    """
    if (direction == "right") & (ball_dir > 180):
        if def_dir > ball_dir + dir_tol:
            return "MAN"
        elif 180 < def_dir < ball_dir + dir_tol:
            return "BALL"
        else:
            return "OUT"
    elif (direction == "right") & (ball_dir < 180):
        if def_dir < ball_dir - dir_tol:
            return "MAN"
        elif 180 > def_dir > ball_dir - dir_tol:
            return "BALL"
        else:
            return "OUT"
    elif (direction == "left") & (ball_dir < 180):
        if def_dir > ball_dir + dir_tol:
            return "MAN"
        elif 0 < def_dir < ball_dir + dir_tol:
            return "BALL"
        else:
            return "OUT"
    elif (direction == "left") & (ball_dir > 180):
        if def_dir < ball_dir - dir_tol:
            return "MAN"
        elif 360 > def_dir > ball_dir - dir_tol:
            return "BALL"
        else:
            return "OUT"
    else:
        return "UNKNOWN"


def get_max_frame(all_situations: pd.DataFrame) -> None:
    """Compute and export each player's max acceleration/speed to CSV.

    Args:
        all_situations (pd.DataFrame): Per-player, per-frame reach
            situations across all plays, as returned by
            get_all_situation_data.

    Returns:
        None: Writes the result to max_frame.csv at the repository root.
    """
    max_frame = (
        all_situations.groupby(["nfl_id"]).agg({"a": "max", "s": "max"}).reset_index()
    )
    max_frame.rename(columns={"a": "a_max", "s": "s_max"}, inplace=True)
    max_frame.to_csv(ROOT_DIR / "max_frame.csv", index=False)


def add_player_dir(
    all_situations: pd.DataFrame, play_info: pd.DataFrame
) -> pd.DataFrame:
    """Add the defender's and receiver's overall movement direction.

    Computes each player's movement direction from their first to
    last tracked position during the ball's flight, then merges the
    defender's and receiver's direction onto play_info. Plays where
    the defender's movement direction is exactly 0 (no tracked
    movement) are dropped.

    Args:
        all_situations (pd.DataFrame): Per-player, per-frame reach
            situations across all plays, as returned by
            get_all_situation_data.
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play, including def_id and off_id.

    Returns:
        pd.DataFrame: play_info with def_movement_dir and
        off_movement_dir columns added.
    """
    positions_frame = all_situations[
        ["game_id", "play_id", "nfl_id", "frame_id", "ball_frame", "x", "y"]
    ].copy()

    starting_position = (
        all_situations.groupby(["game_id", "play_id", "nfl_id"])
        .agg({"frame_id": "min", "ball_frame": "min"})
        .reset_index()
    )
    starting_position = starting_position.merge(
        positions_frame,
        on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
    )

    ending_position = (
        all_situations.groupby(["game_id", "play_id", "nfl_id"])
        .agg({"frame_id": "max", "ball_frame": "max"})
        .reset_index()
    )
    ending_position = ending_position.merge(
        positions_frame,
        on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
    )
    ending_position = ending_position.drop(columns=["frame_id", "ball_frame"])

    play_direction = starting_position.merge(
        ending_position,
        on=["game_id", "play_id", "nfl_id"],
        suffixes=("_start", "_end"),
    )
    play_direction["movement_dir"] = np.degrees(
        np.arctan2(
            play_direction["y_end"] - play_direction["y_start"],
            play_direction["x_end"] - play_direction["x_start"],
        )
    )
    play_direction.loc[play_direction["movement_dir"] < 0, "movement_dir"] += 360

    play_info = play_info.merge(
        play_direction[["game_id", "play_id", "nfl_id", "movement_dir"]],
        left_on=["game_id", "play_id", "def_id"],
        right_on=["game_id", "play_id", "nfl_id"],
        how="left",
    )
    play_info.drop(columns={"nfl_id"}, inplace=True)
    play_info.rename(columns={"movement_dir": "def_movement_dir"}, inplace=True)
    play_info = play_info.merge(
        play_direction[["game_id", "play_id", "nfl_id", "movement_dir"]],
        left_on=["game_id", "play_id", "off_id"],
        right_on=["game_id", "play_id", "nfl_id"],
        how="left",
    )
    play_info.drop(columns={"nfl_id"}, inplace=True)
    play_info.rename(columns={"movement_dir": "off_movement_dir"}, inplace=True)

    play_info = play_info[play_info["def_movement_dir"] != 0.0]

    return play_info


def add_ball_dir(all_situations: pd.DataFrame, play_info: pd.DataFrame) -> pd.DataFrame:
    """Add the ball's direction of travel at each player's first frame.

    Looks up the ball_dir recorded at the defender's and receiver's
    first tracked frame (relative to the ball's last frame) and merges
    them onto play_info.

    Args:
        all_situations (pd.DataFrame): Per-player, per-frame reach
            situations across all plays, as returned by
            get_all_situation_data.
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play, including def_id and off_id.

    Returns:
        pd.DataFrame: play_info with def_ball_dir and off_ball_dir
        columns added.
    """
    ball_direction = (
        all_situations.groupby(["game_id", "play_id", "nfl_id"])
        .agg({"frame_id": "min", "ball_frame": "max"})
        .reset_index()
    )
    ball_direction = ball_direction.merge(
        all_situations,
        on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
    )
    ball_direction = ball_direction[["game_id", "play_id", "nfl_id", "ball_dir"]]

    play_info = play_info.merge(
        ball_direction[["game_id", "play_id", "nfl_id", "ball_dir"]],
        left_on=["game_id", "play_id", "def_id"],
        right_on=["game_id", "play_id", "nfl_id"],
        how="left",
    )
    play_info.drop(columns={"nfl_id"}, inplace=True)
    play_info.rename(columns={"ball_dir": "def_ball_dir"}, inplace=True)

    play_info = play_info.merge(
        ball_direction[["game_id", "play_id", "nfl_id", "ball_dir"]],
        left_on=["game_id", "play_id", "off_id"],
        right_on=["game_id", "play_id", "nfl_id"],
        how="left",
    )
    play_info.drop(columns={"nfl_id"}, inplace=True)
    play_info.rename(columns={"ball_dir": "off_ball_dir"}, inplace=True)

    return play_info


def add_time_delta(
    play_info: pd.DataFrame, all_situations: pd.DataFrame
) -> pd.DataFrame:
    """Add the defender's and receiver's time delta relative to the ball.

    For the defender, finds the frame where their movement direction
    most closely matches the ball's direction (restricted to plays
    classified as def_move_path == "BALL") and takes that frame's
    t_delta. For the receiver, takes the t_delta at their last tracked
    frame, when the ball arrives.

    Args:
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play, including def_id, off_id, def_movement_dir,
            and def_move_path.
        all_situations (pd.DataFrame): Per-player, per-frame reach
            situations across all plays, as returned by
            get_all_situation_data.

    Returns:
        pd.DataFrame: play_info with def_t_delta and off_t_delta
        columns added.
    """
    # Calculate direction t_delta for defenders moving toward the ball
    direction_tdelta = (
        all_situations.groupby(["game_id", "play_id", "nfl_id", "ball_frame"])
        .agg({"frame_id": "min"})
        .reset_index()
    )
    direction_tdelta = direction_tdelta.merge(
        all_situations,
        on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
    )
    direction_tdelta = direction_tdelta.merge(
        play_info[
            ["game_id", "play_id", "def_id", "def_movement_dir", "def_move_path"]
        ],
        left_on=["game_id", "play_id", "nfl_id"],
        right_on=["game_id", "play_id", "def_id"],
        how="left",
    )

    direction_tdelta = direction_tdelta.dropna(subset=["def_movement_dir"])
    direction_tdelta = direction_tdelta[direction_tdelta["def_move_path"] == "BALL"]

    direction_tdelta["dir_delta"] = np.abs(
        direction_tdelta["ball_dir"] - direction_tdelta["def_movement_dir"]
    )
    direction_selection = (
        direction_tdelta.groupby(["game_id", "play_id", "nfl_id"])
        .agg({"dir_delta": "min"})
        .reset_index()
    )
    direction_tdelta = direction_selection.merge(
        direction_tdelta, on=["game_id", "play_id", "nfl_id", "dir_delta"], how="left"
    )
    direction_tdelta = direction_tdelta[
        ["game_id", "play_id", "nfl_id", "t_delta"]
    ].rename(columns={"t_delta": "def_t_delta"})

    play_info = play_info.merge(
        direction_tdelta,
        left_on=["game_id", "play_id", "def_id"],
        right_on=["game_id", "play_id", "nfl_id"],
        how="left",
    )
    play_info.drop(columns={"nfl_id"}, inplace=True)

    # Find the offensive time delta from the ball for all plays.
    offensive_tdelta = (
        all_situations.groupby(["game_id", "play_id", "nfl_id"])
        .agg({"frame_id": "min", "ball_frame": "max"})
        .reset_index()
    )
    offensive_tdelta = offensive_tdelta.merge(
        all_situations,
        on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
    )
    offensive_tdelta.rename(columns={"t_delta": "off_t_delta"}, inplace=True)
    offensive_tdelta = offensive_tdelta[["game_id", "play_id", "nfl_id", "off_t_delta"]]

    play_info = play_info.merge(
        offensive_tdelta,
        left_on=["game_id", "play_id", "off_id"],
        right_on=["game_id", "play_id", "nfl_id"],
        how="left",
    )
    play_info.drop(columns={"nfl_id"}, inplace=True)

    return play_info


def add_starting_separation(
    play_info: pd.DataFrame, all_situations: pd.DataFrame
) -> pd.DataFrame:
    """Add the defender-receiver separation at the start of ball flight.

    Args:
        play_info (pd.DataFrame): Play-level supplementary data, one
            row per play, including def_id and off_id.
        all_situations (pd.DataFrame): Per-player, per-frame reach
            situations across all plays, as returned by
            get_all_situation_data.

    Returns:
        pd.DataFrame: play_info with a start_separation column added,
        giving the defender-receiver distance in yards at the first
        tracked frame.
    """
    starting_separation = (
        all_situations.groupby(["game_id", "play_id"])
        .agg({"frame_id": "min", "ball_frame": "min"})
        .reset_index()
    )
    starting_separation = starting_separation.merge(
        play_info[["game_id", "play_id", "off_id", "def_id"]],
        on=["game_id", "play_id"],
        how="left",
    )
    starting_separation = starting_separation.merge(
        all_situations[
            ["game_id", "play_id", "nfl_id", "frame_id", "ball_frame", "x", "y"]
        ],
        left_on=["game_id", "play_id", "off_id", "frame_id", "ball_frame"],
        right_on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
    )
    starting_separation.drop(columns=["nfl_id"], inplace=True)
    starting_separation = starting_separation.merge(
        all_situations[
            ["game_id", "play_id", "nfl_id", "frame_id", "ball_frame", "x", "y"]
        ],
        left_on=["game_id", "play_id", "def_id", "frame_id", "ball_frame"],
        right_on=["game_id", "play_id", "nfl_id", "frame_id", "ball_frame"],
        how="left",
        suffixes=("_off", "_def"),
    )
    starting_separation.drop(columns=["nfl_id"], inplace=True)
    starting_separation["start_separation"] = np.sqrt(
        (starting_separation["x_off"] - starting_separation["x_def"]) ** 2
        + (starting_separation["y_off"] - starting_separation["y_def"]) ** 2
    )
    play_info = play_info.merge(
        starting_separation[["game_id", "play_id", "start_separation"]],
        on=["game_id", "play_id"],
        how="left",
    )

    return play_info


def get_animation_data(
    play_input: pd.DataFrame,
    play_output: pd.DataFrame,
    ball_position_frame: pd.DataFrame,
    play_probabilities: pd.DataFrame,
    play_row: NamedTuple,
) -> pd.DataFrame:
    """Build a single frame-by-frame DataFrame for animating a play.

    Combines pre- and post-throw player tracking, the ball's
    trajectory, and the DIP probabilities into one timeline, then
    reorients all coordinates so the offense always moves left to
    right.

    Args:
        play_input (pd.DataFrame): Pre-throw tracking data for one
            play.
        play_output (pd.DataFrame): Post-throw tracking data for one
            play.
        ball_position_frame (pd.DataFrame): The ball's interpolated
            position for the play, as returned by
            get_ball_position_frame.
        play_probabilities (pd.DataFrame): The defender's and
            receiver's DIP at each frame, as returned by
            get_play_probabilities.
        play_row (NamedTuple): A play_info row from itertuples(), used
            for def_id, off_id, play_direction, and
            absolute_yardline_number.

    Returns:
        pd.DataFrame: One row per player/ball per frame_id, with
        oriented coordinates (x_oriented, y_oriented, x_from_los,
        x_norm, y_norm) and a bal_prob column for the defender and
        receiver.
    """
    play_input = play_input[
        [
            "game_id",
            "play_id",
            "nfl_id",
            "frame_id",
            "x",
            "y",
            "play_direction",
            "absolute_yardline_number",
            "player_name",
            "player_position",
            "player_side",
        ]
    ]

    input_max_frame = play_input["frame_id"].max()
    play_output["frame_id"] = play_output["frame_id"] + input_max_frame

    full_play = pd.concat([play_input, play_output], ignore_index=True).reset_index(
        drop=True
    )
    full_play.sort_values(by=["nfl_id", "frame_id"], inplace=True)
    full_play.fillna(method="ffill", inplace=True)

    augmented_ball_frame = ball_position_frame.copy()
    augmented_ball_frame["frame_id"] = (
        augmented_ball_frame["frame_id"] + input_max_frame
    )
    augmented_ball_frame.rename(
        columns={"x_ball": "x", "y_ball": "y", "z_ball": "z"}, inplace=True
    )
    augmented_ball_frame["nfl_id"] = 0

    full_play = pd.concat(
        [full_play, augmented_ball_frame], ignore_index=True
    ).reset_index(drop=True)
    full_play["z"].fillna(0, inplace=True)

    augmented_probabilities = play_probabilities.copy()
    augmented_probabilities["frame_id"] = (
        play_probabilities["frame_id"] + input_max_frame
    )
    augmented_probabilities["def_id"] = play_row.def_id
    augmented_probabilities["off_id"] = play_row.off_id

    def_probabilities = augmented_probabilities[
        ["frame_id", "def_bal_prob", "def_id"]
    ].rename(columns={"def_bal_prob": "bal_prob", "def_id": "nfl_id"})
    off_probabilities = augmented_probabilities[
        ["frame_id", "off_bal_prob", "off_id"]
    ].rename(columns={"off_bal_prob": "bal_prob", "off_id": "nfl_id"})
    augmented_probabilities = pd.concat(
        [def_probabilities, off_probabilities], ignore_index=True
    ).reset_index(drop=True)

    full_play = full_play.merge(
        augmented_probabilities, on=["nfl_id", "frame_id"], how="outer"
    )

    # Normalize coordinates to offense moving left-to-right
    field_length = 120.0
    field_width = 53.3

    full_play["play_direction"] = full_play["play_direction"].fillna(
        play_row.play_direction
    )
    full_play["absolute_yardline_number"] = full_play[
        "absolute_yardline_number"
    ].fillna(play_row.absolute_yardline_number)

    full_play["x_oriented"] = np.where(
        full_play["play_direction"] == "right",
        full_play["x"],
        field_length - full_play["x"],
    )
    full_play["y_oriented"] = np.where(
        full_play["play_direction"] == "right",
        full_play["y"],
        field_width - full_play["y"],
    )

    full_play["yardline_oriented"] = np.where(
        full_play["play_direction"] == "right",
        full_play["absolute_yardline_number"],
        field_length - full_play["absolute_yardline_number"],
    )

    full_play["x_from_los"] = full_play["x_oriented"] - full_play["yardline_oriented"]

    full_play["x_norm"] = full_play["x_oriented"] / field_length
    full_play["y_norm"] = full_play["y_oriented"] / field_width

    full_play[["x_norm", "y_norm", "x_from_los"]] = full_play[
        ["x_norm", "y_norm", "x_from_los"]
    ].fillna(0)

    return full_play


def create_play_gif(
    full_play: pd.DataFrame,
    play_row: NamedTuple,
    field_width: float = 53.33,
    field_length: float = 120,
    output_dir: str | Path = ROOT_DIR / "gifs",
) -> None:
    """Render and save a GIF animation of a play with DIP overlays.

    Draws the field, animates each player's position colored by
    offense/defense with a DIP probability overlay where available,
    and animates the ball as a star marker.

    Args:
        full_play (pd.DataFrame): Frame-by-frame play data, as
            returned by get_animation_data.
        play_row (NamedTuple): A play_info row from itertuples(), used
            for game_id and play_id in the title and output filename.
        field_width (float): Field width, in yards. Defaults to 53.33.
        field_length (float): Field length, in yards, including end
            zones. Defaults to 120.
        output_dir (str | Path): Folder to write the GIF into. Created
            if missing. Defaults to a gifs folder at the repository root.

    Returns:
        None: Saves the animation to
        <output_dir>/play_<game_id>_<play_id>.gif.
    """
    # Prepare data
    anim_df = full_play.copy()
    anim_df = anim_df.sort_values(["frame_id", "nfl_id"])
    frames = sorted(anim_df["frame_id"].unique())

    # Split ball vs players
    ball_df = anim_df[anim_df["nfl_id"] == 0]
    players_df = anim_df[anim_df["nfl_id"] != 0]

    # Determine offense and defense players
    players_df = players_df.copy()
    players_df["is_offense"] = players_df["player_side"] == "Offense"

    # Probability normalization (0 to 1) and colormap (blue->red)
    prob_norm = Normalize(vmin=0, vmax=1)
    cmap = plt.cm.coolwarm  # blue (low) to red (high)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, field_length)
    ax.set_ylim(0, field_width)
    ax.set_xlabel("X (yards, oriented)")
    ax.set_ylabel("Y (yards, oriented)")
    ax.set_title(f"Game {play_row.game_id}, Play {play_row.play_id}")

    # Field background
    field = plt.Rectangle((0, 0), field_length, field_width, color="#0a5a26", zorder=0)
    ax.add_patch(field)
    ax.add_patch(plt.Rectangle((0, 0), 10, field_width, color="#043515", zorder=0.1))
    ax.add_patch(
        plt.Rectangle(
            (field_length - 10, 0), 10, field_width, color="#043515", zorder=0.1
        )
    )
    for x in range(10, int(field_length), 10):
        ax.axvline(x, color="white", lw=0.8, alpha=0.6, zorder=0.2)
    ax.grid(False)

    # Base player scatter (offense/defense, no probability)
    player_scat = ax.scatter([], [], s=70, edgecolors="k", linewidths=0.5, zorder=2)

    # Probability overlay (with color gradient, larger and more opaque)
    prob_scat = ax.scatter(
        [], [], s=260, marker="o", linewidths=0, alpha=0.9, cmap="coolwarm", zorder=3
    )

    # Ball scatter
    ball_scat = ax.scatter(
        [],
        [],
        s=140,
        marker="*",
        c="orange",
        edgecolors="k",
        linewidths=1.0,
        label="Ball",
        zorder=4,
    )

    # Create legend patches for offense/defense
    off_patch = plt.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="#E6E1E1",
        markersize=10,
        label="Offense",
        markeredgecolor="k",
        markeredgewidth=0.5,
    )
    def_patch = plt.Line2D(
        [],
        [],
        marker="o",
        color="w",
        markerfacecolor="#1c1c1c",
        markersize=10,
        label="Defense",
        markeredgecolor="k",
        markeredgewidth=0.5,
    )

    # Colorbar for probability gradient
    sm = ScalarMappable(cmap=cmap, norm=prob_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Ball Probability", rotation=270, labelpad=20)

    # Create legend for main elements (without colorbar)
    leg = ax.legend(
        handles=[
            off_patch,
            def_patch,
            ball_scat.get_children()[0] if ball_scat.get_children() else ball_scat,
        ],
        loc="upper left",
        frameon=True,
    )
    leg.set_zorder(10)

    def init() -> Tuple[PathCollection, PathCollection, PathCollection]:
        """Reset the player, probability, and ball scatters to empty.

        Returns:
            Tuple[PathCollection, PathCollection,
            PathCollection]: The cleared player, probability, and
            ball scatter artists.
        """
        empty_offsets = np.empty((0, 2))
        player_scat.set_offsets(empty_offsets)
        player_scat.set_facecolors([])
        prob_scat.set_offsets(empty_offsets)
        prob_scat.set_array(np.array([]))
        ball_scat.set_offsets(empty_offsets)
        return player_scat, prob_scat, ball_scat

    def update(
        frame,
    ) -> Tuple[PathCollection, PathCollection, PathCollection]:
        """Update the player, probability, and ball scatters for one frame.

        Args:
            frame (int): The frame_id to render.

        Returns:
            Tuple[PathCollection, PathCollection,
            PathCollection]: The updated player, probability, and
            ball scatter artists.
        """
        frame_players = players_df[players_df["frame_id"] == frame]
        frame_ball = ball_df[ball_df["frame_id"] == frame]

        # Base player scatter (offense/defense color, no probability
        # overlay yet)
        coords = frame_players[["x_oriented", "y_oriented"]].values
        colors = np.where(frame_players["is_offense"].values, "#E6E1E1", "#1c1c1c")
        player_scat.set_offsets(coords)
        player_scat.set_facecolors(colors)

        # Probability overlay for players with bal_prob
        prob_players = frame_players[frame_players["bal_prob"].notna()]
        prob_coords = (
            prob_players[["x_oriented", "y_oriented"]].values
            if not prob_players.empty
            else np.empty((0, 2))
        )
        prob_values = (
            prob_players["bal_prob"].values if not prob_players.empty else np.array([])
        )

        prob_scat.set_offsets(prob_coords)
        if len(prob_values) > 0:
            prob_scat.set_array(prob_values)
            prob_scat.set_norm(prob_norm)
            prob_scat.set_cmap(cmap)
        else:
            prob_scat.set_array(np.array([]))

        # Ball position
        if not frame_ball.empty:
            ball_coords = frame_ball[["x_oriented", "y_oriented"]].values
        else:
            ball_coords = np.empty((0, 2))
        ball_scat.set_offsets(ball_coords)

        ax.set_title(
            f"Game {play_row.game_id}, Play {play_row.play_id} | Frame {frame}"
        )
        return player_scat, prob_scat, ball_scat

    anim = animation.FuncAnimation(
        fig, update, frames=frames, init_func=init, blit=True, interval=100, repeat=True
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"play_{play_row.game_id}_{play_row.play_id}.gif"
    writer = animation.PillowWriter(fps=10)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    logging.info(f"Saved animation to {output_path}")


def create_qb_dip_boxplot(
    qb_first_frame: pd.DataFrame, output_dir: str | Path
) -> None:
    """Plot and save the offensive DIP distribution by quarterback.

    Renders a box-and-whisker chart of each quarterback's offensive
    DIP at the moment of the throw, limited to the ten quarterbacks
    with the most plays, and saves it as qb_rep_win_prob.png.

    Args:
        qb_first_frame (pd.DataFrame): Throw-frame DIP probabilities
            joined to the passing quarterback, with off_bal_prob and
            player_name columns (one row per play).
        output_dir (str | Path): Folder to write the PNG into. Created
            if missing.

    Returns:
        None: Saves the figure to <output_dir>/qb_rep_win_prob.png.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Limit to the ten quarterbacks with the most plays
    vis_list = (
        qb_first_frame.groupby("player_name")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .index.to_list()
    )
    qb_data = qb_first_frame[qb_first_frame["player_name"].isin(vis_list)]

    plt.figure(figsize=(14, 7))
    sns.boxplot(
        data=qb_data,
        x="player_name",
        y="off_bal_prob",
        palette="Set1",
        width=0.6,
        linewidth=2,
        fliersize=6,
    )
    plt.xlabel("Quarterback", fontsize=12, fontweight="bold")
    plt.ylabel("Offensive DIP", fontsize=12, fontweight="bold")
    plt.title(
        "Distribution of Offensive DIP by QB at Ball Throw",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()

    output_path = output_dir / "qb_rep_win_prob.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Saved figure to {output_path}")


def create_dip_logistic_fit(
    play_info_subset: pd.DataFrame, output_dir: str | Path
) -> None:
    """Plot and save the logistic fit of offensive DIP versus outcome.

    Fits and renders a logistic regression of the offensive DIP
    against the binary pass outcome (1 = completion, 0 = incompletion
    or interception) and saves it as off_mwp_log_fit.png.

    Args:
        play_info_subset (pd.DataFrame): Plays to fit, with off_bal_prob
            and outcome columns (typically high-confidence plays with a
            defender movement path other than OUT).
        output_dir (str | Path): Folder to write the PNG into. Created
            if missing.

    Returns:
        None: Saves the figure to <output_dir>/off_mwp_log_fit.png.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="darkgrid")
    g = sns.lmplot(
        x="off_bal_prob",
        y="outcome",
        data=play_info_subset,
        y_jitter=0.02,
        logistic=True,
        truncate=False,
        height=4,
        aspect=1.2,
        scatter_kws={"alpha": 0.4, "s": 25, "edgecolor": "none"},
    )
    g.set_axis_labels("Offensive DIP", "Outcome (0 = Loss, 1 = Win)")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Logistic Fit of Offensive DIP", fontsize=14)
    g.set(xlim=(0, 1), ylim=(-0.05, 1.05))

    output_path = output_dir / "off_mwp_log_fit.png"
    g.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)

    logging.info(f"Saved figure to {output_path}")
