import logging
import os
import numpy as np
from dotenv import load_dotenv
import data_bowl_functions as dbf

load_dotenv(dbf.ROOT_DIR / ".env")

WRITEUP_DIR = dbf.ROOT_DIR / os.getenv("WRITEUP_DIR")
SCENARIO_PLAYS = [
    (int(game_id), int(play_id))
    for game_id, play_id in (
        pair.split(":") for pair in os.getenv("SCENARIO_PLAYS").split(",")
    )
]


def main() -> None:
    """Run the DIP pipeline and write all reproducible writeup visuals.

    Returns:
        None: Writes qb_rep_win_prob.png, off_mwp_log_fit.png, and the
        three scenario play GIFs into WRITEUP_DIR.
    """
    logging.info("Starting DIP pipeline run")

    # Load all weeks of tracking data, then narrow down to the man-coverage,
    # short-route plays that the DIP model is built around.
    input_data, output_data, play_info = dbf.get_all_data()
    input_data, output_data, play_info = dbf.get_target_plays(
        input_data, output_data, play_info
    )
    play_info = dbf.add_to_play_info(input_data, output_data, play_info)

    # Run the per-play DIP simulation loop (defender/receiver reach-time and
    # probability calculations frame-by-frame).
    max_frame = dbf.get_player_maxes(input_data)
    all_situations, all_probabilities, _ = dbf.get_all_situation_data(
        play_info, input_data, output_data, max_frame
    )

    # frame_id == 0 is the moment of throw, so this isolates the DIP
    # probabilities at release for play-level analysis.
    throw_frame_probabilities = all_probabilities[
        all_probabilities["frame_id"] == 0
    ].drop(columns=["frame_id"])
    play_info = play_info.merge(
        throw_frame_probabilities, on=["game_id", "play_id"], how="left"
    )
    play_info = dbf.add_player_dir(all_situations, play_info)
    play_info = dbf.add_ball_dir(all_situations, play_info)
    play_info = dbf.add_starting_separation(play_info, all_situations)
    play_info["def_move_path"] = np.vectorize(dbf.classify_movement_path)(
        play_info["play_direction"],
        play_info["def_ball_dir"],
        play_info["def_movement_dir"],
        5,
    )
    # Encode pass outcome as a binary target: completions are 1, everything
    # else (incompletions, interceptions) is 0.
    play_info["outcome"] = np.select(
        [
            play_info["pass_result"] == "C",
            play_info["pass_result"] == "I",
            play_info["pass_result"] == "IN",
        ],
        [1, 0, 0],
        default=0,
    )
    # Plays where the combined offense/defense ball probability exceeds 0.90
    # are the ones with enough allocated probability mass to trust for the
    # logistic fit below.
    play_info["total_bal_prob"] = play_info["off_bal_prob"] + play_info["def_bal_prob"]
    high_confidence_plays = play_info[play_info["total_bal_prob"] > 0.90]

    # Pull one QB name per play to attribute throw-frame DIP scores to a passer.
    qbs = (
        input_data[input_data["player_position"] == "QB"]
        .groupby(["game_id", "play_id"])["player_name"]
        .first()
        .reset_index()
    )
    qb_first_frame = throw_frame_probabilities.merge(
        qbs, on=["game_id", "play_id"], how="left"
    )

    dbf.create_qb_dip_boxplot(qb_first_frame, WRITEUP_DIR)
    # Exclude "OUT" movement-path plays since the defender breaking outward
    # is a different coverage dynamic than the logistic fit is modeling.
    dbf.create_dip_logistic_fit(
        high_confidence_plays[high_confidence_plays["def_move_path"] != "OUT"],
        WRITEUP_DIR,
    )

    # Render GIFs for the hand-picked scenario plays referenced in the writeup.
    for game_id, play_id in SCENARIO_PLAYS:
        try:
            play_row = next(
                play_info[
                    (play_info["game_id"] == game_id)
                    & (play_info["play_id"] == play_id)
                ].itertuples()
            )
            play_input = input_data[
                (input_data["game_id"] == game_id) & (input_data["play_id"] == play_id)
            ]
            play_output = output_data[
                (output_data["game_id"] == game_id)
                & (output_data["play_id"] == play_id)
            ]
            augmented_output = dbf.get_augmented_output(
                input_data, output_data, play_row
            )
            ball_position_frame = dbf.get_ball_position_frame(
                augmented_output, play_row
            )
            play_probabilities = all_probabilities[
                (all_probabilities["game_id"] == game_id)
                & (all_probabilities["play_id"] == play_id)
            ]
            full_play = dbf.get_animation_data(
                play_input, play_output, ball_position_frame, play_probabilities, play_row
            )
            dbf.create_play_gif(full_play, play_row, output_dir=WRITEUP_DIR)
        except Exception as e:
            logging.error(
                f"Failed to render scenario GIF for game_id={game_id}, "
                + f"play_id={play_id}: {e}"
            )
            continue

    logging.info("DIP pipeline run complete")


if __name__ == "__main__":
    main()
