# Visual generation pipeline flowchart

Stage-by-stage flow of `code/main.py`, which regenerates the reproducible
visuals referenced by `writeup/writeup.ipynb`.

```mermaid
flowchart TD
    A[data/input_2023_wNN.csv<br/>data/output_2023_wNN.csv<br/>data/supplementary_data.csv] --> B[get_all_data]
    B --> C[get_target_plays<br/>filter to man coverage / short routes]
    C --> D[add_to_play_info<br/>def_id, off_id, ball release/landing]
    D --> E[get_player_maxes<br/>per-player a_max / s_max]
    D --> F[get_all_situation_data<br/>per-play DIP simulation]
    E --> F
    F --> G[all_situations]
    F --> H[all_probabilities]
    D --> I[Enrichment chain:<br/>add_player_dir, add_ball_dir,<br/>add_starting_separation,<br/>classify_movement_path, outcome]
    G --> I
    H --> I
    I --> J[create_qb_dip_boxplot]
    I --> K[create_dip_logistic_fit]
    H --> L[Scenario plays loop<br/>get_augmented_output,<br/>get_ball_position_frame,<br/>get_animation_data]
    L --> M[create_play_gif]
    J --> N[writeup/qb_rep_win_prob.png]
    K --> O[writeup/off_mwp_log_fit.png]
    M --> P[writeup/play_game-id_play-id.gif x3]
    N --> Q[writeup/writeup.ipynb]
    O --> Q
    P --> Q
```

Note: the hand-made full-play GIFs (`cardinals_td.gif`, `houston_fourth_qtr_int.gif`,
`raiders_fourth_down_inc.gif`) and all `.mp4` files in `writeup/` are not produced by
this pipeline — there is no generator for them in the codebase.
